import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from einops import rearrange
from src.vq import VectorQuantizer, VectorQuantizerEMA
from src.hpenalty import hessian_penalty
from src.utils import compute_eigen
from geomstats.geometry.hypersphere import Hypersphere



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SlotAttention(nn.Module):
    def __init__(self, num_slots, 
                        dim, 
                        iters = 5, 
                        eps = 1e-8, 
                        hidden_dim = 128,
                        max_slots=64, 
                        nunique_slots=8,
                        temp=1,
                        beta=0.75,
                        encoder_intial_res=(8, 8),
                        decoder_intial_res=(8, 8),
                        quantize=False,
                        cosine=False,
                        cb_decay=0.99,
                        cb_querykey=False,
                        eigen_quantizer=False,
                        cb_variational=False,
                        cov_binarize=False,
                        no_position=True,
                        restart_cbstats=False
                        ):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.dim = dim
        self.scale = dim ** -0.5
        self.temperature = temp
        self.cb_querykey = cb_querykey
        self.eigen_quantizer = eigen_quantizer
        self.restart_cbstats = restart_cbstats
        ntokens = np.prod(encoder_intial_res)
        
        self.nunique_slots = nunique_slots
        self.quantize = quantize
        self.cov_binarize = cov_binarize
        self.no_position = no_position

        assert self.num_slots < np.prod(encoder_intial_res), f'reduce number of slots, max possible {np.prod(encoder_intial_res)}'

        # ===========================================
        # encoder postional embedding with linear transformation
        self.encoder_position = SoftPositionEmbed(dim, encoder_intial_res)
        self.encoder_norm = nn.LayerNorm([ntokens, dim])
        self.encoder_feature_mlp = nn.Sequential(nn.Linear(dim, dim),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(dim, dim),
                                                nn.ReLU(inplace=True))


        self.positional_encoder = PositionalEncoding(ntokens, dim)
        self.slot_transformation = nn.Sequential(nn.Linear(dim, dim),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(dim, dim),
                                                nn.ReLU(inplace=True))



        # # decoder positional embeddings
        self.decoder_position    = SoftPositionEmbed(dim, decoder_intial_res)
        self.decoder_initial_res = decoder_intial_res

        # ===========================================

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, dim))


        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)



        if self.no_position:
            self.to_k_np = nn.Linear(dim, dim)
            self.encoder_norm_np = nn.LayerNorm([ntokens, dim])
            self.encoder_feature_mlp_np = nn.Sequential(nn.Linear(dim, dim),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(dim, dim),
                                                nn.ReLU(inplace=True))

        # ===================================
        if self.quantize:
            if cb_decay > 0.0:
                self.slot_quantizer = VectorQuantizerEMA(max_slots,
                                                    self.dim,
                                                    commitment_cost=beta,
                                                    decay=cb_decay,
                                                    cosine=cosine,
                                                    variational=cb_variational,
                                                    qk=self.cb_querykey)
                print ('VQVAE MODEL EMA', cb_decay)
            else:
                self.slot_quantizer = VectorQuantizer(max_slots,
                                                    self.dim,
                                                    commitment_cost=beta,
                                                    cosine=cosine,
                                                    variational=cb_variational,
                                                    qk=self.cb_querykey)
                print ('VQVAE model', cb_decay)
        else:         
            self.slots_mu    = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, 1, dim)))
            self.slots_sigma = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, 1, dim)))



    def encoder_transformation(self, features, position=True):
        #features: B x C x Wx H
        features = features.permute(0, 2, 3, 1)
        if position:
            features = self.encoder_position(features)
            features = torch.flatten(features, 1, 2)
            features = self.encoder_norm(features)
            features = self.encoder_feature_mlp(features)
        else:
            features = torch.flatten(features, 1, 2)
            features = self.encoder_norm_np(features)
            features = self.encoder_feature_mlp_np(features)
        return features


    def decoder_transformation(self, slots):
        # features: B x nslots x dim
        slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        features = slots.repeat((1, self.decoder_initial_res[0], self.decoder_initial_res[1], 1))
        features = self.decoder_position(features)
        return features.permute(0, 3, 1, 2)



    def passthrough_eigen_basis(self, x):
        # x: token embeddings B x ntokens x token_embeddings

        x = F.normalize(x, p=2, dim=-1)
        coveriance_matrix = torch.einsum('bkf, bgf -> bkg', x, x)

        if self.cov_binarize:
            # apply softmax on token dimension 
            # feats = torch.softmax(feats, dim = 1)
            # feats = feats / feats.sum(dim=-1, keepdim=True)
            
            coveriance_matrix = torch.sigmoid(coveriance_matrix)
            coveriance_matrix[coveriance_matrix >= 0.5] = 1
            coveriance_matrix[coveriance_matrix < 0.5] = 0


        # eigen vectors are arranged in ascending order of their eigen values, so picking last n objects
        eigen_values, eigen_vectors = torch.symeig(coveriance_matrix, eigenvectors=True)

        # eigen_vectors = eigen_vectors[:, :, -self.nunique_slots+1:].permute(0, 2, 1)
        # eigen_values = eigen_values[:, -self.nunique_slots+1:]

        eigen_vectors = eigen_vectors.permute(0, 2, 1)
        eigen_values = eigen_values
        
        eigen_vectors = eigen_vectors.softmax(dim = -1) + self.eps
        # eigen_vectors = eigen_vectors/torch.sum(eigen_vectors, dim =-1, keepdim=True)

        eigen_vectors = torch.flip(eigen_vectors, [1])
        eigen_values = torch.flip(eigen_values, [1])
        return eigen_vectors, eigen_values


    def extract_eigen_basis(self, features, batch=None):
        batched_concepts = []
        batched_scale = []

        shape = features.shape


        for i, feature in enumerate(features):            
            eigen_vectors, eigen_values = compute_eigen(feature, 
                                        image = None if batch is None else batch[i],
                                        K = shape[1],# self.nunique_slots+1, 
                                        which_matrix = 'matting_laplacian',
                                        normalize  = True,
                                        binarize = self.cov_binarize,
                                        lapnorm = True,
                                        threshold_at_zero = True,
                                        image_color_lambda = 0 if batch is None else 10)

            batched_concepts.append(eigen_vectors.unsqueeze(0))
            batched_scale.append(eigen_values.unsqueeze(0))
        
        batched_scale = torch.cat(batched_scale, 0).to(features.device)
        batched_concepts = torch.cat(batched_concepts, 0).to(features.device)
        
        batched_concepts = batched_concepts.softmax(dim = -1) + self.eps
        # batched_concepts = batched_concepts/torch.sum(batched_concepts, dim =-1, keepdim=True)

        batched_concepts = torch.flip(batched_concepts, [1])
        batched_scale = torch.flip(batched_scale, [1])
        return batched_concepts, batched_scale


    def masked_projection(self, features, z):
        # features: B x nanchors x f
        # z: basis B x K x nanchors

        # b, n, f = features.shape
        # k = z.shape[1]
        # features = features.unsqueeze(1).repeat(1, k, 1, 1) # B x k x n x f
        # z = z.unsqueeze(-1)
        # z = z.repeat(1, 1, 1, f)# B x k x n x f

        # projection = features*z
        # return torch.sum(projection, 2)
        return torch.bmm(z, features)



    def forward(self, inputs, num_slots = None, epoch=0, batch= 0, train = True, images=None):
        b, d, w, h = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots        
        
        qloss = torch.Tensor([0]).to(inputs.device)
        cbidxs = torch.Tensor([[0]]).to(inputs.device)
        perplexity = torch.Tensor([[0]]).to(inputs.device)

        # ========================
        inputs_features = self.encoder_transformation(inputs, position = True)
        inputs_features = self.norm_input(inputs_features)


        k = self.to_k(inputs_features)
        v = self.to_v(inputs_features)



        if self.quantize:

            # with torch.no_grad():
            if self.no_position:
                inputs_features_noposition = self.encoder_transformation(inputs, position = False)
                k_noposition =  self.to_k_np(inputs_features_noposition)
            else:
                inputs_features_noposition = inputs_features
                k_noposition = k
            # k_noposition = k + (k_noposition - k).detach() 



            if self.eigen_quantizer:
                # eigen_basis, eigen_values = self.passthrough_eigen_basis(k)
                eigen_basis, eigen_values = self.extract_eigen_basis(k_noposition, batch=images)
                objects = self.masked_projection(k_noposition, eigen_basis)   

                # context loss
                qloss += F.mse_loss(objects.mean(1), k_noposition.mean(1))

                qloss1, _, _, _, _ = self.slot_quantizer(objects, 
                                                    avg = False, 
                                                    unique=True,
                                                    nunique=self.nunique_slots +1,
                                                    loss_type = 0,
                                                    update = self.restart_cbstats,
                                                    reset_usage = (batch == 0))
                qloss += qloss1 


                # sample objects
                objects, cbidxs, _, slots = self.slot_quantizer.sample(k_noposition)
                                                            # , 
                                                            # unique=True,
                                                            # nunique=self.nunique_slots +1)

            else:
                qloss, objects, perplexity, cbidxs, slots = self.slot_quantizer(k_noposition, 
                                                    avg = False, 
                                                    unique=False,
                                                    loss_type = 0,
                                                    update = self.restart_cbstats,
                                                    reset_usage = (batch == 0))
            



            # if self.restart_cbstats and (epoch % 5 == 4) and (batch == 0) and (epoch < 25): self.slot_quantizer.entire_cb_restart()

            if self.cb_querykey: slots = slots[:, :n_s, :]
            else: slots = objects[:, :n_s, :]
            
            cbidxs = cbidxs[:, :n_s]
        else:
            mu = self.slots_mu.expand(b, n_s, -1)
            sigma = torch.exp(0.5*self.slots_sigma.expand(b, n_s, -1))
            slots = torch.normal(mu, sigma)
        


        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)


            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            # slots = self.positional_encoder(slots)
            slots = slots + self.mlp(self.norm_pre_ff(slots))



        return slots, cbidxs, qloss, perplexity, self.decoder_transformation(slots)


def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)


"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid


class PositionalEncoding(nn.Module):

    def __init__(self, max_len, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model), requires_grad=True)
        nn.init.trunc_normal_(self.pe)

    def forward(self, input):
        """
        input: batch_size x seq_len x d_model
        return: batch_size x seq_len x d_model
        """
        T = input.shape[1]
        return self.dropout(input + self.pe[:, :T])

class Encoder(nn.Module):
    def __init__(self, resolution, hid_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hid_dim, 5, stride=2, padding = 2)
        # self.conv11 = nn.Conv2d(hid_dim, hid_dim, 5,  stride=2, padding = 2)

        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5,  stride=2, padding = 2)
        # self.conv21 = nn.Conv2d(hid_dim, hid_dim, 5,  stride=2, padding = 2)

        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, stride=2, padding = 2)
        # self.conv31 = nn.Conv2d(hid_dim, hid_dim, 5,  stride=2, padding = 2)

        self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, stride=1, padding = 2)
        # self.conv41 = nn.Conv2d(hid_dim, hid_dim, 5,  stride=2, padding = 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        # x = self.conv11(x)
        # x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        # x = self.conv21(x)
        # x = F.relu(x)


        x = self.conv3(x)
        x = F.relu(x)

        # x = self.conv31(x)
        # x = F.relu(x)


        x = self.conv4(x)
        x = F.relu(x)

        # x = self.conv41(x)
        # x = F.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, hid_dim, resolution):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2).to(device)
        self.conv6 = nn.ConvTranspose2d(hid_dim, 4, 3, stride=(1, 1), padding=1)

        self.resolution = resolution

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        # x = F.relu6(x)
        x = x[:,:,:self.resolution[0], :self.resolution[1]]
        return x


"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution, 
                        num_slots, 
                        num_iterations, 
                        hid_dim, 
                        max_slots=64, 
                        nunique_slots=10,
                        quantize=False,
                        cosine=False, 
                        unique_samling=False, 
                        cb_decay=0.99,
                        encoder_res=4,
                        decoder_res=4,
                        variational=False, 
                        binarize=False,
                        no_position=True,
                        cb_qk=False,
                        eigen_quantizer=False,
                        restart_cbstats=False
                        ):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.encoder_cnn = Encoder(self.resolution, self.hid_dim)
        self.decoder_cnn = Decoder(self.hid_dim, self.resolution)


        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=hid_dim,
            iters = self.num_iterations,
            eps = 1e-8, 
            hidden_dim = 128,
            nunique_slots=nunique_slots,
            quantize = quantize,
            max_slots=max_slots,
            cosine=cosine,
            cb_decay=cb_decay,
            encoder_intial_res=(encoder_res, encoder_res),
            decoder_intial_res=(decoder_res, decoder_res),
            cb_variational=variational,
            cov_binarize=binarize,
            no_position=no_position,
            cb_querykey=cb_qk,
            eigen_quantizer=eigen_quantizer,
            restart_cbstats=restart_cbstats)


    def forward(self, image, num_slots=None, epoch=0, batch=0):
        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        # `x` has shape: [batch_size, input_size, width, height].

        # Slot Attention module.
        slots, cbidxs, qloss, perplexity, features = self.slot_attention(x, 
                                                                        num_slots, 
                                                                        epoch, 
                                                                        batch, 
                                                                        images=image)
        # `slots` has shape: [batch_size, num_slots, slot_size].
        # `features` has shape: [batch_size*num_slots, width_init, height_init, slot_size]

        x = self.decoder_cnn(features)
        x = x.permute(0, 2, 3, 1)
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(image.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([3,1], dim=-1)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0,3,1,2)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].

        return recon_combined, recons, masks, slots, cbidxs, qloss, perplexity