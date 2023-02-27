import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from einops import rearrange
from src.vq import VectorQuantizer 
from src.legacy_vq import VectorQuantizer as legacy_quantizer
from src.hpenalty import hessian_penalty
from src.utils import compute_eigen, uniform_init
from geomstats.geometry.hypersphere import Hypersphere
from einops import rearrange, repeat


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def reorder_slots(slots, slots_mu, cidxs, scales = None, ns=10):
    # eigenvalues in decreasing order
    # cidxs are ordered wrt eigenvalues

    B, N = cidxs.shape
    if ns > N:
        orig_slots = slots.clone()
        orig_slots_mu = slots_mu.clone()
        orig_idxs = cidxs.clone()

        counter = 1
        while cidxs.shape[1] < ns:
            nunique_objects = -1
            if not (scales is None):
                nunique_objects = int((1.0*(scales > counter).sum(1)).max().item())

            start_idx = 0
            if nunique_objects > 1:
                start_idx = 1
            
            slots = torch.cat([slots, orig_slots[:, start_idx:nunique_objects, :]], 1)
            slots_mu = torch.cat([slots_mu, orig_slots_mu[:, start_idx:nunique_objects, :]], 1)
            cidxs = torch.cat([cidxs, orig_idxs[:, start_idx:nunique_objects]], 1)
            
            counter += 1


    slots, slots_mu, cidxs = slots[:, :ns, :], slots_mu[:, :ns, :], cidxs[:, :ns]
    
    return slots, slots_mu, cidxs


class SlotAttention(nn.Module):
    def __init__(self, num_slots, 
                        dim, 
                        iters = 3, 
                        eps = 1e-8, 
                        hidden_dim = 128,
                        max_slots=64, 
                        nunique_slots=8,
                        beta=0.0,
                        encoder_intial_res=(8, 8),
                        decoder_intial_res=(8, 8),
                        quantize=False,
                        cosine=False,
                        cb_decay=0.8,
                        cb_querykey=False,
                        eigen_quantizer=False,
                        restart_cbstats=False,
                        implicit=True,
                        gumble=False,
                        temperature=2.0,
                        kld_scale=1.0
                        ):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.dim = dim
        self.scale = dim ** -0.5
        self.implicit = implicit
        self.cb_querykey = cb_querykey
        self.eigen_quantizer = eigen_quantizer
        self.restart_cbstats = restart_cbstats
        ntokens = np.prod(encoder_intial_res)
        
        self.max_slots = max_slots
        self.nunique_slots = nunique_slots
        self.quantize = quantize
        self.min_number_elements = 2
        self.beta = beta
        legacy = True

        assert self.num_slots <= np.prod(encoder_intial_res), f'reduce number of slots, max possible {np.prod(encoder_intial_res)}'

        # ===========================================
        # encoder postional embedding with linear transformation
        self.encoder_position = SoftPositionEmbed(dim, encoder_intial_res)
        self.encoder_norm = nn.LayerNorm([ntokens, dim])
        self.encoder_feature_mlp = nn.Sequential(nn.Linear(dim, dim),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(dim, dim),
                                                nn.ReLU(inplace=True))


        self.slot_transformation = nn.Sequential(nn.Linear(dim, dim),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(dim, dim),
                                                nn.ReLU(inplace=True))



        # # decoder positional embeddings
        self.decoder_position    = SoftPositionEmbed(dim, decoder_intial_res)
        self.decoder_initial_res = decoder_intial_res

        # ===========================================

        self.to_q = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        self.norm_input  = nn.LayerNorm(dim)



        self.to_k_np = nn.Linear(dim, dim)
        self.norm_input_np  = nn.LayerNorm(dim)
        self.encoder_norm_np = nn.LayerNorm([ntokens, dim])
        self.encoder_feature_mlp_np = nn.Sequential(nn.Linear(dim, dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(dim, dim),
                                            nn.ReLU(inplace=True))

        # ===================================
        if self.quantize:
            if not legacy:
                self.slot_quantizer = VectorQuantizer(
                                                self.dim, # ntokens,
                                                max_slots,
                                                codebook_dim = 8, # self.dim, # ntokens,
                                                nhidden = self.dim,
                                                decay = cb_decay,                                                
                                                kmeans_init = True,
                                                kmeans_iters = 10,
                                                use_cosine_sim = cosine,
                                                qk_codebook = self.cb_querykey,
                                                sample_codebook_temp = temperature,
                                                commitment_weight = 0.0,
                                              )
                print ('VQVAE model', cb_decay, cosine, self.dim)
            else:
                self.slot_quantizer = legacy_quantizer(num_embeddings = max_slots, 
                                                        embedding_dim = self.dim, # ntokens,
                                                        codebook_dim = 32,
                                                        nhidden = self.dim,
                                                        commitment_cost = self.beta,
                                                        decay = cb_decay,
                                                        qk=self.cb_querykey,
                                                        cosine= cosine,
                                                        gumble=gumble,
                                                        temperature=temperature
                                                        )
        else:         
            self.slots_mu    = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, 1, dim)))
            self.slots_sigma = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, 1, dim)))
            self.slots_mu.requires_grad = True
            self.slots_sigma.requires_grad = True



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


    @torch.no_grad()
    def passthrough_eigen_basis(self, x):
        # x: token embeddings B x ntokens x token_embeddings

        x = F.normalize(x, p=2, dim=-1)
        coveriance_matrix = torch.einsum('bkf, bgf -> bkg', x, x)

        # eigen vectors are arranged in ascending order of their eigen values
        # eigen_values, eigen_vectors = torch.symeig(coveriance_matrix, eigenvectors=True)
        eigen_values, eigen_vectors = torch.linalg.eigh(coveriance_matrix)

        eigen_vectors = eigen_vectors.permute(0, 2, 1)        

        eigen_vectors = torch.flip(eigen_vectors, [1])
        eigen_values = torch.flip(eigen_values, [1])

        # Sign ambiguity
        for k in range(eigen_vectors.shape[0]):
            mean_vector = torch.mean((eigen_vectors[:, k] > 0).float(), -1)
            idxs = (mean_vector > 0.5)*(mean_vector < 1.0)
            # reverse segment
            eigen_vectors[idxs, k] = 0 - eigen_vectors[idxs, k]

        return eigen_vectors, eigen_values


    @torch.no_grad()
    def svd_spectral_decomposition(self, x):
        # x: token embeddings B x ntokens x token_embeddings

        x = F.normalize(x, p=2, dim=-1)

        u, s, v = torch.linalg.svd(x)
        
        print(x.shape, u.shape, s.shape, v.shape, '========================')
        import pdb;pdb.set_trace()


    @torch.no_grad()
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
                                        image_color_lambda = 0 if batch is None else 1.0)

            batched_concepts.append(eigen_vectors.unsqueeze(0))
            batched_scale.append(eigen_values.unsqueeze(0))
        
        batched_scale = torch.cat(batched_scale, 0).to(features.device)
        batched_concepts = torch.cat(batched_concepts, 0).to(features.device)
        
        batched_concepts = batched_concepts.softmax(dim = -1) 

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


    def feature_abstraction(self, inputs):
        # Compute Principle directions and scale
        eigen_basis, eigen_values = self.passthrough_eigen_basis(inputs.clone().detach())
        # eigen_basis, eigen_values = self.svd_spectral_decomposition(inputs.clone().detach())
        # eigen_basis, eigen_values = self.extract_eigen_basis(inputs, batch=images)

        eigen_values = torch.round(eigen_values)
        nunique_objects = max(3, int((1.0*(eigen_values > 0).sum(1)).max().item()))

        # Principle components
        objects = self.masked_projection(inputs, eigen_basis)   
        objects = objects[:, :nunique_objects, :]
        return objects, eigen_values[:, :nunique_objects]


    def sample_quantized_slots(self, n_s, k, MCsamples = 1, epoch = 0, batch = 0, images=None):

        if self.eigen_quantizer: objects, scales = self.feature_abstraction(k)                       
        else: objects = k; scales = torch.ones_like(k)
        
        # Quantizing components----
        qobjects, cbidxs, qloss, perplexity, slots  = self.slot_quantizer(objects.clone().detach(), 
                                                                    loss_type = 1,
                                                                    MCsamples = MCsamples,
                                                                    update = self.restart_cbstats,
                                                                    reset_usage = (batch == 0))
        
        if self.cb_querykey: 
            slots, qobjects, cbidxs = reorder_slots(slots, qobjects, cbidxs, scales, n_s)
            slots = slots.reshape(k.shape[0], MCsamples, -1, slots.shape[-1])
        else: 
            qobjects = objects[:, :n_s, :]
            slots = qobjects.clone().unsqueeze(1); cbidxs = cbidxs[:, :n_s]

        return slots, qobjects, qloss, perplexity, cbidxs


    def sample_baseline_slots(self, inputs, n_s, b):
        qloss = torch.Tensor([0]).to(inputs.device)
        cbidxs = torch.Tensor([[0]*n_s]*b).to(inputs.device)
        perplexity = torch.Tensor([[0]]).to(inputs.device)

        slot_mu = self.slots_mu.expand(b, n_s, -1)
        slot_sigma = self.slots_sigma.expand(b, n_s, -1)
        slot_sigma = torch.exp(0.5*slot_sigma)

        slots = slot_mu + slot_sigma * torch.randn(slot_sigma.shape, 
                                                device = slot_sigma.device, 
                                                dtype = slot_sigma.dtype)

        # slots = torch.normal(slot_mu, slot_sigma)

        # add MC axis
        slots = slots.unsqueeze(1)
        return slots, slot_mu.clone(), qloss, perplexity, cbidxs


    def step(self, slots_prev, k , v, MCsamples, ns, b):
        slots = self.norm_slots(slots_prev)
        q = self.to_q(slots)

        dots = torch.einsum('bmid,bjd->bmij', q, k) * self.scale
        attn = dots.softmax(dim=2) + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum('bjd,bmij->bmid', v, attn)

        slots = self.gru(
            rearrange(updates, 'b m n d -> (b m n) d'),
            rearrange(slots_prev, 'b m n d -> (b m n) d')
        )

        slots = slots.reshape(b, MCsamples, ns, self.dim)
        slots = slots + self.slot_transformation(self.norm_pre_ff(slots))
        return slots


    def forward(self, inputs, 
                    num_slots = None,
                    MCsamples = 1, 
                    epoch=0, batch= 0, 
                    train = True, 
                    images=None):

        b, d, w, h = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots        

        # Compute Projections ========================
        inputs_features = self.encoder_transformation(inputs, position = True)
        inputs_features = self.norm_input(inputs_features)


        # Sample Slots ========================
        if self.quantize:
            inputs_features_np = self.encoder_transformation(inputs, position = False)
            inputs_features_np = self.norm_input_np(inputs_features_np)
            k_np = self.to_k_np(inputs_features_np)

            
            slots, objects, qloss, perplexity, cbidxs = self.sample_quantized_slots(n_s = n_s, 
                                                                        MCsamples = MCsamples,
                                                                        k = k_np, 
                                                                        epoch = epoch, 
                                                                        batch = batch, 
                                                                        images = images)


        else:
            slots, objects, qloss, perplexity, cbidxs = self.sample_baseline_slots(inputs, n_s, b)


        # Key-Value projection vectors ====================
        k = self.to_k(inputs_features)
        v = self.to_v(inputs_features)

        # Slot attention =========================
        for _ in range(self.iters):
            slots = self.step(slots, k, v, MCsamples, n_s, b)

        if self.implicit: slots = self.step(slots.detach(), k, v)


        # update slots ===================
        # if self.quantize and self.cb_querykey:
        #     qloss += F.mse_loss(objects, slots.detach().mean(1))
            
        return slots, cbidxs, qloss, perplexity, self.decoder_transformation(slots)



    @torch.no_grad()
    def given_idxs(self, inputs, 
                        slot_idxs,
                        MCsamples = 1,
                        images=None):
        b, d, w, h = inputs.shape
        n_s = slot_idxs.shape[1]

        # Compute Projections ========================
        inputs_features = self.encoder_transformation(inputs, position = True)
        inputs_features = self.norm_input(inputs_features)

        # sample slots ==============================
        qloss = torch.Tensor([0]).to(inputs.device)
        perplexity = torch.Tensor([[0]]).to(inputs.device)

        shape = slot_idxs.shape
        encodings = torch.zeros(np.prod(shape), self.max_slots, device=inputs.device)
        encodings.scatter_(1, slot_idxs.reshape(-1, 1), 1)

        slots = self.slot_quantizer.qkclass.sample_slots(encodings, (b, n_s, self.dim))
        slots = slots.view(b, MCsamples, n_s, -1)

        # # Key-Value projection vectors ====================
        # k = self.to_k(inputs_features)
        # v = self.to_v(inputs_features)

        # # Slot attention =========================
        # for _ in range(self.iters):
        #     slots = self.step(slots, k, v, MCsamples, n_s, b)

        # if self.implicit: slots = self.step(slots.detach(), k, v)

        return slots, slot_idxs, qloss, perplexity, self.decoder_transformation(slots)




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
    def __init__(self, resolution, hid_dim, kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hid_dim, kernel_size, stride=2, padding = 2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, kernel_size,  stride=2, padding = 2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, kernel_size, stride=2, padding = 2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, kernel_size, stride=1, padding = 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        return x


class Decoder(nn.Module):
    def __init__(self, hid_dim, resolution, kernel_size=5):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 3, stride=1, padding=1).to(device)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, kernel_size, stride=2, padding=2, output_padding=1).to(device)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, kernel_size, stride=2, padding=2, output_padding=1).to(device)
        self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, kernel_size, stride=2, padding=2, output_padding=1).to(device)
        self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, kernel_size, stride=1, padding=2).to(device)
        self.conv6 = nn.ConvTranspose2d(hid_dim, 4, 3, stride=1, padding=1)

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
                        cb_decay=0.99,
                        encoder_res=4,
                        decoder_res=4,
                        kernel_size=5,
                        cb_qk=False,
                        eigen_quantizer=False,
                        restart_cbstats=False,
                        implicit=False,
                        gumble=False,
                        temperature=1.0,
                        kld_scale=1.0
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
        self.quantize = quantize
        
        self.encoder_cnn = Encoder(self.resolution, self.hid_dim, kernel_size)
        self.decoder_cnn = Decoder(self.hid_dim, self.resolution, kernel_size)


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
                                    cb_querykey=cb_qk,
                                    eigen_quantizer=eigen_quantizer,
                                    restart_cbstats=restart_cbstats,
                                    implicit=implicit,
                                    gumble=gumble,
                                    temperature=temperature,
                                    kld_scale=kld_scale)


    def forward(self, image, 
                    num_slots=None, 
                    MCsamples = 1,
                    epoch=0, batch=0):

        n_s = num_slots if num_slots is not None else self.num_slots   
        MCsamples = MCsamples if self.quantize else 1  

        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        # `x` has shape: [batch_size, input_size, width, height].

        # Slot Attention module.
        slots, cbidxs, qloss, perplexity, features = self.slot_attention(x, 
                                                                        num_slots,
                                                                        MCsamples = MCsamples, 
                                                                        epoch = epoch, 
                                                                        batch = batch, 
                                                                        images=image)
        # `slots` has shape: [batch_size, MCsamples, num_slots, slot_size].
        # `features` has shape: [batch_size*MCsamples*num_slots, width_init, height_init, slot_size]

        x = self.decoder_cnn(features)
        x = x.permute(0, 2, 3, 1)
        # `x` has shape: [batch_size*MCsamples*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(image.shape[0], MCsamples, n_s, x.shape[1], x.shape[2], x.shape[3]).split([3,1], dim=-1)
        # `recons` has shape: [batch_size, MCsamples, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, MCsamples, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=2)(masks)
        recon_combined = torch.sum(recons * masks, dim=2)  # Recombine image.

        # Average over MC samples.....
        recon_combined = recon_combined.mean(1); slots = slots.mean(1); 
        recons = recons.mean(1); masks = masks.mean(1)

        recon_combined = recon_combined.permute(0, 3, 1, 2)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].

        return recon_combined, recons, masks, slots, cbidxs, qloss, perplexity


    def given_idxs(self, image, slot_idxs=None, 
                                    MCsamples = 1):
        if slot_idxs is None:
            return self.forward(image)
        

        n_s = slot_idxs.shape[1]  
        MCsamples = MCsamples if self.quantize else 1  

        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        # `x` has shape: [batch_size, input_size, width, height].

        # Slot Attention module.
        slots, cbidxs, qloss, perplexity, features = self.slot_attention.given_idxs(x, 
                                                                                slot_idxs,
                                                                                MCsamples = MCsamples,
                                                                                images=image)
        # `slots` has shape: [batch_size, MCsamples, num_slots, slot_size].
        # `features` has shape: [batch_size*MCsamples*num_slots, width_init, height_init, slot_size]

        x = self.decoder_cnn(features)
        x = x.permute(0, 2, 3, 1)
        # `x` has shape: [batch_size*MCsamples*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(image.shape[0], MCsamples, n_s, x.shape[1], x.shape[2], x.shape[3]).split([3,1], dim=-1)
        # `recons` has shape: [batch_size, MCsamples, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, MCsamples, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=2)(masks)
        recon_combined = torch.sum(recons * masks, dim=2)  # Recombine image.

        # Average over MC samples.....
        recon_combined = recon_combined.mean(1); slots = slots.mean(1); 
        recons = recons.mean(1); masks = masks.mean(1)

        recon_combined = recon_combined.permute(0, 3, 1, 2)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].

        return recon_combined, recons, masks, slots, cbidxs, qloss, perplexity



class SetPredictor(nn.Module):
    def __init__(self,
                    hid_dim,
                    nproperties):
        super().__init__()

        self.nproperties = nproperties
        self.mlp_classifier = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hid_dim, self.nproperties),
                                        nn.Sigmoid())        

    def forward(self, x, epoch=0, batch=0):
        return self.mlp_classifier(x)



class SlotAttentionClassifier(nn.Module):
  """Slot Attention-based classifier for property prediction."""

  def __init__(self,  resolution, 
                        num_slots, 
                        num_iterations, 
                        hid_dim, 
                        nproperties,
                        max_slots=64, 
                        nunique_slots=10,
                        quantize=False,
                        cosine=False, 
                        cb_decay=0.99,
                        encoder_res=4,
                        decoder_res=4,
                        kernel_size=5,
                        cb_qk=False,
                        eigen_quantizer=False,
                        restart_cbstats=False,
                        implicit=True,
                        gumble=False,
                        temperature=1.0,
                        kld_scale=1.0
                        ):

    """Builds the Slot Attention-based classifier.
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
    self.quantize = quantize

    # nclasses: numpber of nodes as outputs
    # In case of CLEVR: (coords=3) + (color=8) + (size=2) + (material=2) + (shape=3) + (real=1) = 19
    self.nproperties = nproperties

    self.encoder_cnn = Encoder(self.resolution, self.hid_dim, kernel_size)
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
                                cb_querykey=cb_qk,
                                eigen_quantizer=eigen_quantizer,
                                restart_cbstats=restart_cbstats,
                                implicit=implicit,
                                gumble=gumble,
                                temperature=temperature,
                                kld_scale=kld_scale)


    self.mlp_classifier = SetPredictor(hid_dim, self.nproperties)
    
    

  def forward(self, image, 
                    num_slots=None, 
                    MCsamples = 5,
                    epoch=0, batch=0):

    n_s = num_slots if num_slots is not None else self.num_slots   
    MCsamples = MCsamples if self.quantize else 1  

    # `image` has shape: [batch_size, width, height, num_channels].

    # Convolutional encoder with position embedding.
    x = self.encoder_cnn(image)  # CNN Backbone.
    # `x` has shape: [batch_size, input_size, width, height].


    
    # Slot Attention module.
    slots, cbidxs, qloss, perplexity, features = self.slot_attention(x, 
                                                                num_slots,
                                                                MCsamples = MCsamples, 
                                                                epoch = epoch, 
                                                                batch = batch, 
                                                                images=image)
    # `slots` has shape: [batch_size, MCsamples, num_slots, slot_size].
    # `features` has shape: [batch_size*num_slots, width_init, height_init, slot_size]

    # MC flatten
    slots = rearrange(slots, 'b m n d -> (b m) n d')
    predictions = self.mlp_classifier(slots)

    slots = slots.view(image.shape[0], MCsamples, n_s, -1)
    slots = slots.mean(1)
    return predictions, cbidxs, qloss, perplexity



class ReasoningAttention(nn.Module):
    def __init__(self, nconcepts, cdim):
        super().__init__()

        self.nconcepts = nconcepts
        self.edim = cdim

        self.attention_layer = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.nconcepts, self.edim)))


    def forward(self, cb, sampled_idx):
        #z_q: sampled K eign vectors
        cb = cb.unsqueeze(0).repeat(sampled_idx.shape[0], 1, 1)
        mask = torch.zeros_like(cb)

        for i, idx in enumerate(sampled_idx):
            for __idx__ in idx:
                mask[i, int(__idx__), :] = 1

        sampled_cb = cb*mask
        attention_vector = torch.sum(sampled_cb*self.attention_layer.weight.unsqueeze(0), 2)
        return attention_vector



class SlotAttentionReasoning(nn.Module):
    def __init__(self,
                    slot_dim, 
                    max_slots,
                    nproperties, 
                    nclasses,
                    hid_dim=64
                    ):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.nclasses = nclasses
        self.max_slots = max_slots

        self.fc1_w = nn.Parameter(uniform_init(max_slots, slot_dim, hid_dim))
        self.fc1_b = nn.Parameter(uniform_init(max_slots, hid_dim))

        self.fc2_w = nn.Parameter(uniform_init(max_slots, hid_dim, nproperties))
        self.fc2_b = nn.Parameter(uniform_init(max_slots, nproperties))


        self.property_classifier = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hid_dim, nproperties),
                                        nn.Sigmoid())


        self.reasoning_classifier = nn.Sequential(nn.Linear(nproperties, nclasses))


    def forward(self, slots, cidxs, epoch=0, batch=0):
        
        shape = cidxs.shape
        encodings = torch.zeros(np.prod(shape), self.max_slots, device=slots.device)
        encodings.scatter_(1, cidxs.reshape(-1, 1), 1)

        # weights for transformations =======================
        encodings = encodings.view(shape[0], shape[1], -1)
        fc1w = torch.einsum('bnd,dgk->bngk', encodings, self.fc1_w)
        fc1b = torch.einsum('bnd,dg->bng',encodings, self.fc1_b)

        fc2w = torch.einsum('bnd,dgk->bngk',encodings, self.fc2_w)
        fc2b = torch.einsum('bnd,dg->bng',encodings, self.fc2_b)

        # apply transformation ===============
        properties = F.relu(torch.einsum('bnd,bndw->bnw', slots, fc1w) + fc1b)
        properties = F.relu(torch.einsum('bnd,bndw->bnw', properties, fc2w) + fc2b) 

        properties = properties.sum(1)
        class_logits = self.reasoning_classifier(properties)

        return class_logits


class DefaultCNN(nn.Module):
    def __init__(self,
                    resolution,
                    hid_dim,
                    kernel_size=5,
                    encoder_res=4,
                    nclasses = 3):
        super().__init__()

        self.resolution = resolution
        self.hid_dim = hid_dim
        self.encoder_res = encoder_res
        self.nclasses = nclasses 

        self.encoder_cnn = Encoder(self.resolution, self.hid_dim, kernel_size)

        self.classifier = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hid_dim, self.nclasses))

    def forward(self, x, epoch=0, batch=0):
        features = self.encoder_cnn(x)
        avg_pooled = F.adaptive_avg_pool2d(features, (1,1))
        return self.classifier(avg_pooled.squeeze())