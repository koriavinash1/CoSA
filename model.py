import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from einops import rearrange
from geomstats.geometry.hypersphere import Hypersphere



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SlotAttention(nn.Module):
    def __init__(self, num_slots, 
                        dim, 
                        iters = 3, 
                        eps = 1e-8, 
                        hidden_dim = 128,
                        max_slots=64, 
                        temp=1,
                        beta=0.99,
                        encoder_intial_res=(8, 8),
                        decoder_intial_res=(8, 8),
                        hyperspherical=False,
                        unique_sampling=False,
                        gumble = False
                        ):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.dim = dim
        self.scale = dim ** -0.5
        self.temperature = temp
        ntokens = int(np.prod(encoder_intial_res))
        
        assert self.num_slots < np.prod(encoder_intial_res), f'reduce number of slots, max possible {np.prod(encoder_intial_res)}'

        # ===========================================
        # encoder postional embedding with linear transformation
        self.encoder_position = SoftPositionEmbed(dim, encoder_intial_res)
        self.encoder_norm = nn.LayerNorm([ntokens, dim])
        self.encoder_feature_mlp = nn.Sequential(nn.Linear(dim, dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(dim, dim))



        # # decoder positional embeddings
        self.decoder_position = SoftPositionEmbed(dim, decoder_intial_res)
        self.decoder_initial_res = decoder_intial_res

        # ===========================================

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        torch.nn.init.xavier_uniform_(self.slots_logsigma)


        self.anchorsto_k = nn.Sequential(nn.Linear(ntokens, ntokens),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(ntokens, ntokens))


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


        # slot quantization parameters
        self.max_slots = max_slots

        self.gumble = gumble


        cb_dim = dim
        self.embedding = nn.Embedding(self.max_slots, cb_dim)

        if self.gumble:

            self.proj = nn.Linear(dim, self.max_slots, 1)
            self.embedding.weight.data.uniform_(-1.0 / self.max_slots, 1.0 / self.max_slots)
        else:
            # to_q transformation needs to be invertable in case of gumble softmax
            # we assume that the cb entries can directly learned to map to_q transformation
            # and only include that in traditional cb sampling


            self.beta = beta
            self.hyperspherical = hyperspherical
            self.unique_sampling = unique_sampling
            if not self.hyperspherical:
                self.embedding.weight.data.uniform_(-1.0/self.max_slots, 1.0/self.max_slots)
            else:
                sphere = Hypersphere(dim= cb_dim - 1)
                points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=self.max_slots))
                self.embedding.weight.data.copy_(points_in_manifold)

        self.embedding.weight.requires_grad=True
        self.straight_through = True



    def get_euclidian_distance(self, u, v):
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(u ** 2, dim=1, keepdim=True) + \
            torch.sum(v**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', u, rearrange(v, 'n d -> d n'))
        return d


    
    def get_hyperspherical_distance(self, u, v):
        # distance on sphere
        d = torch.einsum('bd,dn->bn', u, rearrange(v, 'n d -> d n'))
        ed1 = torch.sqrt(torch.sum(u**2, dim=1, keepdim=True))
        ed2 = torch.sqrt(torch.sum(v**2, dim=1, keepdim=True))
        ed3 = torch.einsum('bd,dn->bn', ed1, rearrange(ed2, 'n d  -> d n'))
        geod = torch.clamp(d/(ed3), min=-0.99999, max=0.99999)
        return torch.acos(geod)


    def unique_sampling_fn(self, distances, features=None, scales=None):
        # distance: Bxntokensxntokens
        # _min_: bool to include distance minimization or maximization

        B, S, N = distances.shape

        batch_sampled_vectors = []
        for b in range(B):
            distance_vector = distances[b, ...]
            sorted_idx = torch.argsort(distance_vector, -1)
            
            sampled_vectors = []
            sampled_distances = []
            for i in range(S):
                for k in range(N):
                    current_idx = sorted_idx[i, k]
                    if not (current_idx in sampled_vectors):
                        sampled_vectors.append(current_idx.unsqueeze(0))
                        sampled_distances.append(distance_vector[i, current_idx].unsqueeze(0))
                        break
            
            sampled_vectors = torch.cat(sampled_vectors, 0)
            sampled_distances = torch.cat(sampled_distances, 0)
            sampled_vectors = sampled_vectors[torch.argsort(sampled_distances)]
            batch_sampled_vectors.append(sampled_vectors.unsqueeze(0))

        batch_sampled_vectors = torch.cat(batch_sampled_vectors, 0)
        return batch_sampled_vectors.view(-1)

    

    def sample_slotemb_gumble(self, z, update=0, unique= False):
        # z: slot keys (B x ntokens x token_dim)
        shape = z.shape
        z_flattened = z.reshape(-1, shape[-1])
        hard = self.straight_through if self.training else True
        temp = self.temperature

        logits = self.proj(z_flattened)
        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)

        z_q = torch.einsum('b n, n d -> b d', soft_one_hot, self.embedding.weight)
        z_q = z_q.view(shape)

        indices = soft_one_hot.argmax(dim=1)
        indices = indices.view(shape[0], -1)

        qy = F.softmax(logits, dim=1)
        kl_loss = torch.sum(qy * torch.log(qy + 1e-10), dim=1).mean()

        
        # preserce gradients
        z_q = z + (z_q - z).detach()
        return z_q, indices, kl_loss


    def sample_slotemb_traditional(self, z, update=0, unique = False):
        # z: slot keys (B x ntokens x token_dim)
        shape = z.shape

        z_flattened = z.reshape(-1, self.dim)
        transformed_slots = self.to_q(self.embedding.weight)


        if not self.hyperspherical:
            d = self.get_euclidian_distance(z_flattened, transformed_slots) 
        else: 
            d = self.get_hyperspherical_distance(z_flattened, transformed_slots)
        

        if unique:
            d = d.view(shape[0], -1, self.max_slots)
            indices = self.unique_sampling_fn(d)
        else:
            sampled_dist, indices = torch.min(d, dim=1)
            indices = indices.view(shape[0], -1)
            sampled_dist = sampled_dist.view(shape[0], -1)
            indices = indices.view(-1)
            indices = indices[torch.argsort(sampled_dist, 1).view(-1)]


        z_q = self.embedding(indices)
        transformed_zq = transformed_slots[indices]

        z_q = z_q.view(shape[0], -1, self.dim)
        transformed_zq = transformed_zq.view(shape[0], -1, self.dim)
        indices = indices.view(shape[0], -1)


        if update == 0: 
            loss = ((transformed_zq.detach() - z) ** 2).sum()/shape[0]  
            loss += self.beta * ((transformed_zq - z.detach()) ** 2).sum()/shape[0]
        elif update == 1:
            loss = self.beta * ((transformed_zq - z.detach()) ** 2).sum()/shape[0]
        elif update == 2:
            loss = ((transformed_zq.detach() - z) ** 2).sum()/shape[0]  
        else:
            raise ValueError()

        # # preserce gradients
        z_q = z + (z_q - z).detach()
        return z_q, indices, loss

    
    def sample_slots(self, z, update= 0, unique = False):
        # z: slot keys (B x ntokens x token_dim)
        if self.gumble:
            return self.sample_slotemb_gumble(z, update, unique)
        else:
            return self.sample_slotemb_traditional(z, update, unique)


    def get_slot_variance(self):
        cd = self.get_euclidian_distance(self.embedding.weight, self.embedding.weight)
        return torch.mean(torch.var(cd, 1)) 


    def passthrough_eigen_basis(self, x, ns):
        # x: token embeddings B x ntokens x token_embeddings

        x = F.normalize(x, p=2, dim=-1)
        coveriance_matrix = torch.einsum('bkf, bgf -> bkg', x, x)
        eigen_values, eigen_vectors = torch.symeig(coveriance_matrix, eigenvectors=True)
        eigen_vectors = eigen_vectors[:, :, 1:ns+1].permute(0, 2, 1)
        eigen_values = eigen_values[:, 1:ns+1]
        return eigen_vectors, eigen_values

    
    def masked_projection(self, features, z):
        # features: B x nanchors x f
        # z: basis B x K x nanchors

        b, n, f = features.shape
        k = z.shape[1]
        features = features.unsqueeze(1).repeat(1, k, 1, 1) # B x k x n x f
        z = z.unsqueeze(-1)
        z = z.repeat(1, 1, 1, f)# B x k x n x f

        projection = features*z
        return torch.sum(projection, 2)


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
            features = self.encoder_norm(features)

        return features


    def decoder_transformation(self, slots):
        # features: B x nslots x dim
        slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        features = slots.repeat((1, self.decoder_initial_res[0], self.decoder_initial_res[1], 1))
        features = self.decoder_position(features)
        return features.permute(0, 3, 1, 2)


    def forward(self, inputs, num_slots = None, epoch=0):
        b, d, w, h = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        # compute slots w.r.t feature_keys
        inputs_without_position = self.encoder_transformation(inputs, position = False)
        k_noposition = self.to_k(inputs_without_position)

        eigen_basis, eigen_values = self.passthrough_eigen_basis(k_noposition, n_s)
        object_vectors = self.masked_projection(inputs_without_position, eigen_basis)   
        object_vectors = F.layer_norm(object_vectors, object_vectors.shape[1:])
        

        # update codebook with object vectors
        _, _, qloss = self.sample_slots(F.normalize(object_vectors, p=2, dim=2), update = 0, unique = True)


        # input with position encoding
        inputs = self.encoder_transformation(inputs)
        k, v= self.to_k(inputs), self.to_v(inputs)


        slots, cbidxs, transformation_loss = self.sample_slots(F.normalize(k, p=2, dim=2), update = 2, unique = False)
        slots = slots[:, :n_s, :]
        qloss += transformation_loss

        # add slot noise...
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)
        slots = slots + sigma * torch.randn(slots.shape).to(inputs.device)


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
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, cbidxs, qloss, self.decoder_transformation(slots)



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

class Encoder(nn.Module):
    def __init__(self, resolution, hid_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hid_dim, 5, stride=2, padding = 2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5,  stride=2, padding = 2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, stride=2, padding = 2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, stride=2, padding = 2)

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
        x = x[:,:,:self.resolution[0], :self.resolution[1]]
        return x


"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution, 
                        num_slots, 
                        num_iterations, 
                        hid_dim, 
                        max_slots, 
                        hyperspherical, 
                        unique_samling, 
                        gumble,
                        encoder_res,
                        decoder_res):
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
            max_slots=max_slots,
            hyperspherical=hyperspherical,
            unique_sampling=unique_samling,
            gumble = gumble,
            encoder_intial_res=(encoder_res, encoder_res),
            decoder_intial_res=(decoder_res, decoder_res))


    def forward(self, image, num_slots=None, epoch=0):
        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        # `x` has shape: [batch_size, input_size, width, height].

        # Slot Attention module.
        slots, cbidxs, qloss, features = self.slot_attention(x, num_slots, epoch)
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

        return recon_combined, recons, masks, slots, cbidxs, qloss
