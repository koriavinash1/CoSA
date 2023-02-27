import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat
from src.utils import (uniform_init, hsphere_init, unique_sampling_fn, 
                        get_euclidian_distance, sorting_idxs, 
                        get_cosine_distance, get_cb_variance)




class QKCodebook(nn.Module):
    def __init__(self, dim, codebook_size):
        super().__init__()
        self.dim = dim
        
        self.register_buffer('mu_embeddings', hsphere_init(codebook_size, dim))
        self.register_buffer('sigma_embeddings', uniform_init(codebook_size, dim))

        self.fc1_w = nn.Parameter(uniform_init(codebook_size, dim, dim))
        self.fc1_b = nn.Parameter(uniform_init(codebook_size, dim))

        self.fc2_w = nn.Parameter(uniform_init(codebook_size, dim, dim))
        self.fc2_b = nn.Parameter(uniform_init(codebook_size, dim))


    def sample_slots(self, encodings, shape, MCsamples=1):
        # encodings: encoded index MB*Ntokens x 1
        # shape: MB x Ntokens x dim

        shape = (shape[0], shape[1], self.dim)
        slot_mu = torch.matmul(encodings, self.mu_embeddings)
        slot_sigma = torch.matmul(encodings, self.sigma_embeddings)
        slot_sigma = torch.exp(0.5*slot_sigma)
        slots = torch.cat([torch.normal(slot_mu, slot_sigma).view(shape).unsqueeze(1) for _ in range(MCsamples)], 1)


        # weights for transformations =======================
        encodings = encodings.view(shape[0], shape[1], -1)
        fc1w = torch.einsum('bnd,dgk->bngk', encodings, self.fc1_w)
        fc1b = torch.einsum('bnd,dg->bng',encodings, self.fc1_b)

        fc2w = torch.einsum('bnd,dgk->bngk',encodings, self.fc2_w)
        fc2b = torch.einsum('bnd,dg->bng',encodings, self.fc2_b)


        # apply transformation ===============
        slots = F.relu(torch.einsum('bmnd,bndw->bmnw', slots, fc1w) + fc1b.unsqueeze(1))
        slots = F.relu(torch.einsum('bmnd,bndw->bmnw', slots, fc2w) + fc2b.unsqueeze(1)) 


        # slot_mu = slot_mu.view(shape)
        # slot_sigma = slot_sigma.view(shape)

        # # MC expectation estimation
        # sampling_shape = (slot_sigma.shape[0], MCsamples, slot_sigma.shape[1], slot_sigma.shape[2])  
        # slots = slot_mu.unsqueeze(1) + slot_sigma.unsqueeze(1) * torch.randn(sampling_shape, 
        #                                                     device = slot_sigma.device, 
        #                                                     dtype = slot_sigma.dtype)

        # MC samples along batch axis.....

        slots = rearrange(slots, 'b m n d -> (b m) n d')
        return slots #, slot_mu, slot_sigma


    def compute_qkloss(self, embed_ind, inputs):
        # include all QKcodebook regularizations
        input_shape = inputs.shape

        unique_code_ids = torch.unique(embed_ind)

        qkloss = 0
        # qkloss += get_cb_variance(mu_codebook)

        # kl loss between sampled marginal distributions
        qkloss += 0 # torch.mean(-0.5 * (1 + logvar - mean ** 2 - logvar.exp()))
        
        return qkloss
