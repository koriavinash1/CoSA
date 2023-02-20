import torch
import torch.nn as nn
from einops import rearrange, repeat
from src.utils import unique_sampling_fn, get_euclidian_distance, sorting_idxs, get_cosine_distance, get_cb_variance


class QKCodebook(nn.Module):
    def __init__(self, dim, codebook_size):
        super().__init__()
        self.dim = dim
        self.mu_embeddings = nn.Embedding(codebook_size, dim)
        self.sigma_embeddings = nn.Embedding(codebook_size, dim)
        nn.init.xavier_uniform_(self.mu_embeddings.weight)
        nn.init.xavier_uniform_(self.sigma_embeddings.weight)

    def sample_mu_sigma(self, quantized, encodings, shape, MCsamples=1):
        # quantized: quantized feature encodings MB*Ntokens x dim
        # encodings: encoded index MB*Ntokens x 1
        # shape: MB x Ntokens x dim

        shape = (shape[0], -1, self.dim)
        slot_mu = torch.matmul(encodings, self.mu_embeddings.weight)
        slot_sigma = torch.matmul(encodings, self.sigma_embeddings.weight)

        slot_mu = slot_mu.view(shape)
        slot_sigma = slot_sigma.view(shape)

        # MC expectation estimation
        sampling_shape = (slot_sigma.shape[0], MCsamples, slot_sigma.shape[1], slot_sigma.shape[2])  
        slots = slot_mu.unsqueeze(1) + slot_sigma.unsqueeze(1) * torch.randn(sampling_shape, 
                                                            device = slot_sigma.device, 
                                                            dtype = slot_sigma.dtype)

        # MC samples along batch axis.....
        slots = rearrange(slots, 'b m n d -> (b m) n d')
        return slots, slot_mu, slot_sigma


    def compute_qkloss(self, embed_ind, inputs):
        # include all QKcodebook regularizations
        input_shape = inputs.shape

        unique_code_ids = torch.unique(embed_ind)
        mu_codebook = self.mu_embeddings.weight[unique_code_ids]
        logvar_codebook = self.sigma_embeddings.weight[unique_code_ids]

        qkloss = 0
        # qkloss += get_cb_variance(mu_codebook)

        # kl loss between sampled marginal distributions
        qkloss += 0 # torch.mean(-0.5 * (1 + logvar - mean ** 2 - logvar.exp()))
        
        return qkloss
