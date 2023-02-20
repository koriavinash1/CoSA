import torch
import torch.nn as nn
from src.utils import unique_sampling_fn, get_euclidian_distance, sorting_idxs, get_cosine_distance, get_cb_variance


class QKCodebook(nn.Module):
    def __init__(self, dim, codebook_size):
        super().__init__()
        self.dim = dim
        self.mu_embeddings = nn.Embedding(codebook_size, dim)
        self.sigma_embeddings = nn.Embedding(codebook_size, dim)
        nn.init.xavier_uniform_(self.mu_embeddings.weight)
        nn.init.xavier_uniform_(self.sigma_embeddings.weight)

    def sample_mu_sigma(self, quantized, encodings, shape):
        # quantized: quantized feature encodings MB*Ntokens x dim
        # encodings: encoded index MB*Ntokens x 1
        # shape: MB x Ntokens x dim

        shape = (shape[0], -1, self.dim)
        slot_mu = torch.matmul(encodings, self.mu_embeddings.weight)
        slot_sigma = torch.matmul(encodings, self.sigma_embeddings.weight)

        # slot_sigma = torch.clamp(torch.exp(0.5*slot_sigma), min=1e-3, max=1)
        slots = slot_mu + slot_sigma * torch.randn(slot_sigma.shape, 
                                                device = slot_sigma.device, 
                                                dtype = slot_sigma.dtype)
        slots =  slots.view(shape)
        slot_mu = slot_mu.view(shape)
        
        slot_sigma = slot_sigma.view(shape)
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
