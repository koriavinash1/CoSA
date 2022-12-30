import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from torch.autograd import Variable
import torchvision.models as models
import numpy as np

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from einops import rearrange
from src.hpenalty import hessian_penalty

from torch.distributions import Categorical


def unique_sampling_fn(distances, nunique=-1):
    # distance: Bxntokensxncbtokens

    B, S, N = distances.shape
    if (nunique == -1): nunique = min(S, N)

    batch_sampled_vectors = []
    for b in range(B):
        distance_vector = distances[b, ...]
        sorted_idx = torch.argsort(distance_vector, dim=-1, descending=False)
        
        sampled_vectors = []
        sampled_distances = []
        for i in range(S):
            if len(sampled_vectors) <= nunique:
                for k in range(N):
                    current_idx = sorted_idx[i, k]
                    if not (current_idx in sampled_vectors):
                        sampled_vectors.append(current_idx.unsqueeze(0))
                        sampled_distances.append(distance_vector[i, current_idx].unsqueeze(0))
                        break
            else:
                current_idx = sorted_idx[i, 0]
                sampled_vectors.append(current_idx.unsqueeze(0))
                sampled_distances.append(distance_vector[i, current_idx].unsqueeze(0))
 
        
        sampled_vectors = torch.cat(sampled_vectors, 0)
        sampled_distances = torch.cat(sampled_distances, 0)
        sampled_vectors = sampled_vectors[torch.argsort(sampled_distances, descending=False)]
        batch_sampled_vectors.append(sampled_vectors.unsqueeze(0))

    batch_sampled_vectors = torch.cat(batch_sampled_vectors, 0)
    return batch_sampled_vectors.view(-1)

    
def get_euclidian_distance(u, v):
    # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
    d = torch.sum(u**2, dim=1, keepdim=True) + \
        torch.sum(v**2, dim=1) - 2 * torch.matmul(u, v.t())
    return d


def get_cosine_distance(u, v):
    # distance on sphere
    d = torch.einsum('bd,dn->bn', u, rearrange(v, 'n d -> d n'))
    ed1 = torch.sqrt(torch.sum(u**2, dim=1, keepdim=True))
    ed2 = torch.sqrt(torch.sum(v**2, dim=1, keepdim=True))
    ed3 = torch.einsum('bd,dn->bn', ed1, rearrange(ed2, 'n d  -> d n'))
    geod = torch.clamp(d/(ed3), min=-0.99999, max=0.99999)
    return torch.acos(geod)


def get_cb_variance(cb):
    cd = get_euclidian_distance(cb, cb)
    return torch.mean(torch.var(cd, 1)) 



class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, 
                        embedding_dim, 
                        commitment_cost=0.99,
                        usage_threshold=1.0e-9,
                        variational=False,
                        cosine=False):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._usage_threshold = usage_threshold
        self.variational = variational
        self._cosine = cosine

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._get_distance = get_euclidian_distance

        if self._cosine:
            sphere = Hypersphere(dim=self._embedding_dim - 1)
            points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=self._num_embeddings))
            self._embedding.weight.data.copy_(points_in_manifold)
            self._get_distance = get_cosine_distance

        self.modulator = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(embedding_dim, embedding_dim)
                                        )

        # ======================
        # Variational
        if self.variational:
            self.mean = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(embedding_dim, 1))
            self.logvar = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(embedding_dim, 1))

            self.sampler = lambda mu, std: torch.randn_like(std) * std + mu

        self.register_buffer('_usage', torch.ones(self._num_embeddings), persistent=False)
        self.get_variance = lambda: get_cb_variance(self._embedding.weight) 

    def update_usage(self, min_enc):
        self._usage[min_enc] = self._usage[min_enc] + 1  # if code is used add 1 to usage
        self._usage /= 2 # decay all codes usage

    def reset_usage(self):
        self._usage.zero_() #  reset usage between epochs


    def random_restart(self):
        #  randomly restart all dead codes below threshold with random code in codebook
        dead_codes = torch.nonzero(self._usage < self._usage_threshold).squeeze(1)
        useful_codes = torch.nonzero(self._usage > self._usage_threshold).squeeze(1)
        mean = self._embedding.weight[useful_codes].mean(0).unsqueeze(0)
        var = self._embedding.weight[useful_codes].var(0).unsqueeze(0)
        eps = torch.randn((len(dead_codes), self._embedding_dim)).to(var.device)
        rand_codes = eps*var + mean

        with torch.no_grad():
            self._embedding.weight[dead_codes] = rand_codes
        self._embedding.weight.requires_grad = True


    def entire_cb_restart(self):
        mean = self._embedding.weight.mean(0).unsqueeze(0)
        var = self._embedding.weight.var(0).unsqueeze(0)
        eps = torch.randn((self._num_embeddings, self._embedding_dim)).to(var.device)
        rand_codes = eps*var + mean

        with torch.no_grad():
            self._embedding.weight = rand_codes
        self._embedding.weight.requires_grad = True


    def sample(self, features, unique=False):
        input_shape = features.shape
        features = features.view(-1, self._embedding_dim)

        # Calculate distances
        distances = self._get_distance(features, self._embedding.weight)


        # Encoding
        if not unique: 
            # encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            sampled_dist, encoding_indices = torch.min(distances, dim=1)
            encoding_indices = encoding_indices.view(input_shape[0], -1)
            sampled_dist = sampled_dist.view(input_shape[0], -1)

            encoding_indices = encoding_indices.view(-1)
            encoding_indices = encoding_indices[torch.argsort(sampled_dist, dim=1, descending=False).view(-1)]
            encoding_indices = encoding_indices.unsqueeze(1)
        else: 
            encoding_indices = unique_sampling_fn(distances.view(input_shape[0], -1, self._num_embeddings)).unsqueeze(1)

        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=features.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        return quantized, encoding_indices, encodings


    def forward(self, inputs, avg=False, 
                        unique = False,
                        nunique = -1,
                        update=True,
                        loss_type=0,
                        reset_usage=False):
                        
        input_shape = inputs.shape
        # modulator pass
        # inputs_ = self.modulator(inputs)
        # hloss = hessian_penalty(G_ = self.modulator, z=inputs, G_z=inputs_)
        # inputs = inputs_

        # Restart vectors
        if update:
            if np.random.uniform() > 0.99: self.random_restart()
            if reset_usage: self.reset_usage()


        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        quantized, encoding_indices, encodings = self.sample(inputs, unique)

        # Reset unused cb vectors...
        self.update_usage(encoding_indices)

        # Loss
        if not avg:
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
        else:
            e_latent_loss = F.mse_loss(quantized.mean(1).detach(), inputs.mean(1))
            q_latent_loss = F.mse_loss(quantized.mean(1), inputs.mean(1).detach())

        if loss_type == 0:
            loss = q_latent_loss + self._commitment_cost * e_latent_loss
        elif loss_type == 1:
            loss = q_latent_loss
        else:
            loss = self._commitment_cost * e_latent_loss

        # Variational...
        if self.variational:
            klloss = 0
            # conti. divergence 
            mean = self.mean(inputs).squeeze(-1)
            logvar = self.logvar(inputs).squeeze(-1)
            klloss += 0.5*torch.mean(-0.5 * (1 + logvar - mean ** 2 - logvar.exp()))

            # dis. divergence 
            mean = self.mean(quantized).squeeze(-1)
            logvar = self.logvar(quantized).squeeze(-1)
            klloss += 0.5*torch.mean(-0.5 * (1 + logvar - mean ** 2 - logvar.exp()))
            loss += klloss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        encoding_indices = encoding_indices.view(input_shape[0], -1)
        return loss, quantized, perplexity, encoding_indices




class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, 
                        embedding_dim, 
                        commitment_cost, 
                        decay, 
                        epsilon=1e-5,
                        usage_threshold=1.0e-9,
                        variational=False,
                        cosine=False):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._usage_threshold = usage_threshold
        self.variational = variational
        self._cosine = cosine


        self.modulator = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(embedding_dim, embedding_dim)
                                        )



        # ======================

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._get_distance = get_euclidian_distance

        if self._cosine:
            sphere = Hypersphere(dim=self._embedding_dim - 1)
            points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=self._num_embeddings))
            self._embedding.weight.data.copy_(points_in_manifold)
            self._get_distance = get_cosine_distance



        # ======================
        # Variational
        if self.variational:
            self.mean = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(embedding_dim, 1))
            self.logvar = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(embedding_dim, 1))

            self.sampler = lambda mu, std: torch.randn_like(std) * std + mu

        # ======================
        self.register_buffer('_usage', torch.ones(self._num_embeddings), persistent=False)
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon


        self.get_variance = lambda: get_cb_variance(self._embedding.weight) 


    def update_usage(self, min_enc):
        self._usage[min_enc] = self._usage[min_enc] + 1  # if code is used add 1 to usage
        self._usage /= 2 # decay all codes usage

    def reset_usage(self):
        self._usage.zero_() #  reset usage between epochs


    def random_restart(self):
        #  randomly restart all dead codes below threshold with random code in codebook
        dead_codes = torch.nonzero(self._usage < self._usage_threshold).squeeze(1)
        useful_codes = torch.nonzero(self._usage > self._usage_threshold).squeeze(1)
        mean = self._embedding.weight[useful_codes].mean(0).unsqueeze(0)
        var = self._embedding.weight[useful_codes].var(0).unsqueeze(0)
        eps = torch.randn((len(dead_codes), self._embedding_dim)).to(var.device)
        rand_codes = eps*var + mean

        with torch.no_grad():
            self._embedding.weight[dead_codes] = rand_codes
        self._embedding.weight.requires_grad = True
        # print ('Restarting dead cb embeddings...: ', len(dead_codes))


    def entire_cb_restart(self):
        mean = self._embedding.weight.mean(0).unsqueeze(0)
        var = self._embedding.weight.var(0).unsqueeze(0)
        eps = torch.randn((self._num_embeddings, self._embedding_dim)).to(var.device)
        rand_codes = eps*var + mean

        with torch.no_grad():
            self._embedding.weight[:] = rand_codes
        self._embedding.weight.requires_grad = True


    def sample(self, features, unique=False, nunique=-1):
        input_shape = features.shape
        features = features.view(-1, self._embedding_dim)

        # Calculate distances
        distances = self._get_distance(features, self._embedding.weight)


        # Encoding
        if not unique: 
            # encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            sampled_dist, encoding_indices = torch.min(distances, dim=1)
            encoding_indices = encoding_indices.view(input_shape[0], -1)
            sampled_dist = sampled_dist.view(input_shape[0], -1)

            encoding_indices = encoding_indices.view(-1)
            encoding_indices = encoding_indices[torch.argsort(sampled_dist, dim=1, descending=False).view(-1)]
            encoding_indices = encoding_indices.unsqueeze(1)
        else: 
            encoding_indices = unique_sampling_fn(distances.view(input_shape[0], -1, self._num_embeddings), nunique).unsqueeze(1)

        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=features.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        return quantized, encoding_indices, encodings


    def forward(self, inputs, avg=False, 
                        unique = False,
                        nunique = -1,
                        update=True,
                        loss_type=0,
                        reset_usage=False):
        input_shape = inputs.shape
        # modulator pass
        # inputs_ = self.modulator(inputs)
        # hloss = hessian_penalty(G_ = self.modulator, z=inputs, G_z=inputs_)
        # inputs = inputs_

        # Restart vectors
        if update:
            if np.random.uniform() > 0.99: self.random_restart()
            if reset_usage: self.reset_usage()
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        quantized, encoding_indices, encodings = self.sample(inputs, unique, nunique)

        # Reset unused cb vectors...
        if update: self.update_usage(encoding_indices)
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = ((self._ema_cluster_size + self._epsilon)
                                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        

        klloss = 0
        # Variational...
        if self.variational:
            # conti. divergence 
            mean = self.mean(inputs).squeeze(-1)
            logvar = self.logvar(inputs).squeeze(-1)
            klloss += 0.5*torch.mean(-0.5 * (1 + logvar - mean ** 2 - logvar.exp()))

            # dis. divergence 
            # mean = self.mean(quantized).squeeze(-1)
            # logvar = self.logvar(quantized).squeeze(-1)
            # klloss += 0.5*torch.mean(-0.5 * (1 + logvar - mean ** 2 - logvar.exp()))
            # print (klloss)

        # Loss
        if not avg:
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
        else:
            e_latent_loss = F.mse_loss(quantized.mean(1).detach(), inputs.mean(1))
            q_latent_loss = F.mse_loss(quantized.mean(1), inputs.mean(1).detach())


        if loss_type == 0:
            loss = q_latent_loss + self._commitment_cost * e_latent_loss
        elif loss_type == 1:
            loss = q_latent_loss
        else:
            loss = self._commitment_cost * e_latent_loss


        # print (loss, klloss, hloss)
        loss += klloss
        # loss += hloss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        encoding_indices = encoding_indices.view(input_shape[0], -1)
        return loss, quantized, perplexity, encoding_indices