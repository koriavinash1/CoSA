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
    if not isinstance(nunique, list):
        if (nunique == -1): 
            nunique = min(S, N)
        nunique = [nunique]*B


    batch_sampled_vectors = []
    for b in range(B):
        distance_vector = distances[b, ...]
        sorted_idx = torch.argsort(distance_vector, dim=-1, descending=False)
        
        sampled_vectors = []
        sampled_distances = []
        for i in range(S):
            if i < nunique[b]:
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
    return 2.0*torch.acos(torch.abs(geod))/np.pi


def get_cb_variance(cb):
    # cb = cb / torch.norm(cb, dim=1, keepdim=True)
    cd = get_cosine_distance(cb, cb)
    return 1 - torch.mean(torch.var(cd, 1)) 



class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, 
                        embedding_dim, 
                        commitment_cost=0.99,
                        usage_threshold=1.0e-9,
                        variational=False,
                        qk=False,
                        cosine=False):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._usage_threshold = usage_threshold
        self.variational = variational
        self._cosine = cosine
        self.qk=qk

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._get_distance = get_euclidian_distance

        if self._cosine:
            sphere = Hypersphere(dim=self._embedding_dim - 1)
            points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=self._num_embeddings))
            self._embedding.weight.data.copy_(points_in_manifold)
            self._get_distance = get_cosine_distance


        if self.qk:
            self.mu_embeddings = nn.Embedding(self._num_embeddings, self._embedding_dim)
            self.sigma_embeddings = nn.Embedding(self._num_embeddings, self._embedding_dim)
            nn.init.xavier_uniform_(self.mu_embeddings.weight)
            nn.init.xavier_uniform_(self.sigma_embeddings.weight)



        self.data_mean = 0
        self.data_std = 0

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

        eps = torch.randn((len(dead_codes), self._embedding_dim)).to(self._embedding.weight.device)
        rand_codes = eps*self.data_std + self.data_mean

        with torch.no_grad():
            self._embedding.weight[dead_codes] = rand_codes
        self._embedding.weight.requires_grad = True


    def entire_cb_restart(self):
        eps = torch.randn((self._num_embeddings, self._embedding_dim)).to(self._embedding.weight.device)
        rand_codes = eps*self.data_std + self.data_mean

        with torch.no_grad():
            self._embedding.weight = rand_codes
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

        slots = None
        # slot sampling
        if self.qk:
            slot_mu = torch.matmul(encodings, self.mu_embeddings.weight)
            slot_sigma = torch.matmul(encodings, self.sigma_embeddings.weight)
            slots = torch.normal(slot_mu, slot_sigma).view(input_shape)

        return quantized, encoding_indices, encodings, slots


    def forward(self, inputs, avg=False, 
                        unique = False,
                        nunique = -1,
                        update=True,
                        loss_type=0,
                        reset_usage=False):
                        
        input_shape = inputs.shape

        # Restart vectors
        if update:
            if np.random.uniform() > 0.99: self.random_restart()
            if reset_usage: self.reset_usage()


        scale = torch.norm(inputs, dim = -1, keepdim=True)
        inputs = inputs / scale



        klloss = 0
        # Variational...
        if self.variational:
            # conti. divergence 
            mean = self.mean(inputs).squeeze(-1)
            logvar = self.logvar(inputs).squeeze(-1)
            klloss += torch.mean(-0.5 * (1 + logvar - mean ** 2 - logvar.exp()))

            # sample quantized
            sigma = torch.exp(0.5*logvar)
            inputs = self.variational_sampler(inputs, sigma)


        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        self.data_mean = 0.9*self.data_mean + 0.1*flat_input.clone().detach().mean(0)
        self.data_std = 0.9*self.data_std + 0.1*flat_input.clone().detach().std(0)

        
        quantized, encoding_indices, encodings, slots = self.sample(inputs, unique, nunique)

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

        loss += klloss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        encoding_indices = encoding_indices.view(input_shape[0], -1)


        quantized = quantized*scale

        return loss, quantized, perplexity, encoding_indices, slots




class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, 
                        embedding_dim, 
                        commitment_cost, 
                        decay, 
                        epsilon=1e-5,
                        usage_threshold=1.0e-9,
                        variational=False,
                        qk=False,
                        cosine=False):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._usage_threshold = usage_threshold
        self.variational = variational
        self._cosine = cosine
        self.qk = qk

        self.data_mean = 0
        self.data_std = 0

        # ======================

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._get_distance = get_euclidian_distance
        self.loss_fn = F.mse_loss

        if self._cosine:
            sphere = Hypersphere(dim=self._embedding_dim - 1)
            points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=self._num_embeddings))
            self._embedding.weight.data.copy_(points_in_manifold)
            self._get_distance = get_cosine_distance
            self.loss_fn = lambda x1,x2: 2.0*(1 - F.cosine_similarity(x1, x2).mean())



        # ======================
        # Variational
        if self.variational:
            self.mean = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(embedding_dim, embedding_dim))
            self.logvar = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(embedding_dim, embedding_dim))

            self.variational_sampler = lambda mu, std: torch.randn_like(std) * std + mu



        # ======================
        # Slot sampler
        if self.qk:
            self.mu_embeddings = nn.Embedding(self._num_embeddings, self._embedding_dim)
            self.sigma_embeddings = nn.Embedding(self._num_embeddings, self._embedding_dim)
            nn.init.xavier_uniform_(self.mu_embeddings.weight)
            nn.init.xavier_uniform_(self.sigma_embeddings.weight)


        # ======================
        self.register_buffer('_usage', torch.ones(self._num_embeddings), persistent=False)
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

        self.norm_input  = nn.LayerNorm(self._embedding_dim)
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
        eps = torch.randn((len(dead_codes), self._embedding_dim)).to(self._embedding.weight.device)
        rand_codes = eps*self.data_std + self.data_mean

        with torch.no_grad():
            self._embedding.weight[dead_codes] = rand_codes
        self._embedding.weight.requires_grad = True



    def entire_cb_restart(self):
        eps = torch.randn((self._num_embeddings, self._embedding_dim)).to(self._embedding.weight.device)
        rand_codes = eps*self.data_std + self.data_mean

        with torch.no_grad():
            self._embedding.weight[:] = rand_codes
        self._embedding.weight.requires_grad = True


    def sample(self, features, unique=False, nunique=-1):
        input_shape = features.shape
        features = features.view(-1, self._embedding_dim)
        

        # layer norm on features
        features = self.norm_input(features)

        # update data stats
        self.data_mean = 0.9*self.data_mean + 0.1*features.clone().detach().mean(0)
        self.data_std = 0.9*self.data_std + 0.1*features.clone().detach().std(0)

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
        
        slots = None
        # slot sampling
        if self.qk:
            # import pdb;pdb.set_trace()
            slot_mu = torch.matmul(encodings, self.mu_embeddings.weight)
            slot_sigma = torch.matmul(encodings, self.sigma_embeddings.weight)
            slot_sigma = torch.exp(0.5*slot_sigma)
            slots = torch.normal(slot_mu, slot_sigma).view(input_shape)

        return quantized, encoding_indices, encodings, slots


    def forward(self, inputs, avg=False, 
                        unique = False,
                        nunique = -1,
                        update=True,
                        loss_type=0,
                        reset_usage=False):
        input_shape = inputs.shape

        # Restart vectors
        if update:
            if np.random.uniform() > 0.99: self.random_restart()
            if reset_usage: self.reset_usage()
        

        if self._cosine:
            scale = torch.norm(inputs, dim = -1, keepdim=True)
            inputs = inputs / scale


        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)


        klloss = 0
        # Variational...
        if self.variational:
            # conti. divergence 
            mean = self.mean(inputs).squeeze(-1)
            logvar = self.logvar(inputs).squeeze(-1)
            klloss += torch.mean(-0.5 * (1 + logvar - mean ** 2 - logvar.exp()))

            # sample quantized
            sigma = torch.exp(0.5*logvar)
            inputs = self.variational_sampler(inputs, sigma)


        quantized, encoding_indices, encodings, slots = self.sample(inputs, unique, nunique)

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
        


        # Loss
        if not avg:
            e_latent_loss = self.loss_fn(quantized.detach(), inputs)
            q_latent_loss = self.loss_fn(quantized, inputs.detach())
        else:
            e_latent_loss = self.loss_fn(quantized.mean(1).detach(), inputs.mean(1))
            q_latent_loss = self.loss_fn(quantized.mean(1), inputs.mean(1).detach())


        if loss_type == 0:
            loss = q_latent_loss + self._commitment_cost * e_latent_loss
        elif loss_type == 1:
            loss = q_latent_loss
        else:
            loss = self._commitment_cost * e_latent_loss


        # qk loss
        qkloss = 0
        if self.qk:
            qkloss += get_cb_variance(self.mu_embeddings.weight)
            qkloss += 0.01*torch.mean(torch.norm(self.sigma_embeddings.weight, dim=1))
        #     slot_mu = self.mu_embeddings.weight
        #     slot_logvar = self.sigma_embeddings.weight


        # loss += get_cb_variance(self._embedding.weight)
        loss += klloss
        loss += qkloss


        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        encoding_indices = encoding_indices.view(input_shape[0], -1)

        if self._cosine:
            quantized = quantized*scale
            
        return loss, quantized, perplexity, encoding_indices, slots