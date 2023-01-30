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
    if not (isinstance(nunique, list) or isinstance(nunique, np.ndarray)):
        if (nunique == -1): 
            nunique = min(S, N)
        nunique = [nunique]*B
    
    nunique = np.minimum(nunique, N)
    batch_sampled_vectors = []
    for b in range(B):
        distance_vector = distances[b, ...]
        sorted_idx = torch.argsort(distance_vector, dim=-1, descending=False)
        # Create a bin over codebook direction of distance vectors
        # with nunique bins and sample based on that...
        # 
        #  
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


class BaseVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, 
                        embedding_dim,
                        nhidden,
                        commitment_cost=0.25,
                        usage_threshold=1.0e-9,
                        variational=False,
                        qk=False,
                        cosine=False,
                        gumble=False,
                        temperature=1.0,
                        kld_scale=1.0):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._usage_threshold = usage_threshold
        self.variational = variational
        self._cosine = cosine
        self.qk=qk


        self.proj = nn.Linear(nhidden, embedding_dim)
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._get_distance = get_euclidian_distance
        self.loss_fn = F.mse_loss
        
        self.norm_input  = nn.LayerNorm(self._embedding_dim)

        if self._cosine:
            sphere = Hypersphere(dim=self._embedding_dim - 1)
            points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=self._num_embeddings))
            self._embedding.weight.data.copy_(points_in_manifold)
            self._get_distance = get_cosine_distance
            self.loss_fn = lambda x1,x2: 2.0*(1 - F.cosine_similarity(x1, x2).mean())

        if self.qk:
            self.mu_embeddings = nn.Embedding(self._num_embeddings, self._embedding_dim)
            self.sigma_embeddings = nn.Embedding(self._num_embeddings, self._embedding_dim)
            nn.init.xavier_uniform_(self.mu_embeddings.weight)
            nn.init.xavier_uniform_(self.sigma_embeddings.weight)
            # nn.init.constant_(self.sigma_embeddings.weight, 0)

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
            self.variational_sampler = lambda mu, std: torch.randn_like(std) * std + mu



        self.register_buffer('_usage', torch.ones(self._num_embeddings), persistent=False)
        self.get_variance = lambda: get_cb_variance(self._embedding.weight) 


        # ======================
        # Gumble parameters
        self.gumble = gumble
        if self.gumble:
            self.temperature = temperature
            self.kld_scale = kld_scale
            self.gumble_proj = nn.Linear(nhidden, num_embeddings)


    def update_usage(self, min_enc):
        self._usage[min_enc] = self._usage[min_enc] + 1  # if code is used add 1 to usage
        self._usage /= 2 # decay all codes usage

    def reset_usage(self):
        self._usage.zero_() #  reset usage between epochs


    def random_restart(self):
        #  randomly restart all dead codes below threshold with random code in codebook
        dead_codes = torch.nonzero(self._usage < self._usage_threshold).squeeze(1)
        useful_codes = torch.nonzero(self._usage > self._usage_threshold).squeeze(1)
        N = self.data_std.shape[0]

        if len(dead_codes) > 0:
            eps = torch.randn((len(dead_codes), self._embedding_dim)).to(self._embedding.weight.device)
            rand_codes = eps*self.data_std.repeat(1 + len(dead_codes)//N, 1)[:len(dead_codes), :] +\
                                self.data_mean.repeat(1 + len(dead_codes)//N, 1)[:len(dead_codes), :]

            with torch.no_grad():
                self._embedding.weight[dead_codes] = rand_codes
            self._embedding.weight.requires_grad = True


    def entire_cb_restart(self):
        N = self.data_std.shape[0]
        eps = torch.randn((self._num_embeddings, self._embedding_dim)).to(self._embedding.weight.device)
        rand_codes = eps*self.data_std.repeat(1+len(eps)//N, 1)[:len(eps), :] +\
                                self.data_mean.repeat(1+len(eps)//N, 1)[:len(eps), :]


        with torch.no_grad():
            self._embedding.weight[:] = rand_codes
        self._embedding.weight.requires_grad = True


    def vq_sample(self, features, unique=False, nunique=-1, final=False):
        input_shape = features.shape
        
        # layer norm on features
        features = self.norm_input(features)

        # update data stats
        if not final:
            self.data_mean = 0.9*self.data_mean + 0.1*features.clone().detach().mean(0)
            self.data_std = 0.9*self.data_std + 0.1*features.clone().detach().std(0)

        features = features.view(-1, self._embedding_dim)

        # Calculate distances
        distances = self._get_distance(features, self._embedding.weight)

        def _min_encoding_(distances):
            # encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            sampled_dist, encoding_indices = torch.min(distances, dim=1)
            encoding_indices = encoding_indices.view(input_shape[0], -1)
            sampled_dist = sampled_dist.view(input_shape[0], -1)

            # import pdb;pdb.set_trace()
            encoding_indices = encoding_indices.view(-1)
            # encoding_indices = encoding_indices[torch.argsort(sampled_dist, dim=1, descending=False).view(-1)]
            encoding_indices = encoding_indices.unsqueeze(1)
            return encoding_indices

        # Encoding
        encoding_indices = _min_encoding_(distances)

        # if not unique: 
        #     encoding_indices = _min_encoding_(distances)
        # else: 
        #     if np.random.randn() > 0.5:
        #         encoding_indices = _min_encoding_(distances)
        #     else:
        #         encoding_indices = unique_sampling_fn(distances.view(input_shape[0], -1, self._num_embeddings), nunique).unsqueeze(1)

        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=features.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight)
        
        slots = None
        # slot sampling
        if self.qk and (not final):
            slot_mu = torch.matmul(encodings, self.mu_embeddings.weight)
            slot_mu = slot_mu + quantized.clone()

            slot_sigma = torch.matmul(encodings, self.sigma_embeddings.weight)
            # slot_sigma = torch.clamp(slot_sigma, min=-1, max=1)
            slot_sigma = torch.clamp(torch.exp(0.5*slot_sigma), min=1e-3, max=2)
            slots = torch.normal(slot_mu, slot_sigma).view(input_shape)

        return quantized.view(input_shape), encoding_indices, encodings, slots, None


    def gumble_sample(self, features, st=False, final=False):
        input_shape = features.shape

        # force hard = True when we are in eval mode, as we must quantize
        hard = st if self.training else True


        # layer norm on features
        features = self.norm_input(features)

        # update data stats
        if not final:
            self.data_mean = 0.9*self.data_mean + 0.1*features.clone().detach().mean(0)
            self.data_std = 0.9*self.data_std + 0.1*features.clone().detach().std(0)


        logits = self.gumble_proj(features)
        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard)
        quantized = torch.einsum('b t k, k d -> b t d', soft_one_hot, self._embedding.weight)


        encoding_indices = soft_one_hot.argmax(dim=-1).view(-1, 1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=features.device)
        encodings.scatter_(1, encoding_indices, 1)

        slots = None
        # slot sampling
        if self.qk and (not final):
            slot_mu = torch.matmul(encodings, self.mu_embeddings.weight)
            slot_mu = slot_mu + quantized.clone().view(-1, self._embedding_dim)

            slot_sigma = torch.matmul(encodings, self.sigma_embeddings.weight)
            # slot_sigma = torch.clamp(slot_sigma, min=-1, max=1)
            slot_sigma = torch.clamp(torch.exp(0.5*slot_sigma), min=1e-3, max=2)
            slots = torch.normal(slot_mu, slot_sigma).view(input_shape)
       
        return quantized, encoding_indices, encodings, slots, logits



    def sample(self, features, unique=False, nunique=-1, st=False, final=False):
        features = self.proj(features)

        if self.gumble:
            return self.gumble_sample(features, st, final)
        else:
            return self.vq_sample(features, unique=unique, nunique=nunique, final=final)


    def compute_baseloss(self, quantized, inputs, logits=None, avg=False, loss_type=0):
        if self.gumble:
            # + kl divergence to the prior loss
            qy = F.softmax(logits, dim=-1)
            loss = self.kld_scale * torch.mean(qy * torch.log(qy * self._num_embeddings + 1e-10), dim=1).mean()
        else:
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

        return loss 


    def forward(self, args):
        return NotImplementedError()



class VectorQuantizer(BaseVectorQuantizer):
    def __init__(self, num_embeddings, 
                        embedding_dim, 
                        nhidden,
                        commitment_cost=0.99,
                        usage_threshold=1.0e-9,
                        variational=False,
                        qk=False,
                        cosine=False,
                        gumble=False,
                        temperature=1.0,
                        kld_scale=1.0):
        super(VectorQuantizer, self).__init__(num_embeddings, 
                                                    embedding_dim, 
                                                    nhidden,
                                                    commitment_cost,
                                                    usage_threshold,
                                                    variational,
                                                    qk,
                                                    cosine,
                                                    gumble,
                                                    temperature,
                                                    kld_scale)
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._usage_threshold = usage_threshold
        self.variational = variational
        self._cosine = cosine
        self.qk=qk


    def forward(self, inputs, avg=False, 
                        unique = False,
                        nunique = -1,
                        update=True,
                        loss_type=0,
                        final=False,
                        reset_usage=False):

        input_shape = inputs.shape
        # Restart vectors
        if update:
            if np.random.uniform() > 0.99: self.random_restart()
            if reset_usage: self.reset_usage()


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


        # Flatten input
        quantized, encoding_indices, encodings, slots, logits = self.sample(inputs, unique, nunique, final=final)

        # Reset unused cb vectors...
        if update: self.update_usage(encoding_indices)


        # Loss
        loss = self.compute_baseloss(quantized, inputs, logits, avg, loss_type)

        # qk loss
        qkloss = 0
        if self.qk and (not final):
            qkloss += get_cb_variance(self.mu_embeddings.weight)
            qkloss += 0.01*torch.mean(torch.norm(self.sigma_embeddings.weight, dim=1))
        #     slot_mu = self.mu_embeddings.weight
        #     slot_logvar = self.sigma_embeddings.weight


        loss += get_cb_variance(self._embedding.weight)
        loss += klloss
        loss += qkloss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        encoding_indices = encoding_indices.view(input_shape[0], -1)


        return loss, quantized, perplexity, encoding_indices, slots





class VectorQuantizerEMA(BaseVectorQuantizer):
    def __init__(self, num_embeddings, 
                        embedding_dim, 
                        nhidden,
                        commitment_cost, 
                        decay, 
                        epsilon=1e-5,
                        usage_threshold=1.0e-9,
                        variational=False,
                        qk=False,
                        cosine=False,
                        gumble=False,
                        temperature=1.0,
                        kld_scale=1.0):
        super(VectorQuantizerEMA, self).__init__(num_embeddings, 
                                                    embedding_dim,
                                                    nhidden, 
                                                    commitment_cost,
                                                    usage_threshold,
                                                    variational,
                                                    qk,
                                                    cosine,
                                                    gumble,
                                                    temperature,
                                                    kld_scale)
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._usage_threshold = usage_threshold
        self.variational = variational
        self._cosine = cosine
        self.qk = qk

       

        # ======================
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon


    def forward(self, inputs, avg=False, 
                        unique = False,
                        nunique = -1,
                        update=False,
                        loss_type=0,
                        final=False,
                        reset_usage=False):
        input_shape = inputs.shape

        # Restart vectors
        if update:
            if np.random.uniform() > 0.99: self.random_restart()
            if reset_usage: self.reset_usage()
        
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


        quantized, encoding_indices, encodings, slots, logits = self.sample(inputs, unique, nunique, final=final)

        # Reset unused cb vectors...
        if update: self.update_usage(encoding_indices)


        # Use EMA to update the embedding vectors
        if self.training and (not final):
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = ((self._ema_cluster_size + self._epsilon)
                                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), self.data_mean.repeat(input_shape[0], 1))
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        

        loss = self.compute_baseloss(quantized, inputs, logits, avg, loss_type)

        # qk loss
        qkloss = 0
        if self.qk and (not final):
            qkloss += get_cb_variance(self.mu_embeddings.weight)
            qkloss += 0.01*torch.mean(torch.norm(self.sigma_embeddings.weight, dim=1))
        #     slot_mu = self.mu_embeddings.weight
        #     slot_logvar = self.sigma_embeddings.weight


        loss += get_cb_variance(self._embedding.weight)
        loss += klloss
        loss += qkloss


        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        encoding_indices = encoding_indices.view(input_shape[0], -1)
        
        return loss, quantized, perplexity, encoding_indices, slots