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



def sorting_idxs(quantized, cidxs):
    B, N, _ = quantized.shape

    batch_sampled_vectors = []
    for b in range(B):
        sampled_idxs = cidxs[b, ...]

        st_pointer = -1
        end_pointer = 0
        unique = torch.zeros_like(sampled_idxs)
        for unique_idx in torch.sort(torch.unique(sampled_idxs)):
            idxs = torch.argwhere(sampled_idxs == unique_idx)
            
            st_pointer += 1
            end_pointer -= len(idxs[1:])

            unique[st_pointer] = idxs[0]
            unique_idx[end_pointer : end_pointer + len(idxs[1:])] = idxs[1:]

            pass 


    return quantized, cidxs


def get_cosine_distance(u, v):
    # distance on sphere
    d = torch.einsum('bd,dn->bn', u, rearrange(v, 'n d -> d n'))
    ed1 = torch.sqrt(torch.sum(u**2, dim=1, keepdim=True))
    ed2 = torch.sqrt(torch.sum(v**2, dim=1, keepdim=True))
    ed3 = torch.einsum('bd,dn->bn', ed1, rearrange(ed2, 'n d  -> d n'))
    geod = torch.clamp(d/(ed3), min=-0.99999, max=0.99999)
    return torch.acos(torch.abs(geod))/(2.0*np.pi)


def get_cb_variance(cb):
    # cb = cb / (1e-5 + torch.norm(cb, dim=1, keepdim=True))
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
            self.loss_fn = F.mse_loss # lambda x1,x2: 2.0*(1 - F.cosine_similarity(x1, x2).mean())

        if self.qk:
            self.mu_embeddings = nn.Embedding(self._num_embeddings, self._embedding_dim)
            self.sigma_embeddings = nn.Embedding(self._num_embeddings, self._embedding_dim)
            nn.init.xavier_uniform_(self.mu_embeddings.weight)
            nn.init.xavier_uniform_(self.sigma_embeddings.weight)
            # nn.init.constant_(self.sigma_embeddings.weight, 0)

        self.data_mean = 0
        self.data_std = 0

        self.register_buffer('_usage', torch.ones(self._num_embeddings), persistent=False)
        self.get_variance = lambda: get_cb_variance(self._embedding.weight) 



        self.kld_scale = kld_scale


        # ======================
        # Gumble parameters
        self.gumble = gumble
        if self.gumble:
            self.temperature = temperature
            self.gumble_proj = nn.Sequential(
                                    nn.Linear(nhidden, num_embeddings),
                                    nn.ReLU())
            if self.qk:
                self.gumble_slot_proj = nn.Sequential(
                                    nn.Linear(nhidden, num_embeddings),
                                    nn.ReLU())


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
            rand_codes = eps*self.data_std.unsqueeze(0).repeat(len(dead_codes), 1) +\
                                self.data_mean.unsqueeze(0).repeat(len(dead_codes), 1)

            with torch.no_grad():
                self._embedding.weight[dead_codes] = rand_codes
                self.mu_embeddings.weight[dead_codes] = rand_codes


            self._embedding.weight.requires_grad = True
            self.mu_embeddings.weight.requires_grad = True

    def entire_cb_restart(self):
        N = self.data_std.shape[0]
        eps = torch.randn((self._num_embeddings, self._embedding_dim)).to(self._embedding.weight.device)
        rand_codes = eps*self.data_std.unsqueeze(0).repeat(len(eps), 1) +\
                                self.data_mean.unsqueeze(0).repeat(len(eps), 1)


        with torch.no_grad():
            self._embedding.weight[:] = rand_codes
            self.mu_embeddings.weight[:] = rand_codes

        self._embedding.weight.requires_grad = True
        self.mu_embeddings.weight.requires_grad = True



    def sample_mu_sigma_given_encodings(self, quantized, encodings, shape):
        slot_mu = torch.matmul(encodings, self.mu_embeddings.weight)
        slot_sigma = torch.matmul(encodings, self.sigma_embeddings.weight)

        # slot_sigma = torch.clamp(torch.exp(0.5*slot_sigma), min=1e-3, max=1)
        slots = slot_mu + slot_sigma * torch.randn(slot_sigma.shape, 
                                                device = slot_sigma.device, 
                                                dtype = slot_sigma.dtype)
        slots =  slots.view(shape) # + quantized.view(shape).clone().detach()
        slot_mu = slot_mu.view(shape) # + quantized.view(shape).clone().detach()
        
        slot_sigma = slot_sigma.view(shape)
        return slots, slot_mu, slot_sigma


    def vq_sample(self, features, st = False, idxs=None, final=False):
        input_shape = features.shape
        
        # update data stats
        if not st:
            self.data_mean = 0.9*self.data_mean + 0.1*features.clone().detach().mean(1).mean(0)
            self.data_std = 0.9*self.data_std + 0.1*features.clone().detach().std(1).mean(0)


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

        # Flatten features
        features = features.view(-1, self._embedding_dim)


        # Update prior with previosuly wrt principle components
        # This will ensure that codebook indicies that are not in idxs wont be sampled
        # hacky way need to find a better way of implementing this
        key_codebook = self._embedding.weight.clone()
        if not (idxs is None):
            idxs = torch.unique(idxs).reshape(-1, 1)
            for i in range(self._num_embeddings):
                if not (i in idxs): 
                    key_codebook[i, :] = 2*torch.max(features) 


        # Quantize and unflatten
        distances = self._get_distance(features, key_codebook)
        encoding_indices = _min_encoding_(distances)

        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=features.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight)
        slot_mu = None; slot_sigma = None

        slots = None
        # slot sampling
        if self.qk and (not final):
            slots, slot_mu, slot_sigma = self.sample_mu_sigma_given_encodings(quantized, encodings, input_shape)

        return quantized.view(input_shape), encoding_indices, encodings, slots, None, slot_sigma, slot_mu


    def gumble_sample(self, features, 
                        st=False, idxs = None, 
                        final=False):
        
        input_shape = features.shape

        # force hard = True when we are in eval mode, as we must quantize
        hard = st if self.training else True


        # update data stats
        if not st:
            self.data_mean = 0.9*self.data_mean + 0.1*features.clone().detach().mean(1).mean(0)
            self.data_std = 0.9*self.data_std + 0.1*features.clone().detach().std(1).mean(0)


        logits = self.gumble_proj(features)
        
        # Update prior with previosuly wrt principle components
        if not (idxs is None):
            mask_idxs = torch.zeros_like(logits)

            for b in range(input_shape[0]):
                mask_idxs[b, :, idxs[b]] = 1
            logits = mask_idxs*logits

        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard)
        quantized = torch.einsum('b t k, k d -> b t d', soft_one_hot, self._embedding.weight)            


        encoding_indices = soft_one_hot.argmax(dim=-1).view(-1, 1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=features.device)
        encodings.scatter_(1, encoding_indices, 1)
        slot_mu = None; slot_sigma = None


        slots = None
        # slot sampling
        if self.qk and (not final):
            slots, slot_mu, slot_sigma = self.sample_mu_sigma_given_encodings(quantized, encodings, input_shape)

       
        return quantized, encoding_indices, encodings, slots, logits, slot_sigma, slot_mu



    def sample(self, features, 
                        st=False, 
                        idxs = None, 
                        final=False,
                        from_train = False):
        if not from_train:
            features = self.proj(features)
            # layer norm on features
            features = self.norm_input(features)

        if self.gumble:
            return self.gumble_sample(features, st=st, idxs=idxs, 
                                        final=final)
        else:
            return self.vq_sample(features, 
                                    st=st, idxs=idxs, final=final)


    def compute_baseloss(self, quantized, inputs, logits=None, avg=False, loss_type=0):
        if self.gumble:
            # + kl divergence to the prior loss
            if (logits is None):
                loss = 0
            else:
                qy = F.softmax(logits, dim=-1)
                loss = self.kld_scale * torch.sum(qy * torch.log(qy * self._num_embeddings + 1e-10), dim=1).mean()

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



    def compute_qkloss(self, final, encodings, inputs, avg, loss_type, logvar=None, mean=None):
        # qk loss
        input_shape = inputs.shape

        qkloss = 0
        if final:
            # may not be required (push K embeddings towards slots)
            kquantized = torch.matmul(encodings, self._embedding.weight)
            qkloss += self.compute_baseloss(kquantized.view(input_shape), 
                                                inputs, None, avg, loss_type)
            
            qkloss += self.loss_fn(inputs.clone().detach(), mean.view(input_shape))
        else:
            qkloss += get_cb_variance(self.mu_embeddings.weight)
            # qkloss += 0.001 * torch.mean(torch.norm(self.sigma_embeddings.weight, dim=1))

            # kl loss between sampled marginal distributions
            qkloss += torch.mean(-0.5 * (1 + logvar - mean ** 2 - logvar.exp()))
           
        return qkloss


    def forward(self, *args):
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
                        idxs=None,
                        reset_usage=False):

        input_shape = inputs.shape

        quantized, encoding_indices, encodings, slots, logits, logvar, mean = self.sample(inputs, unique, nunique, idxs=idxs,final=final)

        # Restart vectors
        if final and update:
            if np.random.uniform() > 0.99: self.random_restart()
            if reset_usage: self.reset_usage()


        # Reset unused cb vectors...
        if final and update: self.update_usage(encoding_indices)


        # Loss
        loss = self.compute_baseloss(quantized, inputs, logits, avg, loss_type)


        if not (loss_type == 2):
            if self.qk:
                qkloss = self.compute_qkloss(final, encodings, inputs, avg, loss_type, logvar, mean)
            
            loss += get_cb_variance(self._embedding.weight)
            loss += qkloss


        # Straight Through Estimator
        if not self.gumble: quantized = inputs + (quantized - inputs).detach()
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


    def forward(self, inputs, 
                        avg=False, 
                        update=False,
                        loss_type=0,
                        final=False,
                        idxs=None,
                        reset_usage=False):
        input_shape = inputs.shape

        inputs = self.proj(inputs)
        # layer norm on features
        inputs = self.norm_input(inputs)

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        quantized, encoding_indices, encodings, slots, logits, logvar, mean  = self.sample(inputs, 
                                                                                            idxs=idxs, 
                                                                                            final=final,
                                                                                            from_train=True)


        # ===================================
        # Tricks to prevent codebook collapse---
        # Restart vectors
        if update:
            if np.random.uniform() > 0.99: self.random_restart()
            if reset_usage: self.reset_usage()


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
        
        # ============================


        loss = self.compute_baseloss(quantized, inputs, logits, avg, loss_type)

        if not (loss_type == 2):
            if self.qk:
                qkloss = 0.01*self.compute_qkloss(final, 
                                                encodings, 
                                                inputs, 
                                                avg, 
                                                loss_type, 
                                                logvar, 
                                                mean)
            
            # loss += get_cb_variance(self._embedding.weight)
            # print (f'feature: {inputs.max()}, qfeatures: {quantized.max()}, quant loss: {loss}, QKloss: {qkloss}, key: {torch.max(self._embedding.weight)}, mu: {torch.max(self.mu_embeddings.weight)}, sigma: {torch.max(self.sigma_embeddings.weight)}')
            loss += qkloss


        # Straight Through Estimator
        if not self.gumble: quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        encoding_indices = encoding_indices.view(input_shape[0], -1)
        
        return loss, quantized, perplexity, encoding_indices, slots, mean