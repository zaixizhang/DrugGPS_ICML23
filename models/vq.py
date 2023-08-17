import numpy as np
import torch
from torch import nn
from torch.autograd import Function, Variable
import torch.nn.functional as F


class VQ(nn.Module):
    def __init__(
            self, 
            num_embeddings, 
            embedding_dim,
            device,
            decay=0.9):
        super(VQ, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self.device = device

        self.register_buffer('vq_embedding', torch.rand(self._num_embeddings, self._embedding_dim)) #normalized codebook
        self.register_buffer('vq_embedding_output', torch.rand(self._num_embeddings, self._embedding_dim)) # output codebook
        self.register_buffer('vq_cluster_size', torch.ones(num_embeddings))

        self._decay = decay
        self.bn = torch.nn.BatchNorm1d(self._embedding_dim, affine=False, momentum=None)

    def get_k(self) :
        return self.vq_embedding_output

    def get_v(self) :
        return self.vq_embedding_output

    def update(self, x):
        inputs_normalized = self.bn(x)
        embedding_normalized = self.vq_embedding

        # Calculate distances
        distances = (torch.sum(inputs_normalized ** 2, dim=1, keepdim=True)
                     + torch.sum(embedding_normalized ** 2, dim=1)
                     - 2 * torch.matmul(inputs_normalized, embedding_normalized.t()))

        # FindNearest
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(self.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Use momentum to update the embedding vectors
        dw = torch.matmul(encodings.t(), inputs_normalized)
        self.vq_embedding.data = self.vq_cluster_size.unsqueeze(1) * self.vq_embedding * self._decay + (1 - self._decay) * dw
        self.vq_cluster_size.data = self.vq_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)

        self.vq_embedding.data = self.vq_embedding.data / self.vq_cluster_size.unsqueeze(1)

        # Output
        running_std = torch.sqrt(self.bn.running_var + 1e-5).unsqueeze(dim=0)
        running_mean = self.bn.running_mean.unsqueeze(dim=0)
        self.vq_embedding_output.data = self.vq_embedding*running_std + running_mean

