import torch

from torchfm.layer import FeaturesEmbedding, FeaturesLinear, CrossAttentionalProductNetwork, MultiLayerPerceptron


class DeepCrossAttentionalProductNetwork(torch.nn.Module):
    """
    A pytorch implementation of inner/outer Product Neural Network.
    Reference:
        Y Qu, et al. Product-based Neural Networks for User Response Prediction, 2016.
    """

    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout):
        super().__init__()
        num_fields = len(field_dims)
        self.cap = CrossAttentionalProductNetwork(num_fields, embed_dim, num_layers)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.num_layers = num_layers
        # self.linear = FeaturesLinear(field_dims)
        self.embed_output_dim = num_fields * embed_dim
        self.attn_output_dim = (num_layers * num_fields * (num_fields + 1) // 2) * embed_dim
        self.mlp = MultiLayerPerceptron(embed_dim + self.embed_output_dim, mlp_dims, dropout)
        # self.mlp_attn = MultiLayerPerceptron(self.attn_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        cross_term = self.cap(embed_x)
        # y = self.mlp_attn(cross_term.view(-1, self.attn_output_dim))
        # x = y + self.mlp(embed_x.view(-1, self.embed_output_dim))
        y = torch.cat([embed_x.view(-1, self.embed_output_dim), cross_term], dim=1)
        x = self.mlp(y)
        # x = self.mlp(embed_x.view(-1, self.embed_output_dim)) + 
        return torch.sigmoid(x.squeeze(1))
