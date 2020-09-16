import torch

from torchfm.layer import (
    FeaturesEmbedding, 
    FeaturesLinear, 
    MultiLayerPerceptron
)
from torchfm.attention_layer import CrossAttentionNetwork

class DeepCrossAttentionalNetworkModel(torch.nn.Module):
    """
    A pytorch implementation of Multihead Attention Factorization Machine Model.

    Reference: on going
    """

    def __init__(self, field_dims, embed_dim, num_heads, ffn_embed_dim, num_layers, mlp_dims, dropout):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.can = CrossAttentionNetwork(self.num_fields, embed_dim, num_heads, ffn_embed_dim, num_layers, dropout, ensemble='addition')
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=True)
        # self.fc = torch.nn.Linear(mlp_dims[-1] + embed_dim, 1)

        self._reset_parameters()

    def generate_square_subsequent_mask(self, num_fields):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(num_fields, num_fields)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        device = x.device
        attn_mask = self.generate_square_subsequent_mask(x.size(1)).to(device)
        embed_x = self.embedding(x)
        x_l1 = self.can(embed_x, attn_mask=attn_mask)
        h_l2 = self.mlp(embed_x.view(-1, self.embed_output_dim))
        # x_stack = torch.cat([x_l1, h_l2], dim=1)
        # x = self.linear(x) + self.fc(x_stack)
        x = self.linear(x) + x_l1 + h_l2
        return torch.sigmoid(x.squeeze(1))
