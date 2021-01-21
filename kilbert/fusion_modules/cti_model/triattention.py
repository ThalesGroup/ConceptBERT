### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn

# Custom libraries
from kilbert.kilbert import TCNet

### CLASS DEFINITION ###
class TriAttention(nn.Module):
    def __init__(
        self,
        v_dim,
        q_dim,
        kg_dim,
        h_dim,
        h_out,
        rank,
        glimpse,
        k,
        dropout=[0.2, 0.5, 0.2],
    ):
        super(TriAttention, self).__init__()
        self.glimpse = glimpse
        self.TriAtt = TCNet(
            v_dim, q_dim, kg_dim, h_dim, h_out, rank, glimpse, dropout=dropout, k=k
        )

    def forward(self, v, q, kg):
        v_num = v.size(1)
        q_num = q.size(1)
        kg_num = kg.size(1)
        logits = self.TriAtt(v, q, kg)

        mask = (
            (0 == v.abs().sum(2))
            .unsqueeze(2)
            .unsqueeze(3)
            .unsqueeze(4)
            .expand(logits.size())
        )
        logits.data.masked_fill_(mask.data, -float("inf"))

        p = torch.softmax(
            logits.contiguous().view(-1, v_num * q_num * kg_num, self.glimpse), 1
        )

        return p.view(-1, v_num, q_num, kg_num, self.glimpse), logits
