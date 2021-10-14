### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

# Custom libraries
from conceptBert.kilbert import BCNet

### CLASS DEFINITION ###
class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[0.2, 0.5]):
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(
            BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3),
            name="h_mat",
            dim=None,
        )

    def forward(self, q1, q2, v_mask=True):
        """
            q1: [batch, k, vdim]
            q2: [batch, qdim]
        """
        p, logits = self.forward_all(q1, q2, v_mask)
        return p, logits

    def forward_all(self, q1, q2, q1_mask=True):
        q1_num = q1.size(1)
        q2_num = q2.size(1)
        logits = self.logits(q1, q2)  # b x g x v x q

        if q1_mask:
            mask = (
                (0 == q1.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            )
            logits.data.masked_fill_(mask.data, -float("inf"))

        p = nn.functional.softmax(logits.view(-1, self.glimpse, q1_num * q2_num), 2)
        return p.view(-1, self.glimpse, q1_num, q2_num), logits
