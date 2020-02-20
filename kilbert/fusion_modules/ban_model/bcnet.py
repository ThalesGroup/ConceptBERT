### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

# Custom libraries
from fusion_modules.ban_model.fcnet import FCNet

### CLASS DEFINITION ###
class BCNet(nn.Module):
    """
        Simple class for non-linear bilinear connect network
    """

    def __init__(
        self, q1_dim, q2_dim, h_dim, h_out, act="ReLU", dropout=[0.2, 0.5], k=1
    ):
        super(BCNet, self).__init__()

        self.c = 32
        self.k = k
        self.q1_dim = q1_dim
        self.q2_dim = q2_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.q1_net = FCNet([q1_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q2_net = FCNet([q2_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])  # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if None == h_out:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(
                torch.Tensor(1, h_out, 1, h_dim * self.k).normal_()
            )
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, q1, q2):
        if None == self.h_out:
            q1_ = self.q1_net(q1).transpose(1, 2).unsqueeze(3)
            q2_ = self.q2_net(q2).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(q1_, q2_)  # b x h_dim x v x q
            logits = d_.transpose(1, 2).transpose(2, 3)  # b x v x q x h_dim
            return logits.sum(1).sum(1).unsqueeze(1)

        # broadcast Hadamard product, matrix-matrix production
        # fast computation but memory inefficient
        # epoch 1, time: 157.84
        elif self.h_out <= self.c:
            q1_ = self.dropout(self.q1_net(q1)).unsqueeze(1)
            q2_ = self.q2_net(q2)
            h_ = q2_ * self.h_mat  # broadcast, b x h_out x v x h_dim
            logits = torch.matmul(
                h_, q2_.unsqueeze(1).transpose(2, 3)
            )  # b x h_out x v x q
            logits = logits + self.h_bias
            return logits  # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
        # epoch 1, time: 304.87
        else:
            q1_ = self.dropout(self.q1_net(q1)).transpose(1, 2).unsqueeze(3)
            q2_ = self.q_net(q2).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(q1_, q2_)  # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            return logits.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q

    def forward_with_weights(self, q1, q2, w):
        q1_ = self.q1_net(q1).transpose(1, 2).unsqueeze(2)  # b x d x 1 x v
        q2_ = self.q2_net(q2).transpose(1, 2).unsqueeze(3)  # b x d x q x 1
        logits = torch.matmul(torch.matmul(q1_, w.unsqueeze(1)), q2_)  # b x d x 1 x 1
        logits = logits.squeeze(3).squeeze(2)
        if 1 < self.k:
            logits = logits.unsqueeze(1)  # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k  # sum-pooling
        return logits
