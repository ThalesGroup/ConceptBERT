### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn

# Custom libraries
from kilbert_project.kilbert import BiAttention
from kilbert_project.kilbert import BCNet
from kilbert_project.kilbert import FCNet

### CLASS DEFINITION ###
class BANModel(nn.Module):
    def __init__(
        self, q1_relation_dim, num_hid, gamma,
    ):
        super(BANModel, self).__init__()

        self.q1_att = BiAttention(q1_relation_dim, num_hid, num_hid, gamma)
        self.glimpse = gamma

        b_net = []
        q_prj = []
        q_att = []
        v_prj = []

        for _ in range(gamma):
            b_net.append(BCNet(q1_relation_dim, num_hid, num_hid, None, k=1))
            q_prj.append(FCNet([num_hid, num_hid], "", 0.2))

        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.q_att = nn.ModuleList(q_att)
        self.v_prj = nn.ModuleList(v_prj)

    def forward(self, q1_emb, q2_emb, b):
        b_emb = [0] * self.glimpse  # b x g x v x q
        att, att_logits = self.q1_att.forward_all(q1_emb, q2_emb)

        for g in range(self.glimpse):
            # b x l x h
            b_emb[g] = self.b_net[g].forward_with_weights(
                q1_emb, q2_emb, att[:, g, :, :]
            )
            q2_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q2_emb

        joint_emb = q2_emb.sum(1)
        return joint_emb, att

