### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn

# Custom libraries
from kilbert_project.kilbert import FCNet
from kilbert_project.kilbert import TCNet
from kilbert_project.kilbert import TriAttention

### CLASS DEFINITION ###
class CTIModel(nn.Module):
    """
        Instance of a Compact Trilinear Interaction model (see https://arxiv.org/pdf/1909.11874.pdf)
    """

    def __init__(
        self, v_dim, q_dim, kg_dim, glimpse, h_dim=512, h_out=1, rank=32, k=1,
    ):
        super(CTIModel, self).__init__()

        self.glimpse = glimpse

        self.t_att = TriAttention(
            v_dim, q_dim, kg_dim, h_dim, 1, rank, glimpse, k, dropout=[0.2, 0.5, 0.2],
        )

        t_net = []
        q_prj = []
        kg_prj = []
        for _ in range(glimpse):
            t_net.append(
                TCNet(
                    v_dim,
                    q_dim,
                    kg_dim,
                    h_dim,
                    h_out,
                    rank,
                    1,
                    k=2,
                    dropout=[0.2, 0.5, 0.2],
                )
            )
            q_prj.append(FCNet([h_dim * 2, h_dim * 2], "", 0.2))
            kg_prj.append(FCNet([h_dim * 2, h_dim * 2], "", 0.2))

        self.t_net = nn.ModuleList(t_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.kg_prj = nn.ModuleList(kg_prj)

        self.q_pooler = FCNet([q_dim, h_dim * 2])
        self.kg_pooler = FCNet([kg_dim, h_dim * 2])

    # def forward(self, v, q, kg):
    def forward(self, v_emb, q_emb_raw, kg_emb_raw):
        """
            v: [batch, num_objs, obj_dim]
            b: [batch, num_objs, b_dim]
            q: [batch_size, seq_length]
        """
        b_emb = [0] * self.glimpse
        att, logits = self.t_att(v_emb, q_emb_raw, kg_emb_raw)

        q_emb = self.q_pooler(q_emb_raw)
        kg_emb = self.kg_pooler(kg_emb_raw)

        for g in range(self.glimpse):
            b_emb[g] = self.t_net[g].forward_with_weights(
                v_emb, q_emb_raw, kg_emb_raw, att[:, :, :, :, g]
            )

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            kg_emb = self.kg_prj[g](b_emb[g].unsqueeze(1)) + kg_emb

        joint_emb = q_emb.sum(1) + kg_emb.sum(1)
        return joint_emb, att
