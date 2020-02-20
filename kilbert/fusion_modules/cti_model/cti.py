### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn

# Custom libraries
from fusion_modules.cti_model.fcnet import FCNet
from fusion_modules.cti_model.tcnet import TCNet
from fusion_modules.cti_model.triattention import TriAttention

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
            # TODO: Problem here: it's not q_dim or kg_dim, it's 1024 (num_hid <=> v_dim)
            q_prj.append(FCNet([h_dim * 2, h_dim * 2], "", 0.2))
            kg_prj.append(FCNet([h_dim * 2, h_dim * 2], "", 0.2))

        self.t_net = nn.ModuleList(t_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.kg_prj = nn.ModuleList(kg_prj)

    # def forward(self, v, q, kg):
    def forward(self, v_emb, q_emb, kg_emb):
        """
            v: [batch, num_objs, obj_dim]
            b: [batch, num_objs, b_dim]
            q: [batch_size, seq_length]
        """
        b_emb = [0] * self.glimpse
        att, logits = self.t_att(v_emb, q_emb, kg_emb)

        try:
            print("Shape att: ", att.shape)
        except:
            pass
        try:
            print("Shape logits: ", logits.shape)
        except:
            pass

        for g in range(self.glimpse):
            b_emb[g] = self.t_net[g].forward_with_weights(
                v_emb, q_emb, kg_emb, att[:, :, :, :, g]
            )
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            kg_emb = self.kg_prj[g](b_emb[g].unsqueeze(1)) + kg_emb

        joint_emb = q_emb.sum(1) + kg_emb.sum(1)
        return joint_emb, att


"""
def build_cti(args, dataset):
    t_att = TriAttention(
        dataset.v_dim,
        args.num_hid,
        args.num_hid,
        args.h_mm,
        1,
        args.rank,
        args.gamma,
        args.k,
        dropout=[0.2, 0.5],
    )

    t_net = []
    q_prj = []
    a_prj = []

    for i in range(args.gamma):
        t_net.append(
            TCNet(
                dataset.v_dim,
                args.num_hid,
                args.num_hid,
                args.h_mm,
                args.h_out,
                args.rank,
                1,
                k=2,
                dropout=[0.2, 0.5],
            )
        )
        q_prj.append(FCNet([args.num_hid, args.num_hid], "", 0.2))
        a_prj.append(FCNet([args.num_hid, args.num_hid], "", 0.2))

    return CTIModel(
        dataset,
        w_emb,
        q_emb,
        wa_emb,
        ans_emb,
        t_att,
        t_net,
        q_prj,
        a_prj,
        classifier,
        args.op,
        args.gamma,
    )
"""
