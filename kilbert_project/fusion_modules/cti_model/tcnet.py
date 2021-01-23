### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn

# Custom libraries
from kilbert_project.fusion_modules.cti_model.fcnet import FCNet

### UTILS FUNCTION DEFINITION ###
def mode_product(tensor, matrix_1, matrix_2, matrix_3, matrix_4, n_way=3):

    # mode-1 tensor product
    tensor_1 = (
        tensor.transpose(3, 2)
        .contiguous()
        .view(
            tensor.size(0),
            tensor.size(1),
            tensor.size(2) * tensor.size(3) * tensor.size(4),
        )
    )
    tensor_product = torch.matmul(matrix_1, tensor_1)
    tensor_1 = tensor_product.view(
        -1, tensor_product.size(1), tensor.size(4), tensor.size(3), tensor.size(2)
    ).transpose(4, 2)

    # mode-2 tensor product
    tensor_2 = (
        tensor_1.transpose(2, 1)
        .transpose(4, 2)
        .contiguous()
        .view(
            -1, tensor_1.size(2), tensor_1.size(1) * tensor_1.size(3) * tensor_1.size(4)
        )
    )
    tensor_product = torch.matmul(matrix_2, tensor_2.float())
    tensor_2 = (
        tensor_product.view(
            -1,
            tensor_product.size(1),
            tensor_1.size(4),
            tensor_1.size(3),
            tensor_1.size(1),
        )
        .transpose(4, 1)
        .transpose(4, 2)
    )
    tensor_product = tensor_2

    if n_way > 2:
        # mode-3 tensor product
        tensor_3 = (
            tensor_2.transpose(3, 1)
            .transpose(4, 2)
            .transpose(4, 3)
            .contiguous()
            .view(
                -1,
                tensor_2.size(3),
                tensor_2.size(2) * tensor_2.size(1) * tensor_2.size(4),
            )
        )
        tensor_product = torch.matmul(matrix_3, tensor_3.float())
        tensor_3 = (
            tensor_product.view(
                -1,
                tensor_product.size(1),
                tensor_2.size(4),
                tensor_2.size(2),
                tensor_2.size(1),
            )
            .transpose(1, 4)
            .transpose(4, 2)
            .transpose(3, 2)
        )
        tensor_product = tensor_3

    if n_way > 3:
        # mode-4 tensor product
        tensor_4 = (
            tensor_3.transpose(4, 1)
            .transpose(3, 2)
            .contiguous()
            .view(
                -1,
                tensor_3.size(4),
                tensor_3.size(3) * tensor_3.size(2) * tensor_3.size(1),
            )
        )
        tensor_product = torch.matmul(matrix_4, tensor_4)
        tensor_4 = (
            tensor_product.view(
                -1,
                tensor_product.size(1),
                tensor_3.size(3),
                tensor_3.size(2),
                tensor_3.size(1),
            )
            .transpose(4, 1)
            .transpose(3, 2)
        )
        tensor_product = tensor_4

    return tensor_product


### CLASS DEFINITION ###
class TCNet(nn.Module):
    def __init__(
        self,
        v_dim,
        q_dim,
        kg_dim,
        h_dim,
        h_out,
        rank,
        glimpse,
        act="ReLU",
        k=1,
        dropout=[0.2, 0.5, 0.2],
    ):
        super(TCNet, self).__init__()

        self.v_dim = v_dim
        self.q_dim = q_dim
        self.kg_dim = kg_dim
        self.h_out = h_out
        self.rank = rank
        self.h_dim = h_dim * k
        self.hv_dim = int(h_dim / rank)
        self.hq_dim = int(h_dim / rank)
        self.hkg_dim = int(h_dim / rank)

        self.q_tucker = FCNet([q_dim, self.h_dim], act=act, dropout=dropout[0])
        self.v_tucker = FCNet([v_dim, self.h_dim], act=act, dropout=dropout[1])
        self.kg_tucker = FCNet([kg_dim, self.h_dim], act=act, dropout=dropout[2])

        if self.h_dim < 1024:
            self.kg_tucker = FCNet([kg_dim, self.h_dim], act=act, dropout=dropout[2])
            self.q_net = nn.ModuleList(
                [
                    FCNet([self.h_dim, self.hq_dim], act=act, dropout=dropout[0])
                    for _ in range(rank)
                ]
            )
            self.v_net = nn.ModuleList(
                [
                    FCNet([self.h_dim, self.hv_dim], act=act, dropout=dropout[1])
                    for _ in range(rank)
                ]
            )
            self.kg_net = nn.ModuleList(
                [
                    FCNet([self.h_dim, self.hkg_dim], act=act, dropout=dropout[2])
                    for _ in range(rank)
                ]
            )

            if h_out > 1:
                self.ho_dim = int(h_out / rank)
                h_out = self.ho_dim

            self.T_g = nn.Parameter(
                torch.Tensor(
                    1, rank, self.hv_dim, self.hq_dim, self.hkg_dim, glimpse, h_out
                ).normal_()
            )
        self.dropout = nn.Dropout(dropout[1])

    def forward(self, v, q, kg):
        f_emb = 0
        v_tucker = self.v_tucker(v)
        q_tucker = self.q_tucker(q)
        kg_tucker = self.kg_tucker(kg)

        for r in range(self.rank):
            v_ = self.v_net[r](v_tucker)
            q_ = self.q_net[r](q_tucker)
            kg_ = self.kg_net[r](kg_tucker)
            f_emb = (
                mode_product(self.T_g[:, r, :, :, :, :, :], v_, q_, kg_, None) + f_emb
            )

        return f_emb.squeeze(4)

    def forward_with_weights(self, v, q, kg, w):
        v_ = self.v_tucker(v).transpose(2, 1)  # b x d x v
        q_ = self.q_tucker(q).transpose(2, 1).unsqueeze(3)  # b x d x q x 1
        kg_ = self.kg_tucker(kg).transpose(2, 1).unsqueeze(3)  # b x d x kg

        logits = torch.einsum("bdv,bvqa,bdqi,bdaj->bdij", [v_, w, q_, kg_])
        logits = logits.squeeze(3).squeeze(2)

        return logits
