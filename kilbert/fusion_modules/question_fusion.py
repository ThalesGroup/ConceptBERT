### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn

# Custom libraries

### CLASS DEFINITION ###
class SimpleQuestionAddition(nn.Module):
    """
        Simply adds the question embedding and the text attention from ViLBERT
        and the Transformer
    """

    def __init__(self, config):
        # self.output_dim = [
        #     config.batch_size,
        #     config.hidden_size,
        # ]
        print("TBD")

    def forward(self, emb_vilbert, att_vilbert, emb_transformer, att_transformer):
        return emb_vilbert + emb_transformer, att_vilbert + att_transformer


class SimpleQuestionMultiplication(nn.Module):
    """
        Simply multiplies the question embedding and the text attention from ViLBERT
        and the Transformer
    """

    def __init__(self, config):
        # self.output_dim = [
        #     config.batch_size,
        #     config.hidden_size,
        # ]
        print("TBD")

    def forward(self, emb_vilbert, att_vilbert, emb_transformer, att_transformer):
        return emb_vilbert * emb_transformer, att_vilbert * att_transformer


class SimpleQuestionConcatenation(nn.Module):
    """
        Simply concatenates the question embedding and the text attention from ViLBERT
        and the Transformer
    """

    def __init__(self, config):
        # self.output_dim = [
        #     config.batch_size,
        #     config.hidden_size,
        #  ]
        print("TBD")

    def forward(self, emb_vilbert, att_vilbert, emb_transformer, att_transformer):
        return (
            torch.cat(emb_vilbert, emb_transformer),
            torch.cat(att_vilbert, att_transformer),
        )
