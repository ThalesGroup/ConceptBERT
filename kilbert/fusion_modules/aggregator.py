### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn

# Custom libraries

### CLASS DEFINITION ###
class SimpleConcatenation(nn.Module):
    """
        Concatenates all the inputs  
    """

    def __init__(self, config):
        super(SimpleConcatenation, self).__init__()

    def forward(
        self, question_emb, question_att, image_emb, image_att, knowledge_graph_emb
    ):
        # TODO: Convert the conceptnet_graph to a vector that can be concatenated
        return torch.cat(
            [question_emb, question_att, image_emb, image_att, knowledge_graph_emb]
        )
