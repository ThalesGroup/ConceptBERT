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
        # self.output_dim =
        print("TBD")

    def forward(
        self, question_emb, question_att, image_emb, image_att, conceptnet_graph
    ):
        # TODO: Convert the conceptnet_graph to a vector that can be concatenated
        print("TBD")
