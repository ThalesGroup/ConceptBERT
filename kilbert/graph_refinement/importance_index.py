### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn

### CLASS DEFINITION ###
class ImportanceIndex(nn.Module):
    """

    """

    def __init__(self):
        super(ImportanceIndex, self).__init__()

        # self.dense = nn.Linear(16, 16)
        # self.activation = nn.LeakyReLU(-0.1)

    # def forward(self, word, attention_word):
    #     """
    #         Computes the importance index of the given word
    #     """
    #     importance_idx = self.dense(torch.Tensor([attention_word]))
    #     scaled_importance_idx = self.activation(importance_idx)
    #     return scaled_importance_idx

    def forward(self, attention_word):
        """
            Computes the importance index of the given word
        """
        # print(
        #     "Attention word on device " + str(attention_word.get_device()) + " : ",
        #     attention_word,
        # )
        # importance_idx = self.dense(attention_word)
        # scaled_importance_idx = self.activation(importance_idx)
        # return scaled_importance_idx
        return 250 * attention_word
