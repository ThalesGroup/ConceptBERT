### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn

# Custom libraries
from importance_index import ImportanceIndex

### CLASS DEFINITION ###
class GraphRefinement(nn.Module):
    """
        Model "G1" 
    """

    def __init__(self):
        super(GraphRefinement, self).__init__()

        # Module to compute the importance index
        self.importance_index = ImportanceIndex()
        # Won't propagate if the weight is smaller than this value
        self.propagation_threshold = 0.5
        # Coefficient multiplied to the weight at each iteration
        self.attenuation_coef = 0.5

    def compute_importance_index(self, sentence, list_word_attention):
        """
            Given a sentence, computes the importance index of each word

            Input: 
                - `sentence` (str): sentence to process
            Output: 
                - `list_importance_indexes` (List[float]): list of the importance index of each word
        """
        # TODO: Preprocess the question, so that the attention of any stop word or punctuation is 0

        # Given the word and its attention, computes the importance indexes of each one
        list_words = sentence.split(" ")
        list_importance_indexes = []
        for i, word in enumerate(list_words):
            attention_word = list_word_attention[i]
            list_importance_indexes.append(word, self.importance_index(attention_word))

        return list_words, list_importance_indexes

    def forward(self, question, attention_question, conceptnet_graph):
        """
            Refines `conceptnet_graph`, using the `question` and its `attention_question`
        """
        list_words, list_importance_indexes = self.compute_importance_index(
            question, attention_question
        )

        # Convert the words to entities (in place to save memory)
        for i, word in enumerate(len(list_words)):
            list_words[i] = conceptnet_graph.get_index(word)

        # Update the weights in the graph
        # TODO: Try to find a way to compute it faster with less memory
        for i, entity in enumerate(list_words):
            # Initialize the edges
            for edge in self.weight_edges:
                conceptnet_graph.weight_edges[edge]["updated"] = False
            # Propagate the weights for this entity
            conceptnet_graph.propagate_weights(
                [(entity, list_importance_indexes[i])],
                self.propagation_threshold,
                self.attenuation_coef,
            )

