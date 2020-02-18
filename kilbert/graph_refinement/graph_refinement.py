### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn

# Custom libraries
from graph_refinement.importance_index import ImportanceIndex

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

    def compute_importance_index(self, list_question_attention):
        """
            Given a sentence, computes the importance index of each word

            Input: 
                - `sentence` (str): sentence to process
            Output: 
                - `list_importance_indexes` (List[float]): list of the importance index of each word
        """
        # Given the word and its attention, computes the importance indexes of each one

        """
        list_question_attention = list_question_attention.tolist()

        list_importance_indexes = []
        for i, question in enumerate(list_questions):
            question_importance_indexes = []
            for j, word in enumerate(question):
                attention_word = list_question_attention[i][j]
                question_importance_indexes.append(
                    self.importance_index(attention_word)
                )
            list_importance_indexes.append(question_importance_indexes)
        """

        list_importance_indexes = []
        for question_attention in list_question_attention:
            try:
                print("Shape of question_attention: ", question_attention.shape)
            except:
                print("Length of question_attention: ", len(question_attention))
            importance_indexes = self.importance_index(question_attention)
            list_importance_indexes.append(importance_indexes)

        print("LIST_IMPORTANCE_INDEXES: ", list_importance_indexes)

        return list_importance_indexes

    def forward(self, list_questions, attention_question, conceptnet_graph):
        """
            Refines `conceptnet_graph`, using the `question` and its `attention_question`
        """
        list_importance_indexes = self.compute_importance_index(attention_question)

        # Update the weights in the graph
        # TODO: Try to find a way to compute it faster with less memory
        for question in list_questions:
            for i, entity in enumerate(question):
                # Initialize the edges
                for edge in self.weight_edges:
                    conceptnet_graph.weight_edges[edge]["updated"] = False
                # Propagate the weights for this entity
                conceptnet_graph.propagate_weights(
                    [(entity, list_importance_indexes[i])],
                    self.propagation_threshold,
                    self.attenuation_coef,
                )

