### LIBRARIES ###
# Global libraries
from copy import deepcopy

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
        self.attenuation_coef = 0.25

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
            importance_indexes = self.importance_index(question_attention)
            list_importance_indexes.append(importance_indexes)

        return torch.stack(list_importance_indexes)

    def compute_graph_representation(self, conceptnet_graph, num_max_nodes):
        list_main_entities = conceptnet_graph.select_top_edges(num_max_nodes)
        kg_emb = []
        for entity in list_main_entities:
            kg_emb.append(
                self.txt_embedding.conceptnet_embedding.get_node_embedding_tensor(
                    str(entity)
                )
            )
        return torch.stack(kg_emb)

    def forward(
        self, list_questions, attention_question, basic_conceptnet_graph, num_max_nodes
    ):
        """
            Refines `conceptnet_graph`, using the `question` and its `attention_question`
        """
        list_importance_indexes = self.compute_importance_index(attention_question)
        list_questions = torch.stack(list_questions)

        list_kg_embeddings = []

        # Check that the two tensors are in the right GPU
        list_questions = (
            list_questions.cuda(attention_question.get_device())
            if attention_question.is_cuda
            else list_questions
        )
        list_importance_indexes = (
            list_importance_indexes.cuda(attention_question.get_device())
            if attention_question.is_cuda
            else list_importance_indexes
        )

        try:
            print("Shape of list_questions: ", list_questions.shape)
        except:
            print("List question is a list, not a tensor")

        try:
            print("Shape of list_importance_indexes: ", list_importance_indexes.shape)
        except:
            print("list_importance_indexes is a list, not a tensor")

        # Update the weights in the graph
        # TODO: Try to find a way to compute it faster with less memory
        print("Starting propagation")
        for i, question in enumerate(list_questions):
            print("New question (device: " + str(attention_question.get_device()) + ")")
            conceptnet_graph = deepcopy(basic_conceptnet_graph)
            conceptnet_graph.to(attention_question.get_device())
            for j, entity in enumerate(question):
                # Initialize the edges
                for edge in conceptnet_graph.weight_edges:
                    conceptnet_graph.weight_edges[edge]["updated"] = False
                # Propagate the weights for this entity
                conceptnet_graph.propagate_weights(
                    [(entity, list_importance_indexes[i][j])],
                    self.propagation_threshold,
                    self.attenuation_coef,
                )
            question_graph_embedding = self.compute_graph_representation(
                conceptnet_graph, num_max_nodes
            )
            list_kg_embeddings.append(question_graph_embedding)

        return torch.stack(list_kg_embeddings)
