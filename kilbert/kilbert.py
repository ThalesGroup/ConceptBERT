### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn

# Custom libraries
from embeddings import BertEmbeddings, BertImageEmbeddings
from graph_refinement.conceptnet_graph import ConceptNet

from vilbert.vilbert import VILBertForVLTasks
from q_kg_transformer.transformer import QuestionGraphTransformer
from graph_refinement.graph_refinement import GraphRefinement

from fusion_modules.question_fusion import (
    SimpleQuestionAddition,
    SimpleQuestionMultiplication,
    SimpleQuestionConcatenation,
)
from fusion_modules.aggregator import SimpleConcatenation
from fusion_modules.bertpooler import BertTextPooler, BertImagePooler

from classifier.classifier import SimpleClassifier

### VARIABLES ###
# Maximum number of nodes extracted from the knowledge graph (heaviest edges)
k = 20
# Which layer to use (-1 = last, -2 = second last, ...)
bert_layer_used = -2
# Whether to use the first token or all of them
use_pooled_output = True

### CLASS DEFINITION ###
class Kilbert(nn.Module):
    """

    """

    def __init__(self, config, num_labels, split, dropout_prob=0.1, default_gpu=True):
        super(Kilbert, self).__init__(config)
        # Variables
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_prob)

        # Embedding modules
        self.txt_embedding = BertEmbeddings(config, split)
        self.img_embedding = BertImageEmbeddings(config)

        # Main modules
        self.vilbert = VILBertForVLTasks(
            config, num_labels, split, dropout_prob, default_gpu
        )
        self.q_kg_transformer = QuestionGraphTransformer(
            config, split, dropout_prob, default_gpu
        )
        self.graph_refinement = GraphRefinement()

        # Fusion modules
        self.fusion_question = SimpleQuestionAddition(config)
        self.bert_text_pooler = BertTextPooler(config)
        self.bert_image_pooler = BertImagePooler(config)

        self.aggregator = SimpleConcatenation(config)

        # Prediction modules
        # self.vil_prediction = SimpleClassifier(in_dim, hid_dim, num_labels, 0.5)

    def forward(
        self,
        input_txt,
        input_imgs,
        image_loc,
        conceptnet_graph,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        co_attention_mask=None,
        output_all_encoded_layers=None,
    ):
        print("IN PROGRESS")
        ## Step 0: Preprocess the inputs
        if attention_mask is None:
            attention_mask = torch.ones_like(input_txt)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input)
        if image_attention_mask is None:
            image_attention_mask = torch.ones(
                input_imgs.size(0), input_imgs.size(1)
            ).type_as(input_txt)

        """
            We create a 3D attention mask from a 2D tensor mask.
            Sizes are [batch_size, 1, 1, to_seq_length]
            So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            This attention mask is more simple than the triangular masking of causal
            attention used in OpenAI GPT, we just need to prepare the broadcast 
            dimension here.
        """
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = image_attention_mask.unsqueeze(1).unsqueeze(2)

        """
            Since attention_mask is 1.0 for positions we want to attend and 0.0 for 
            masked positions, this operation will create a tensor which is 0.0 for 
            positions we want to attend and -10000.0 for masked positions.
            Since we are adding it to the raw scores before the softmax, this is
            effectively the same as removing these entirely.
        """
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0

        if co_attention_mask is None:
            co_attention_mask = torch.zeros(
                input_txt.size(0), input_imgs.size(1), input_txt.size(1)
            ).type_as(extended_image_attention_mask)

        extended_co_attention_mask = co_attention_mask.unsqueeze(1)

        extended_co_attention_mask = extended_co_attention_mask * 5.0
        extended_co_attention_mask = extended_co_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility

        ## Step 1: Prepare the inputs for the main modules
        # Get the text and knowledge graph embeddings
        txt_embedding, kg_embedding = self.txt_embedding(input_txt, token_type_ids)
        # Get the image embedding
        img_embedding = self.img_embedding(input_imgs, image_loc)

        # Get the results from the ViLBERT module
        (sequence_output_t, sequence_output_v, all_attention_mask,) = self.vilbert(
            txt_embedding,
            img_embedding,
            kg_embedding,
            extended_attention_mask,
            extended_image_attention_mask,
            extended_co_attention_mask,
            output_all_encoded_layers,
        )

        # Choose the layer used
        sequence_output_t = sequence_output_t[bert_layer_used]
        sequence_output_v = sequence_output_v[bert_layer_used]

        if use_pooled_output:
            sequence_output_t = self.bert_text_pooler(sequence_output_t)
            sequence_output_v = self.bert_image_pooler(sequence_output_v)

        # Get the results from the Transformer module
        sequence_output_t_bis, attention_mask_bis = self.q_kg_transformer(
            txt_embedding,
            kg_embedding,
            extended_attention_mask,
            output_all_encoded_layers,
        )

        # Choose the layer used
        sequence_output_t_bis = sequence_output_t_bis[bert_layer_used]

        # Normalize the graph weights, so that high weights don't override
        # the added weights
        conceptnet_graph.normalize_weights()
        # Refine the given ConceptNet graph with the help of `G_1` model
        list_questions = []
        input_questions = input_txt.tolist()

        for list_indexes in input_questions:
            list_words = []
            for index in list_indexes:
                try:
                    list_words.append(conceptnet_graph.get_word(index))
                except Exception as e:
                    print("ERROR: ", e)
            list_questions.append(list_words)

        self.graph_refinement(list_questions, attention_mask_bis, conceptnet_graph)

        # Send the question results from ViLBERT and Transformer to the
        # F1 fusion module
        fused_question_emb, fused_question_att = self.fusion_question(
            sequence_output_t,
            all_attention_mask[0],
            sequence_output_t_bis,
            attention_mask_bis,
        )

        # Reduce the size of the ConceptNet graph by pruning low-weighted edges
        # Keep only the k highest ones
        list_main_entities = conceptnet_graph.select_top_edges(k)
        knowledge_graph_emb = []
        for entity in list_main_entities:
            knowledge_graph_emb.append(
                self.txt_embedding.conceptnet_embedding.get_node_embedding_tensor(
                    str(entity)
                )
            )

        # Send the image, question and ConceptNet to the Aggregator module
        result_vector = self.aggregator(
            fused_question_emb,
            fused_question_att,
            sequence_output_v[-1],
            all_attention_mask[1],
            knowledge_graph_emb,
        )

        # TODO: Send the vector to the SimpleClassifier to get the answer
        return self.vil_prediction(result_vector)
