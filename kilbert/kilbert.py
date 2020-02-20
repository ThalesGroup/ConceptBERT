### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

# Custom libraries
from vilbert.vilbert import BertPreTrainedModel, BertConfig

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
from fusion_modules.cti_model.cti import CTIModel

from classifier.classifier import SimpleClassifier

### VARIABLES ###
# Maximum number of nodes extracted from the knowledge graph (heaviest edges)
k = 20
# Whether to use the first token or all of them
use_pooled_output = False

### CLASS DEFINITION ###
class Kilbert(nn.Module):
    """

    """

    def __init__(
        self, config, num_labels, split="", dropout_prob=0.1, default_gpu=True,
    ):
        super(Kilbert, self).__init__()
        # Variables
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_prob)

        # Load the embedding modules
        self.txt_embedding = BertEmbeddings(config, split)
        self.img_embedding = BertImageEmbeddings(config)

        # Main modules
        config = BertConfig("config/bert_base_6layer_6conect.json")
        # TODO: Replace the pretrained model with VQA by pretrained model with OK-VQA
        self.vilbert = VILBertForVLTasks.from_pretrained(
            "/nas-data/vilbert/data2/VQA_bert_base_6layer_6conect-pretrained/pytorch_model_19.bin",
            config,
            num_labels,
            split,
            default_gpu=default_gpu,
        )

        # Embedding for the ViLBert model
        self.vilbert_txt_embedding = self.vilbert.bert.embeddings
        self.vilbert_img_embedding = self.vilbert.bert.v_embeddings

        self.q_kg_transformer = QuestionGraphTransformer(
            config, split, dropout_prob, default_gpu
        )

        # Self-attention for question (used for importance index)
        self.q_att = QuestionSelfAttention(16, 768, 0.2)
        self.graph_refinement = GraphRefinement(
            self.txt_embedding.conceptnet_embedding, k
        )

        # Fusion modules
        self.fusion_question = SimpleQuestionAddition(config)
        self.bert_text_pooler = BertTextPooler(config)
        self.bert_image_pooler = BertImagePooler(config)

        # self.aggregator = SimpleConcatenation(config)
        self.aggregator = CTIModel(
            v_dim=1024,
            q_dim=768,
            kg_dim=200,
            glimpse=2,
            h_dim=512,
            h_out=1,
            rank=32,
            k=1,
        )

        # Prediction modules
        # classifier_in_dim = self.aggregator.output_dim
        # classifier_hid_dim = self.aggregator.hidden_dim
        # self.vil_prediction = SimpleClassifier(
        #     #             classifier_in_dim, classifier_hid_dim, num_labels, 0.5
        #     1024 + 1024 + 200 * k,
        #     2048,
        #     num_labels,
        #     0.5,
        # )
        self.vil_prediction = SimpleClassifier(1024, 1024 * 2, num_labels, 0.5)

    def convert_tokens(self, input_txt, q_self_attention, conceptnet_graph):
        """

        """
        tokens_conceptnet = []
        q_attention = []

        for i, question in enumerate(input_txt):
            # Convert the list of BERT tokens to a list of words (but still tokens)
            list_bert_tokens = []
            for token in question:
                if int(token.item()) == 0:
                    break
                else:
                    try:
                        list_bert_tokens.append(
                            self.txt_embedding.conceptnet_embedding.token_dictionary[
                                int(token.item())
                            ]
                        )
                    except:
                        pass
            # Check which tokens need to be fused and create list for the assembled words
            list_words = []
            indexes_to_fuse = []

            token_cache = []
            word_cache = ""

            for j, token in enumerate(list_bert_tokens):
                if token[:2] == "##":
                    token_cache.append(j)
                    word_cache += token[2:]
                else:
                    if word_cache != "":
                        list_words.append(word_cache)
                        indexes_to_fuse.append(token_cache)

                    word_cache = str(token)
                    token_cache = [j]

            list_words.append(word_cache)
            indexes_to_fuse.append(token_cache)

            # Create a list for the new question self-attention
            new_q_self_attention = []
            for list_indexes in indexes_to_fuse:
                attention_batch = 0
                for index in list_indexes:
                    attention_batch += q_self_attention[i][index]
                new_q_self_attention.append(attention_batch)

            new_q_self_attention = torch.stack(new_q_self_attention)

            # Convert the assembled words to their ConceptNet indexes
            new_input_txt = []
            for word in list_words:
                try:
                    new_input_txt.append(conceptnet_graph.index_nodes_dict[word])
                except Exception as e:
                    if word not in ["[CLS]", "[SEP]", "'", "?"]:
                        print("ERROR in `convert_tokens`: ", e)

            new_input_txt = torch.stack(new_input_txt)

            tokens_conceptnet.append(new_input_txt)
            q_attention.append(new_q_self_attention)

        # TODO: Send back the tensors in the correct device
        device = input_txt.get_device()

        tokens_conceptnet = torch.stack(tokens_conceptnet)
        tokens_conceptnet = (
            tokens_conceptnet.cuda(device) if device != -1 else tokens_conceptnet
        )

        q_attention = torch.stack(q_attention)
        q_attention = q_attention.cuda(device) if device != -1 else q_attention

        return tokens_conceptnet, q_attention

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
        # img_embedding = self.img_embedding(input_imgs, image_loc)

        # Get the embeddings for ViLBert
        vilbert_txt_embedding = self.vilbert_txt_embedding(input_txt, token_type_ids)
        vilbert_img_embedding = self.vilbert_img_embedding(input_imgs, image_loc)

        # Get the results from the ViLBERT module
        (
            sequence_output_t,
            sequence_output_v,
            pooled_output_t,
            pooled_output_v,
            all_attention_mask,
        ) = self.vilbert(
            vilbert_txt_embedding,
            vilbert_img_embedding,
            kg_embedding,
            extended_attention_mask,
            extended_image_attention_mask,
            extended_co_attention_mask,
            output_all_encoded_layers,
        )

        if use_pooled_output:
            sequence_output_t = pooled_output_t
            sequence_output_v = pooled_output_v
            # sequence_output_t = self.bert_text_pooler(sequence_output_t)
            # sequence_output_v = self.bert_image_pooler(sequence_output_v)

        # Get the results from the Transformer module
        (
            sequence_output_t_bis,
            pooled_output_bis,
            attention_mask_bis,
        ) = self.q_kg_transformer(
            txt_embedding,
            kg_embedding,
            extended_attention_mask,
            output_all_encoded_layers,
        )

        # Compute the question self-attention

        question_self_attention = self.q_att(sequence_output_t_bis)
        # Transfer question self-attention to correct device
        question_self_attention = (
            question_self_attention.cuda(sequence_output_t_bis.get_device())
            if sequence_output_t_bis.is_cuda
            else question_self_attention
        )

        """
        # Choose the layer used
        sequence_output_t_bis = sequence_output_t_bis[bert_layer_used]
        """
        if use_pooled_output:
            sequence_output_t_bis = pooled_output_bis

        # Convert `input_txt` (tensor of BERT tokens) to a tensor of ConceptNet tokens
        # Adapt `question_self_attention` accordingly
        tokens_conceptnet, q_self_attention = self.convert_tokens(
            input_txt, question_self_attention, conceptnet_graph
        )

        # Refine the given ConceptNet graph with the help of `G_1` model
        """
        list_questions = []

        input_questions = input_txt.tolist()

        for list_indexes in input_questions:
            print("LIST INDEXES: ", list_indexes)
            list_words = []
            for index in list_indexes:
                try:
                    list_words.append(conceptnet_graph.get_word(index))
                except Exception as e:
                    print("ERROR: ", e)
            list_questions.append(list_words)
        """

        knowledge_graph_emb = self.graph_refinement(
            #     input_txt, question_self_attention, conceptnet_graph, k
            tokens_conceptnet,
            q_self_attention,
            conceptnet_graph,
            k,
        )

        # Send the question results from ViLBERT and Transformer to the
        # F1 fusion module
        """
        fused_question_emb, fused_question_att = self.fusion_question(
            sequence_output_t,
            all_attention_mask[0],
            sequence_output_t_bis,
            attention_mask_bis,
        )
        """
        fused_question_emb = self.fusion_question(
            sequence_output_t, sequence_output_t_bis
        )

        # Reduce the size of the ConceptNet graph by pruning low-weighted edges
        # Keep only the k highest ones
        """
        list_main_entities = conceptnet_graph.select_top_edges(k)
        kg_emb = []
        for entity in list_main_entities:
            kg_emb.append(
                self.txt_embedding.conceptnet_embedding.get_node_embedding_tensor(
                    str(entity)
                )
            )
        knowledge_graph_emb = torch.stack(kg_emb)
        """
        # TODO: Remove this temporary fix
        ### BEGIN TEMPORARY FIX ###
        # Flatten knowledge_graph_emb to fit in the SimpleClassifier
        # knowledge_graph_emb = torch.flatten(knowledge_graph_emb, start_dim=1, end_dim=2)
        ### END TEMPORARY FIX ###

        # Send the image, question and ConceptNet to the Aggregator module
        result_vector, result_attention = self.aggregator(
            sequence_output_v,
            # all_attention_mask[1],
            fused_question_emb,
            # fused_question_att,
            knowledge_graph_emb,
        )

        # TODO: Send the vector to the SimpleClassifier to get the answer
        return self.vil_prediction(result_vector)


class FCNet(nn.Module):
    """
        Simple class for non-linear fully-connected network
    """

    def __init__(self, dims, act="ReLU", dropout=0, bias=True):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim, bias=bias), dim=None))
            if "" != act and act is not None:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1], bias=bias), dim=None))

        if "" != act and act is not None:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class QuestionSelfAttention(nn.Module):
    """
        Self-attention on the question
    """

    def __init__(self, size_question, num_hid, dropout):
        super(QuestionSelfAttention, self).__init__()
        self.num_hid = num_hid
        self.drop = nn.Dropout(dropout)
        self.W1_self_att_q = FCNet(dims=[num_hid, num_hid], dropout=dropout, act=None)
        self.W2_self_att_q = FCNet(dims=[num_hid, 1], act=None)

    def forward(self, question_features):
        """
            Returns a list of attention values for each word in the question
            Shape: [batch, size_question, num_hid]
        """
        batch_size = question_features.shape[0]
        q_len = question_features.shape[1]

        # (batch * size_question, num_hid)
        question_features_reshape = question_features.contiguous().view(
            -1, self.num_hid
        )
        # (batch, size_question)
        atten_1 = self.W1_self_att_q(question_features_reshape)
        atten_1 = torch.tanh(atten_1)
        atten = self.W2_self_att_q(atten_1).view(batch_size, q_len)
        # (batch, size_question)
        return F.softmax(atten.t(), dim=1).view(-1, q_len)
