### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn

# Custom libraries
from q_kg_transformer.conceptnet_embedding import ConceptNetEmbedding

### CLASS DEFINITION ###
class BertLayerNorm(nn.Module):
    """

    """

    def __init__(self, hidden_size, eps=1e-12):
        """
            Constructs a layernorm module in the TensorFlow style
            (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """
        Constructs the embeddings from word, position and token_type embeddings
    """

    def __init__(self, config, split):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and
        # be able to load any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Initiate the ConceptNet embeddings
        self.conceptnet_embedding = ConceptNetEmbedding(split)
        self.LayerNorm_kb = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout_kb = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        kg_embeddings = self.conceptnet_embedding.get_kg_embedding_tokens_from_bert(
            input_ids, words_embeddings.size(1), words_embeddings.size(2)
        )
        # Send tensor to correct device
        kg_embeddings = (
            kg_embeddings.cuda(words_embeddings.get_device())
            if words_embeddings.is_cuda
            else kg_embeddings
        )

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        kg_embeddings = self.LayerNorm_kb(kg_embeddings)
        kg_embeddings = self.dropout(embeddings)

        return embeddings, kg_embeddings


class BertImageEmbeddings(nn.Module):
    """
        Constructs the embeddings from image, spatial location (omit now) and 
        token_type embeddings.
    """

    def __init__(self, config):
        super(BertImageEmbeddings, self).__init__()

        self.image_embeddings = nn.Linear(config.v_feature_size, config.v_hidden_size)
        self.image_location_embeddings = nn.Linear(5, config.v_hidden_size)
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, input_loc):
        img_embeddings = self.image_embeddings(input_ids)
        loc_embeddings = self.image_location_embeddings(input_loc)
        embeddings = self.LayerNorm(img_embeddings + loc_embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
