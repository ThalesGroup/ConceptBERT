### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

# Custom libraries
from conceptBert.multimodal_pretraining.bert_classes import (
    BertPreTrainedModel,
    BertModel,
    BertPreTrainingHeads,
)

### CLASS DEFINITION ###
class BertForMultiModalPreTraining(BertPreTrainedModel):
    """BERT model with multi modal pre-training heads.
    """

    def __init__(self, config):
        super(BertForMultiModalPreTraining, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight
        )

        self.apply(self.init_bert_weights)
        self.predict_feature = config.predict_feature
        self.loss_fct = CrossEntropyLoss(ignore_index=-1)

        print("model's option for predict_feature is ", config.predict_feature)

        if self.predict_feature:
            self.vis_criterion = nn.MSELoss(reduction="none")
        else:
            self.vis_criterion = nn.KLDivLoss(reduction="none")

    def forward(
        self,
        input_ids,
        image_feat,
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        masked_lm_labels=None,
        image_label=None,
        image_target=None,
        next_sentence_label=None,
        output_all_attention_masks=False,
    ):

        # in this model, we first embed the images.
        (
            sequence_output_t,
            sequence_output_v,
            pooled_output_t,
            pooled_output_v,
            all_attention_mask,
        ) = self.bert(
            input_ids,
            image_feat,
            image_loc,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            output_all_encoded_layers=False,
            output_all_attention_masks=output_all_attention_masks,
        )

        prediction_scores_t, prediction_scores_v, seq_relationship_score = self.cls(
            sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v
        )

        if (
            masked_lm_labels is not None
            and next_sentence_label is not None
            and image_target is not None
        ):

            prediction_scores_v = prediction_scores_v[:, 1:]
            if self.predict_feature:
                img_loss = self.vis_criterion(prediction_scores_v, image_target)
                masked_img_loss = torch.sum(
                    img_loss * (image_label == 1).unsqueeze(2).float()
                ) / max(
                    torch.sum((image_label == 1).unsqueeze(2).expand_as(img_loss)), 1
                )

            else:
                img_loss = self.vis_criterion(
                    F.log_softmax(prediction_scores_v, dim=2), image_target
                )
                masked_img_loss = torch.sum(
                    img_loss * (image_label == 1).unsqueeze(2).float()
                ) / max(torch.sum((image_label == 1)), 0)

            # masked_img_loss = torch.sum(img_loss) / (img_loss.shape[0] * img_loss.shape[1])
            masked_lm_loss = self.loss_fct(
                prediction_scores_t.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
            )
            next_sentence_loss = self.loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
            # total_loss = masked_lm_loss + next_sentence_loss + masked_img_loss
            return (
                masked_lm_loss.unsqueeze(0),
                masked_img_loss.unsqueeze(0),
                next_sentence_loss.unsqueeze(0),
            )
        else:
            return (
                prediction_scores_t,
                prediction_scores_v,
                seq_relationship_score,
                all_attention_mask,
            )

