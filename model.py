from transformers import GPT2PreTrainedModel, AutoModelForCausalLM, GPT2DoubleHeadsModel
from transformers.modeling_utils import SequenceSummary
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput

class CustomGPT2Model(GPT2DoubleHeadsModel):
    def __init__(self, config):
        super().__init__(config)
        #self.multiple_choice_head = SequenceSummary(config)

        # 원본 모델의 파라미터를 동결 (freeze)
        for param in self.parameters():
            param.requires_grad = False

        # 새로운 레이어의 파라미터는 훈련 가능하게 설정
        for param in self.multiple_choice_head.parameters():
            param.requires_grad = True

    def forward(self, input_ids, mc_token_ids, lm_labels=None, mc_labels=None, 
                token_type_ids=None, position_ids=None, head_mask=None, output_hidden_states= None):
        # 원본 모델 호출
        transformer_outputs = self.transformer(
            input_ids, token_type_ids=token_type_ids,
            position_ids=position_ids, head_mask=head_mask, output_hidden_states=output_hidden_states,
        )

        # 원본 모델의 lm_loss, last hidden state, lm_logits 추출
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        lm_loss = None
        if lm_labels is not None:
            lm_labels = lm_labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            print("lm_loss: " , lm_loss)
            
        mc_loss = None
        if mc_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            mc_loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
            print("mc_loss: ", mc_loss)

        output = (lm_logits, mc_logits) + transformer_outputs[1:]
        if mc_loss is not None:
            output = (mc_loss,) + output
        return ((lm_loss,) + output) if lm_loss is not None else output

        # 수정된 출력 반환: lm_loss, mc_loss 및 추가적인 출력 포함
        #return GPT2DoubleHeadsModelOutput(
        #    loss=lm_loss,
        #    mc_loss=mc_loss,
        #    logits=lm_logits,
        #    mc_logits=mc_logits,
        #    past_key_values=t_outputs.past_key_values,
        #    hidden_states=t_outputs.hidden_states,
        #    attentions=t_outputs.attentions,
        #)