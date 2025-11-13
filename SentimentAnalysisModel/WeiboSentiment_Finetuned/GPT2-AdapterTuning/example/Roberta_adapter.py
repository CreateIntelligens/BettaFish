from transformers.models.roberta.modeling_roberta import RobertaLayer

class RobertaLayerWithAdapter(RobertaLayer):
    def __init__(self, config):
        super().__init__(config)
        # 假設Adapter的大小爲64
        adapter_size = 64
        self.adapter = AdapterLayer(config.hidden_size, adapter_size)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        # 調用原始的前向傳播方法
        self_outputs = super().forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
        # 得到Transformer層的輸出
        sequence_output = self_outputs[0]
        # 將輸出通過Adapter層
        sequence_output = self.adapter(sequence_output)
        # 返回修改後的輸出（其他輸出保持不變）
        return (sequence_output,) + self_outputs[1:]

"""
RoBERTa的每個RobertaLayer包含一個自注意力（self-attention）機制和一個前饋網絡，這些層共同構成了RoBERTa的基礎架構。
"""
