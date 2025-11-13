from transformers.models.gpt2.modeling_gpt2 import GPT2Block

class GPT2BlockWithAdapter(GPT2Block):
    def __init__(self, config):
        super().__init__(config)
        # 假設Adapter的大小爲64
        adapter_size = 64
        self.adapter = AdapterLayer(config.n_embd, adapter_size)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # 調用原始的前向傳播方法
        attn_outputs = super().forward(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 得到Transformer層的輸出
        a = attn_outputs[0]  # 輸出的第一部分是attention的結果
        # 將輸出通過Adapter層
        a = self.adapter(a)
        # 返回修改後的輸出（其他輸出保持不變）
        outputs = (a,) + attn_outputs[1:]
        return outputs
"""
每個GPT2Block包含了一系列的自注意力（Self-Attention）和前饋網絡（Feed-Forward）層，這些層共同構成了模型的基礎架構。

"""


