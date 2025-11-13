import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from adapter import AdapterLayer

class GPT2BlockWithAdapter(nn.Module):
    """
    帶Adapter的GPT2Block層
    在原始GPT2Block的基礎上添加Adapter層實現參數高效微調
    """
    def __init__(self, config):
        super(GPT2BlockWithAdapter, self).__init__()
        # 創建標準的GPT2Block
        self.original_block = GPT2Block(config)
        
        # 添加Adapter層
        adapter_size = 64  # Adapter的隱藏層大小
        self.adapter = AdapterLayer(config.hidden_size, adapter_size)
    
    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs  # 使用**kwargs接收所有其他參數
    ):
        # 首先通過原始的GPT2Block，只傳遞它支持的參數
        outputs = self.original_block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        
        # 原始輸出中的第一個元素是隱藏狀態
        hidden_states = outputs[0]
        
        # 將隱藏狀態通過Adapter層
        hidden_states = self.adapter(hidden_states)
        
        # 更新輸出的隱藏狀態
        outputs = (hidden_states,) + outputs[1:]
        
        return outputs
    
    def load_state_dict(self, state_dict, strict=True):
        """
        自定義加載參數方法，用於從原始GPT2Block加載參數
        """
        # 將所有參數傳遞給原始Block
        return self.original_block.load_state_dict(state_dict, strict=strict) 