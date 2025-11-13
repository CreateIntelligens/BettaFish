import torch
import torch.nn as nn

class AdapterLayer(nn.Module):
    """
    Adapter層實現
    將其添加到Transformer層中可以實現參數高效微調
    """
    def __init__(self, input_size, adapter_size):
        super(AdapterLayer, self).__init__()
        # 降維全連接層
        self.down_project = nn.Linear(input_size, adapter_size)
        # 激活函數
        self.activation = nn.ReLU()
        # 升維全連接層
        self.up_project = nn.Linear(adapter_size, input_size)
        
        # 初始化參數
        self._init_weights()
    
    def _init_weights(self):
        # 初始化down_project用較小的值
        nn.init.normal_(self.down_project.weight, std=1e-2)
        nn.init.zeros_(self.down_project.bias)
        
        # 初始化up_project爲接近零的值，確保訓練初期對原始模型影響較小
        nn.init.normal_(self.up_project.weight, std=1e-2)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x):
        # 保存原始輸入用於殘差連接
        residual = x
        
        # 通過降維層
        x = self.down_project(x)
        # 激活
        x = self.activation(x)
        # 通過升維層
        x = self.up_project(x)
        
        # 殘差連接
        return residual + x 