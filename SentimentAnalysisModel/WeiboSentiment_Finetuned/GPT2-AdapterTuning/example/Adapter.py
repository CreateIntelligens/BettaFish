import torch
import torch.nn as nn

class AdapterLayer(nn.Module):
    def __init__(self, input_size, adapter_size):
        super(AdapterLayer, self).__init__()
        # 第一個全連接層降維
        self.down_project = nn.Linear(input_size, adapter_size)
        # ReLU激活函數
        self.relu = nn.ReLU()
        # 第二個全連接層升維
        self.up_project = nn.Linear(adapter_size, input_size)

    def forward(self, x):
        # 通過Adapter層的前向傳播
        down_projected = self.down_project(x)
        relu = self.relu(down_projected)
        up_projected = self.up_project(x)
        # 將Adapter的輸出與輸入相加（殘差連接）
        return x + up_projected
