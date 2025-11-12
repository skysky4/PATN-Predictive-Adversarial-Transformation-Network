import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import os


class PATN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, input_len=30, output_len=10):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        self.output_len = output_len

    def forward(self, x):  # x: (B, input_len, input_dim)
        _, (h, c) = self.encoder(x)
        # 使用 decoder 自动回归生成预测序列
        outputs = []
        decoder_input = x[:, -1:, :]  # 初始输入：最后一个输入时间步
        for _ in range(self.output_len):
            out, (h, c) = self.decoder(decoder_input, (h, c))
            pred = self.fc(out)  # (B, 1, D)
            outputs.append(pred)
            decoder_input = pred
        return torch.cat(outputs, dim=1)  # (B, output_len, D)
    

class Gender_Inference_Model(nn.Module):
    def __init__(self):
        super(Gender_Inference_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 512, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn4 = nn.BatchNorm2d(512)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, H, W]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.global_pool(x).flatten(start_dim=1)  # [B, 512]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x  # 适合配合 CrossEntropyLoss


class HAR_model(nn.Module):
    def __init__(self,
                 input_size=6,
                 hidden_size=64,
                 num_layers=3,
                 num_classes=6,
                 bidirectional=False,
                 fc_hidden_dims=(64, 32),
                 dropout=0.3):
        super(HAR_model, self).__init__()

        self.input_bn = nn.BatchNorm1d(input_size)

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional)

        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        fc1_dim, fc2_dim = fc_hidden_dims
        self.fc1 = nn.Linear(lstm_output_size, fc1_dim)
        self.bn1 = nn.BatchNorm1d(fc1_dim)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.bn2 = nn.BatchNorm1d(fc2_dim)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(fc2_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, 6)
        batch_size, time_steps, feature_dim = x.shape

        x = x.reshape(-1, feature_dim)  # (B*T, F)
        x = self.input_bn(x)
        x = x.reshape(batch_size, time_steps, feature_dim)

        out, _ = self.lstm(x)  # (B, T, H)
        final_out = out[:, -1, :]  # (B, H)

        x = F.relu(self.bn1(self.fc1(final_out)))
        x = self.drop1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)

        logits = self.fc3(x)
        return logits