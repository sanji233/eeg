"""
EEG模型模块 - 用于定义各种EEG分类模型架构

此模块提供了多种模型架构，用于EEG信号分类，包括：
- 基于CNN的模型
- 基于RNN的模型
- 基于Transformer的模型
- 混合架构模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np



class PositionalEncoding(nn.Module):
    """
    位置编码模块，用于Transformer模型
    
    参数:
        num_hiddens: 隐藏层维度
        dropout: dropout率
        max_len: 最大序列长度
    """
    def __init__(self, num_hiddens, dropout, max_len=1600):  # 修改为1600以支持6秒数据
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建足够长的位置编码
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        """
        添加位置编码到输入张量
        
        参数:
            X: 输入张量，形状为[batch_size, seq_len, embedding_dim]
            
        返回:
            添加位置编码后的张量
        """
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class TransformerBlock(nn.Module):
    """
    Transformer块，包含自注意力和前馈网络
    
    参数:
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        dim_feedforward: 前馈网络隐藏层维度
        dropout: dropout率
    """
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # 多头自注意力
        self.attention = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim)
        )
        
        # Layer Normalization层
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Transformer块的前向传播
        
        参数:
            x: 输入张量，形状为[batch_size, seq_len, embed_dim]
            
        返回:
            处理后的张量
        """
        # 自注意力层
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class CNN1DBlock(nn.Module):
    """
    1D卷积块，包含卷积、批归一化、激活和dropout
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        dropout: dropout率
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.1):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        卷积块的前向传播
        
        参数:
            x: 输入张量，形状为[batch_size, in_channels, seq_len]
            
        返回:
            处理后的张量，形状为[batch_size, out_channels, seq_len]
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return x

class SimpleEEGModel(nn.Module):
    """
    简单的EEG分类模型，使用1D卷积和全连接层
    
    参数:
        num_channels: EEG通道数
        num_classes: 分类类别数
        seq_length: 序列长度
    """
    def __init__(self, num_channels=8, num_classes=5, seq_length=1500):  # 修改默认值为1500
        super().__init__()
        
        self.conv1 = nn.Conv1d(num_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        
        # 计算卷积和池化后的特征图大小
        feat_size = (seq_length // 2) // 2  # 两次池化，每次大小减半
        self.fc1 = nn.Linear(32 * feat_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        """
        模型的前向传播
        
        参数:
            x: 输入张量，形状为[batch_size, num_channels, seq_length]
            
        返回:
            类别预测，形状为[batch_size, num_classes]
        """
        # 1D卷积层
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # 展平
        x = x.flatten(1)
        
        # 全连接层
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class DeepConvNet(nn.Module):
    """
    深层卷积网络，类似DeepConvNet架构，用于EEG分类
    
    参数:
        num_channels: EEG通道数
        num_classes: 分类类别数
    """
    def __init__(self, num_channels=8, num_classes=5):
        super().__init__()
        
        # 第一层：时域卷积 + 空间卷积
        self.temporal_conv = nn.Conv1d(num_channels, 25, kernel_size=10, stride=1, padding=0)
        self.spatial_conv = nn.Conv1d(25, 25, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(25)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3, padding=0)
        
        # 剩余卷积块
        self.block1 = nn.Sequential(
            nn.Conv1d(25, 50, kernel_size=10, stride=1, padding=0),
            nn.BatchNorm1d(50),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=0),
            nn.Dropout(0.5)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv1d(50, 100, kernel_size=10, stride=1, padding=0),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=0),
            nn.Dropout(0.5)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv1d(100, 200, kernel_size=10, stride=1, padding=0),
            nn.BatchNorm1d(200),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=0),
            nn.Dropout(0.5)
        )
        
        # 分类器
        self.classifier = nn.Linear(200, num_classes)
        
    def forward(self, x):
        """
        模型的前向传播
        
        参数:
            x: 输入张量，形状为[batch_size, num_channels, seq_length]
            
        返回:
            类别预测，形状为[batch_size, num_classes]
        """
        # 时域卷积
        x = self.temporal_conv(x)
        
        # 空间卷积
        x = self.spatial_conv(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = F.dropout(x, 0.5, self.training)
        
        # 后续卷积块
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # 全局平均池化
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        
        # 分类器
        x = self.classifier(x)
        
        return x

class EEGNet(nn.Module):
    """
    EEGNet模型，专为EEG信号处理设计的轻量级架构
    
    参数:
        num_channels: EEG通道数
        num_classes: 分类类别数
        dropout_rate: dropout率
        kernel_length: 时域卷积核长度
        F1: 时域卷积滤波器数量
        D: 深度乘子
        F2: 分离卷积滤波器数量
    """
    def __init__(self, num_channels=8, num_classes=5, dropout_rate=0.5, 
                 kernel_length=64, F1=8, D=2, F2=16):
        super().__init__()
        
        self.F1 = F1
        self.F2 = F2
        self.D = D
        
        # 块1：时域卷积
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1)
        )
        
        # 块2：空间卷积
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        # 块3：分离卷积
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 16 // 2), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        # 分类器
        self.classifier = nn.Linear(F2, num_classes)
        
    def forward(self, x):
        """
        模型的前向传播
        
        参数:
            x: 输入张量，形状为[batch_size, num_channels, seq_length]
            
        返回:
            类别预测，形状为[batch_size, num_classes]
        """
        # 调整输入形状
        x = x.unsqueeze(1)  # [batch, 1, channels, time]
        
        # 块1
        x = self.block1(x)
        
        # 块2
        x = self.block2(x)
        
        # 块3
        x = self.block3(x)
        
        # 分类器
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

class EEGTransformerModel(nn.Module):
    """
    基于Transformer的EEG分类模型
    
    参数:
        num_channels: EEG通道数
        num_classes: 分类类别数
        d_model: 模型维度
        nhead: 注意力头数
        num_layers: Transformer层数
        dim_feedforward: 前馈网络维度
        dropout: dropout率
        seq_length: 输入序列长度
    """
    def __init__(self, num_channels=8, num_classes=5, d_model=128, nhead=8, num_layers=3, 
                 dim_feedforward=256, dropout=0.1, seq_length=1500):  # 修改默认值为1500
        super().__init__()
        
        # 卷积特征提取
        self.feature_extractor = nn.Sequential(
            CNN1DBlock(num_channels, d_model // 2, kernel_size=11, padding=5, dropout=dropout),
            CNN1DBlock(d_model // 2, d_model, kernel_size=11, padding=5, dropout=dropout)
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )
        
    def forward(self, x):
        """
        模型的前向传播
        
        参数:
            x: 输入张量，形状为[batch_size, num_channels, seq_length]
            
        返回:
            类别预测，形状为[batch_size, num_classes]
        """
        # 特征提取
        x = self.feature_extractor(x)  # [batch_size, d_model, seq_length]
        
        # 变换维度顺序以适应Transformer
        x = x.permute(0, 2, 1)  # [batch_size, seq_length, d_model]
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer层
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 全局池化
        x = x.mean(dim=1)  # [batch_size, d_model]
        
        # 分类
        x = self.classifier(x)  # [batch_size, num_classes]
        
        return x

class CNNLSTM(nn.Module):
    """
    CNN-LSTM混合模型，使用CNN提取特征，然后用LSTM处理时间信息
    
    参数:
        num_channels: EEG通道数
        num_classes: 分类类别数
        hidden_size: LSTM隐藏层大小
        num_layers: LSTM层数
        dropout: dropout率
    """
    def __init__(self, num_channels=8, num_classes=5, hidden_size=128, num_layers=2, dropout=0.5):
        super().__init__()
        
        # CNN特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(num_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM时序处理
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2是因为双向LSTM
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        """
        模型的前向传播
        
        参数:
            x: 输入张量，形状为[batch_size, num_channels, seq_length]
            
        返回:
            类别预测，形状为[batch_size, num_classes]
        """
        # CNN特征提取
        x = self.feature_extractor(x)  # [batch_size, 64, seq_length//8]
        
        # 变换维度顺序以适应LSTM
        x = x.permute(0, 2, 1)  # [batch_size, seq_length//8, 64]
        
        # LSTM处理
        x, _ = self.lstm(x)  # [batch_size, seq_length//8, hidden_size*2]
        
        # 取最后一个时间步的输出
        x = x[:, -1, :]  # [batch_size, hidden_size*2]
        
        # 分类
        x = self.classifier(x)  # [batch_size, num_classes]
        
        return x

def create_model(model_name, num_channels=8, num_classes=5, seq_length=1500, **kwargs):  # 修改默认值为1500
    """
    创建指定名称的模型
    
    参数:
        model_name: 模型名称，可选值为['simple', 'deepconv', 'eegnet', 'transformer', 'cnnlstm']
        num_channels: EEG通道数
        num_classes: 分类类别数
        seq_length: 序列长度
        **kwargs: 其他模型特定参数
        
    返回:
        创建的模型实例
    """
    models = {
        'simple': SimpleEEGModel(num_channels, num_classes, seq_length),
        'deepconv': DeepConvNet(num_channels, num_classes),
        'eegnet': EEGNet(num_channels, num_classes),
        'transformer': EEGTransformerModel(num_channels, num_classes, seq_length=seq_length),
        'cnnlstm': CNNLSTM(num_channels, num_classes)
    }
    
    model_name = model_name.lower()
    if model_name not in models:
        raise ValueError(f"未知的模型名称: {model_name}，可用的模型有: {list(models.keys())}")
    
    return models[model_name]