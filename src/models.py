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


class EEGModelConfig:
    """EEG模型配置类，与数据处理配置对应"""
    
    # 基本参数
    NUM_CHANNELS = 8  # EEG通道数
    SAMPLING_FREQ = 250.0  # 采样频率
    NUM_CLASSES = 5  # 分类类别数（左、右、上、下、静息）
    
    # 默认窗口长度（与EEGConfig默认值保持一致）
    DEFAULT_WINDOW_DURATION = 2.0  # 秒
    DEFAULT_SEQ_LENGTH = int(DEFAULT_WINDOW_DURATION * SAMPLING_FREQ)  # 样本点数
    
    # 模型超参数默认值
    D_MODEL = 128  # 模型维度
    NHEAD = 8  # 注意力头数
    NUM_LAYERS = 3  # Transformer层数
    DIM_FEEDFORWARD = 256  # 前馈网络维度
    DROPOUT = 0.1  # Dropout率
    
    # EEGNet特定参数
    KERNEL_LENGTH = 64  # 时域卷积核长度
    F1 = 8  # 时域卷积滤波器数量
    D = 2  # 深度乘子
    F2 = 16  # 分离卷积滤波器数量
    
    # CNNLSTM特定参数
    HIDDEN_SIZE = 128  # LSTM隐藏层大小
    LSTM_LAYERS = 2  # LSTM层数


class PositionalEncoding(nn.Module):
    """
    位置编码模块，用于Transformer模型
    
    参数:
        num_hiddens: 隐藏层维度
        dropout: dropout率
        max_len: 最大序列长度
    """
    def __init__(self, num_hiddens, dropout, max_len=1000):
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
        activation: 激活函数，默认为ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.1, activation=nn.ReLU):
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
        self.act = activation(inplace=True) if activation == nn.ReLU else activation()
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
        x = self.act(x)
        x = self.dropout(x)
        
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
        input_window_samples: 输入窗口样本数
    """
    def __init__(self, num_channels=EEGModelConfig.NUM_CHANNELS, 
                 num_classes=EEGModelConfig.NUM_CLASSES, 
                 dropout_rate=0.5, 
                 kernel_length=EEGModelConfig.KERNEL_LENGTH, 
                 F1=EEGModelConfig.F1, 
                 D=EEGModelConfig.D, 
                 F2=EEGModelConfig.F2,
                 input_window_samples=EEGModelConfig.DEFAULT_SEQ_LENGTH):
        super().__init__()
        
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.input_window_samples = input_window_samples
        
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
        
        # 自适应平均池化以处理不同长度的输入
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
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
        
        # 自适应池化，处理可变长度输入
        x = self.adaptive_pool(x)
        
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
    def __init__(self, num_channels=EEGModelConfig.NUM_CHANNELS, 
                 num_classes=EEGModelConfig.NUM_CLASSES, 
                 d_model=EEGModelConfig.D_MODEL, 
                 nhead=EEGModelConfig.NHEAD, 
                 num_layers=EEGModelConfig.NUM_LAYERS, 
                 dim_feedforward=EEGModelConfig.DIM_FEEDFORWARD, 
                 dropout=EEGModelConfig.DROPOUT, 
                 seq_length=EEGModelConfig.DEFAULT_SEQ_LENGTH):
        super().__init__()
        
        # 保存配置参数
        self.num_channels = num_channels
        self.d_model = d_model
        self.seq_length = seq_length
        
        # 卷积特征提取 - 增强空间滤波能力
        self.spatial_filter = nn.Sequential(
            nn.Conv1d(num_channels, num_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_channels * 2),
            nn.ELU(),
            nn.Conv1d(num_channels * 2, num_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_channels),
            nn.ELU()
        )
        
        # 时域特征提取
        self.feature_extractor = nn.Sequential(
            CNN1DBlock(num_channels, d_model // 2, kernel_size=15, padding=7, dropout=dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 降采样，减少序列长度
            CNN1DBlock(d_model // 2, d_model, kernel_size=11, padding=5, dropout=dropout)
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 线性投影层
        self.input_proj = nn.Linear(d_model, d_model)
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )
        
        # 注意力池化
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        """
        模型的前向传播
        
        参数:
            x: 输入张量，形状为[batch_size, num_channels, seq_length]
            
        返回:
            类别预测，形状为[batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # 空间滤波增强
        x = self.spatial_filter(x)  # [batch_size, num_channels, seq_length]
        
        # 时域特征提取
        x = self.feature_extractor(x)  # [batch_size, d_model, seq_length/2]
        
        # 变换维度顺序以适应Transformer
        x = x.permute(0, 2, 1)  # [batch_size, seq_length/2, d_model]
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # 线性投影
        x = self.input_proj(x)
        
        # Transformer层
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 注意力池化
        attn_weights = self.attention_pool(x)  # [batch_size, seq_length/2, 1]
        x = torch.sum(x * attn_weights, dim=1)  # [batch_size, d_model]
        
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
        seq_length: 输入序列长度
    """
    def __init__(self, num_channels=EEGModelConfig.NUM_CHANNELS, 
                 num_classes=EEGModelConfig.NUM_CLASSES, 
                 hidden_size=EEGModelConfig.HIDDEN_SIZE, 
                 num_layers=EEGModelConfig.LSTM_LAYERS, 
                 dropout=0.5,
                 seq_length=EEGModelConfig.DEFAULT_SEQ_LENGTH):
        super().__init__()
        
        # 保存配置
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 空间滤波层
        self.spatial_filter = nn.Sequential(
            nn.Conv1d(num_channels, num_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_channels * 2),
            nn.ELU(),
            nn.Conv1d(num_channels * 2, num_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_channels),
            nn.ELU()
        )
        
        # CNN特征提取器
        self.feature_extractor = nn.Sequential(
            CNN1DBlock(num_channels, 32, kernel_size=11, padding=5, dropout=dropout/2),
            nn.MaxPool1d(2),
            CNN1DBlock(32, 64, kernel_size=9, padding=4, dropout=dropout/2),
            nn.MaxPool1d(2),
            CNN1DBlock(64, 128, kernel_size=7, padding=3, dropout=dropout/2),
            nn.MaxPool1d(2)
        )
        
        # 计算降采样后的序列长度
        self.downsampled_length = seq_length // 8  # 经过3次池化，每次池化系数为2
        
        # LSTM时序处理
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
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
        batch_size = x.size(0)
        
        # 空间滤波增强
        x = self.spatial_filter(x)  # [batch_size, num_channels, seq_length]
        
        # CNN特征提取
        x = self.feature_extractor(x)  # [batch_size, 128, seq_length//8]
        
        # 变换维度顺序以适应LSTM
        x = x.permute(0, 2, 1)  # [batch_size, seq_length//8, 128]
        
        # LSTM处理
        all_outputs, _ = self.lstm(x)  # [batch_size, seq_length//8, hidden_size*2]
        
        # 注意力机制
        attn_weights = self.attention(all_outputs)  # [batch_size, seq_length//8, 1]
        context = torch.sum(all_outputs * attn_weights, dim=1)  # [batch_size, hidden_size*2]
        
        # 分类
        x = self.classifier(context)  # [batch_size, num_classes]
        
        return x


class DeepConvNet(nn.Module):
    """
    DeepConvNet模型，使用更深层次的卷积网络进行EEG分类
    
    参数:
        num_channels: EEG通道数
        num_classes: 分类类别数
        dropout: dropout率
    """
    def __init__(self, num_channels=EEGModelConfig.NUM_CHANNELS, 
                 num_classes=EEGModelConfig.NUM_CLASSES, 
                 dropout=0.5):
        super().__init__()
        
        # 空间卷积层
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(num_channels, 1), bias=False),
            nn.BatchNorm2d(25)
        )
        
        # 时域卷积层 - 第一块
        self.block1 = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=(1, 5), stride=1, padding=(0, 2), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropout)
        )
        
        # 时域卷积层 - 第二块
        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), stride=1, padding=(0, 2), bias=False),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropout)
        )
        
        # 时域卷积层 - 第三块
        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), stride=1, padding=(0, 2), bias=False),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropout)
        )
        
        # 时域卷积层 - 第四块
        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), stride=1, padding=(0, 2), bias=False),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropout)
        )
        
        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, num_classes)
        )
        
    def forward(self, x):
        """
        模型的前向传播
        
        参数:
            x: 输入张量，形状为[batch_size, num_channels, seq_length]
            
        返回:
            类别预测，形状为[batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # 增加通道维度
        x = x.unsqueeze(1)  # [batch_size, 1, num_channels, seq_length]
        
        # 空间卷积
        x = self.spatial_conv(x)  # [batch_size, 25, 1, seq_length]
        
        # 卷积块
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # 自适应池化
        x = self.adaptive_pool(x)
        
        # 展平
        x = x.view(batch_size, -1)
        
        # 分类
        x = self.classifier(x)
        
        return x


class EfficientEEGNet(nn.Module):
    """
    EfficientEEGNet模型，一种更轻量高效的EEG分类模型
    
    参数:
        num_channels: EEG通道数
        num_classes: 分类类别数
        dropout: dropout率
    """
    def __init__(self, num_channels=EEGModelConfig.NUM_CHANNELS, 
                 num_classes=EEGModelConfig.NUM_CLASSES, 
                 dropout=0.2):
        super().__init__()
        
        # 第一块：复合时空滤波器
        self.block1 = nn.Sequential(
            # 时域滤波
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=1, padding=(0, 25), bias=False),
            nn.BatchNorm2d(16),
            # 空间滤波
            nn.Conv2d(16, 32, kernel_size=(num_channels, 1), stride=1, groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(dropout)
        )
        
        # 第二块：深度可分离卷积
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=1, padding=(0, 7), groups=32, bias=False),
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=1, bias=False),  # 点卷积
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=(1, 8)),
            nn.Dropout(dropout)
        )
        
        # 第三块：注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(8, 32, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        """
        模型的前向传播
        
        参数:
            x: 输入张量，形状为[batch_size, num_channels, seq_length]
            
        返回:
            类别预测，形状为[batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # 增加通道维度
        x = x.unsqueeze(1)  # [batch_size, 1, num_channels, seq_length]
        
        # 第一块：复合时空滤波器
        x = self.block1(x)
        
        # 第二块：深度可分离卷积
        x = self.block2(x)
        
        # 第三块：注意力机制
        attn = self.attention(x)
        x = x * attn
        
        # 自适应池化
        x = self.adaptive_pool(x)
        
        # 展平
        x = x.view(batch_size, -1)
        
        # 分类
        x = self.classifier(x)
        
        return x


def create_model(model_name, num_channels=EEGModelConfig.NUM_CHANNELS, 
                 num_classes=EEGModelConfig.NUM_CLASSES, 
                 seq_length=EEGModelConfig.DEFAULT_SEQ_LENGTH, **kwargs):
    """
    创建指定名称的模型
    
    参数:
        model_name: 模型名称，可选值为['eegnet', 'deepconv', 'transformer', 'cnnlstm', 'efficient']
        num_channels: EEG通道数
        num_classes: 分类类别数
        seq_length: 序列长度
        **kwargs: 其他模型特定参数
        
    返回:
        创建的模型实例
    """
    models = {
        'eegnet': EEGNet(num_channels, num_classes, input_window_samples=seq_length, **kwargs),
        'transformer': EEGTransformerModel(num_channels, num_classes, seq_length=seq_length, **kwargs),
        'cnnlstm': CNNLSTM(num_channels, num_classes, seq_length=seq_length, **kwargs),
        'deepconv': DeepConvNet(num_channels, num_classes, **kwargs),
        'efficient': EfficientEEGNet(num_channels, num_classes, **kwargs)
    }
    
    if model_name not in models:
        raise ValueError(f"模型不可用 用这些 {list(models.keys())}")
    
    return models[model_name]