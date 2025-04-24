"""
EEG分类主程序 - 用于训练和评估EEG分类模型

此程序集成了数据处理、模型定义和训练模块，提供完整的EEG分类流程
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import warnings
from datetime import datetime

# 导入自定义模块
from data import EEGProcessor, EEGConfig
from models import create_model, EEGModelConfig
from training import ModelWrapper

# 忽略警告
warnings.filterwarnings('ignore')

#-----------------------------
# 配置参数（直接在此处修改）
#-----------------------------

# 数据参数
DATA_DIR = '/home/xiong/eeg/OpenViBE/data'                 # 数据文件夹路径
WINDOW_DURATION = 2             # 窗口持续时间(秒)
WINDOW_STEP = 0.5                 # 窗口步长(秒)
STATE_DURATION = 6             # 状态持续时间(秒)

# 模型参数
MODEL = 'eegnet'             # 可选: 'eegnet', 'transformer', 'cnnlstm', 'deepconv', 'efficient'

# 训练参数
BATCH_SIZE = 32                   # 批量大小
LEARNING_RATE = 0.001             # 学习率
MAX_EPOCHS = 100                  # 最大训练轮数
PATIENCE = 100                     # 早停耐心值
USE_GPU = True                    # 是否使用GPU
SEED = 42                         # 随机种子

# 类别权重参数 (可选)
# 设为None则根据数据分布自动计算
CLASS_WEIGHTS = None              # 例如: {0: 1.0, 1: 1.0, 2: 1.5, 3: 1.0, 4: 0.8}

# 保存参数
SAVE_DIR = 'models'               # 模型保存目录
EXP_NAME = None                   # 实验名称，为None则自动生成

#-----------------------------

def set_seed(seed):
    """设置随机种子以确保结果可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"设置随机种子: {seed}")

def main():
    """主函数，执行EEG分类训练流程"""
    # 设置随机种子
    set_seed(SEED)
    
    # 创建实验目录
    exp_name = EXP_NAME if EXP_NAME else f"{MODEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join(SAVE_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 创建配置
    eeg_config = EEGConfig()
    model_config = EEGModelConfig()
    
    # 创建处理器实例
    processor = EEGProcessor(config=eeg_config)
    
    # 加载并预处理数据
    print(f"正在处理 {DATA_DIR} 目录下的数据...")
    dataset = processor.prepare_dataset(
        folder_path=DATA_DIR,
        window_duration=WINDOW_DURATION,
        window_step=WINDOW_STEP,
        state_duration=STATE_DURATION
    )
    
    if dataset is None:
        print("数据加载失败，请检查数据路径和格式。")
        return
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=BATCH_SIZE)
    
    # 计算类别权重
    if CLASS_WEIGHTS is None:
        # 自动计算类别权重
        train_y = dataset.split("train").dataset['y']
        class_samples = np.bincount(train_y)
        n_samples = len(train_y)
        class_weights = {i: n_samples / (len(class_samples) * count) for i, count in enumerate(class_samples)}
        print("自动计算的类别权重:", class_weights)
    else:
        class_weights = CLASS_WEIGHTS
        print("使用指定的类别权重:", class_weights)
    
    # 选择设备
    use_gpu = USE_GPU and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    print(f"创建模型: {MODEL}")
    model_arch = create_model(
        MODEL,
        num_channels=eeg_config.NUM_CHANNELS,
        num_classes=len(eeg_config.LABEL_NAMES),
        seq_length=int(WINDOW_DURATION * eeg_config.SAMPLING_FREQ)
    )
    
    # 配置回调
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=True, mode='min'),
        ModelCheckpoint(
            dirpath=os.path.join(exp_dir, 'checkpoints'),
            filename='model-{epoch:02d}-{val_acc:.4f}',
            monitor='val_acc',
            save_top_k=1,
            mode='max',
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # 创建模型封装器
    model = ModelWrapper(
        arch=model_arch,
        dataset=dataset,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        max_epoch=MAX_EPOCHS,
        num_classes=len(eeg_config.LABEL_NAMES),
        class_weights=class_weights,
        label_names=eeg_config.LABEL_NAMES
    )
    
    # 创建训练器
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        accelerator='gpu' if use_gpu else 'cpu',
        deterministic=True
    )
    
    # 训练模型
    print("开始训练模型...")
    trainer.fit(model)
    
    # 测试模型
    print("开始测试模型...")
    test_results = trainer.test(model)
    
    # 打印训练结果摘要
    print(f"\n====== 训练结果摘要 ======")
    print(f"模型: {MODEL}")
    print(f"最佳验证准确率: {model.best_val_acc:.4f}")
    print(f"训练轮数: {trainer.current_epoch+1}/{MAX_EPOCHS}")
    print(f"实验目录: {exp_dir}")
    
    # 保存混淆矩阵图
    conf_matrix_path = os.path.join(exp_dir, "confusion_matrix.png")
    if os.path.exists(conf_matrix_path):
        print(f"混淆矩阵已保存到: {conf_matrix_path}")
    
    # 保存损失和准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(model.train_acc, 'r-', label='训练准确率')
    plt.plot(model.val_acc, 'b-', label='验证准确率')
    plt.title("准确率曲线")
    plt.xlabel("Epoch")
    plt.ylabel("准确率")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(exp_dir, "accuracy_curve.png"))
    
    plt.figure(figsize=(10, 6))
    plt.plot(model.train_loss, 'r-', label='训练损失')
    plt.plot(model.val_loss, 'b-', label='验证损失')
    plt.title("损失曲线")
    plt.xlabel("Epoch")
    plt.ylabel("损失")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(exp_dir, "loss_curve.png"))
    
    print(f"准确率和损失曲线已保存到: {exp_dir}")

if __name__ == "__main__":
    main()