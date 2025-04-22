"""
EEG主程序 - 用于训练和评估EEG分类模型

此程序集成了数据处理、模型定义和训练模块，提供完整的EEG分类流程，包括：
- 数据加载和预处理
- 模型创建
- 模型训练
- 模型评估和可视化
"""

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
import random
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
import warnings
import json
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchinfo import summary as model_summary

# 导入自定义模块
from data import load_all_csv_files, EEGDataset, LABEL_NAMES
from models import create_model
from training import ModelWrapper, train_model

# 忽略警告
warnings.filterwarnings('ignore')

#-----------------------------
# 配置参数（直接在此处修改）
#-----------------------------

# 数据参数
# 在main.py中更新SEQ_LENGTH参数
#-----------------------------
# 配置参数（直接在此处修改）
#-----------------------------

# 数据参数
DATA_DIR = 'eeg/OpenViBE/data'  # 数据文件夹路径
DATA_PATTERN = '*.csv'  # 数据文件匹配模式
MIN_REST_EPOCHS = None  # 最小静息epochs数量，设为None则自动判断

# 模型参数
MODEL = 'transformer'  # 可选: 'simple', 'deepconv', 'eegnet', 'transformer', 'cnnlstm'
NUM_CHANNELS = 8  # EEG通道数
NUM_CLASSES = 5  # 分类类别数
SEQ_LENGTH = 1500  # 序列长度 - 修改为1500适应6秒数据(250Hz * 6秒)

# 训练参数
BATCH_SIZE = 10  # 批量大小
LR = 5e-4  # 学习率
MAX_EPOCHS = 100  # 最大训练轮数
EARLY_STOPPING = True  # 是否使用早停
PATIENCE = 100  # 早停耐心值
USE_GPU = True  # 是否使用GPU
SEED = 42  # 随机种子，设为None则随机生成

# 保存参数
SAVE_DIR = './models'  # 模型保存目录
EXP_NAME = f"{MODEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # 实验名称

# 类别权重参数 (可选)
# 格式为字典，例如: {0: 1.0, 1: 1.0, 2: 1.5, 3: 1.0, 4: 0.8}
# 设为None则根据数据分布自动计算
CLASS_WEIGHTS = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.5}  # 增加"上"类别的权重

#-----------------------------
# 数据保存控制参数
#-----------------------------

# 模型保存控制
SAVE_MODEL = True  # 是否保存模型
SAVE_MODEL_FREQUENCY = 10  # 每隔多少个epoch保存一次模型
SAVE_TOP_K_MODELS = 1  # 保存验证指标最好的前K个模型，设为-1保存所有

# 日志保存控制
SAVE_LOGS = True  # 是否保存训练日志
LOG_TYPES = ['tensorboard', 'csv']  # 可选: 'tensorboard', 'csv'

# 评估结果保存控制
SAVE_TEST_PREDICTIONS = True  # 是否保存测试集预测结果
SAVE_CONFUSION_MATRIX = True  # 是否保存混淆矩阵图
SAVE_CLASSIFICATION_REPORT = True  # 是否保存分类报告

# 模型结构保存控制
SAVE_MODEL_SUMMARY = True  # 是否保存模型结构摘要

# 训练数据信息保存控制
SAVE_DATA_STATS = True  # 是否保存数据统计信息

# 可视化控制
SAVE_LOSS_PLOTS = True  # 是否保存损失和准确率曲线
SAVE_FEATURE_MAPS = False  # 是否保存特征图(仅CNN模型)

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

def save_data_statistics(X, y, save_path):
    """保存数据统计信息"""
    os.makedirs(save_path, exist_ok=True)
    
    # 基本统计信息
    stats = {
        "样本总数": len(y),
        "特征形状": X.shape,
        "类别分布": {LABEL_NAMES[i]: int(np.sum(y == i)) for i in range(NUM_CLASSES)},
        "类别比例": {LABEL_NAMES[i]: float(np.sum(y == i)/len(y)*100) for i in range(NUM_CLASSES)},
        "特征最小值": float(np.min(X)),
        "特征最大值": float(np.max(X)),
        "特征均值": float(np.mean(X)),
        "特征标准差": float(np.std(X))
    }
    
    # 保存为JSON
    with open(os.path.join(save_path, 'data_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    
    # 绘制类别分布
    plt.figure(figsize=(10, 6))
    class_counts = [np.sum(y == i) for i in range(NUM_CLASSES)]
    sns.barplot(x=LABEL_NAMES, y=class_counts)
    plt.title('类别分布')
    plt.ylabel('样本数量')
    plt.savefig(os.path.join(save_path, 'class_distribution.png'))
    plt.close()
    
    # 绘制特征分布
    plt.figure(figsize=(12, 8))
    for ch in range(min(NUM_CHANNELS, 8)):  # 最多显示8个通道
        plt.subplot(4, 2, ch+1)
        plt.hist(X[:, ch, :].flatten(), bins=50, alpha=0.7)
        plt.title(f'通道 {ch+1} 特征分布')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'feature_distribution.png'))
    plt.close()
    
    print(f"数据统计信息已保存到: {save_path}")

def save_test_predictions(model, test_loader, save_path):
    """保存测试集的预测结果"""
    os.makedirs(save_path, exist_ok=True)
    
    device = next(model.parameters()).device
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            output = model(x)
            probs = torch.softmax(output, dim=1)
            _, preds = torch.max(output, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 保存为CSV
    results_df = pd.DataFrame({
        'true_label': all_labels,
        'predicted_label': all_preds,
        'true_class': [LABEL_NAMES[i] for i in all_labels],
        'predicted_class': [LABEL_NAMES[i] for i in all_preds],
    })
    
    # 添加每个类别的概率
    for i in range(NUM_CLASSES):
        results_df[f'prob_{LABEL_NAMES[i]}'] = [probs[i] for probs in all_probs]
    
    results_df.to_csv(os.path.join(save_path, 'test_predictions.csv'), index=False)
    
    # 保存混淆矩阵
    if SAVE_CONFUSION_MATRIX:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
        plt.close()
    
    # 保存分类报告
    if SAVE_CLASSIFICATION_REPORT:
        report = classification_report(all_labels, all_preds, 
                                      target_names=LABEL_NAMES, 
                                      output_dict=True)
        with open(os.path.join(save_path, 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=4)
        
        # 可视化分类报告
        report_df = pd.DataFrame(report).transpose()
        plt.figure(figsize=(12, 8))
        sns.heatmap(report_df.iloc[:-3, :3], annot=True, cmap='YlGnBu')
        plt.title('分类性能指标')
        plt.savefig(os.path.join(save_path, 'classification_metrics.png'))
        plt.close()
    
    print(f"测试集预测结果已保存到: {save_path}")

def save_model_architecture(model, input_shape, save_path):
    """保存模型架构摘要"""
    os.makedirs(save_path, exist_ok=True)
    
    # 创建示例输入
    dummy_input = torch.zeros(input_shape)
    
    # 使用torchinfo生成模型摘要
    model_info = model_summary(model, input_size=input_shape, verbose=0)
    
    # 保存模型摘要
    with open(os.path.join(save_path, 'model_summary.txt'), 'w') as f:
        f.write(str(model_info))
    
    # 尝试导出模型图结构
    try:
        from torchviz import make_dot
        y = model(dummy_input)
        dot = make_dot(y, params=dict(model.named_parameters()))
        dot.format = 'png'
        dot.render(os.path.join(save_path, 'model_graph'))
        print(f"模型结构图已保存到: {save_path}")
    except Exception as e:
        print(f"无法生成模型结构图: {e}")
    
    print(f"模型架构摘要已保存到: {save_path}")

def main():
    """主函数，执行EEG分类训练流程"""
    # 准备随机种子
    if SEED is None:
        seed = int(np.random.randint(2147483647))
    else:
        seed = SEED
    
    # 设置随机种子
    set_seed(seed)
    
    # 创建实验目录
    exp_dir = os.path.join(SAVE_DIR, EXP_NAME)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 加载数据
    print(f"正在从 {DATA_DIR} 加载数据...")
    X_all, y_all = load_all_csv_files(
        DATA_DIR, 
        pattern=DATA_PATTERN,
        label_weights=CLASS_WEIGHTS,
        min_rest_epochs=MIN_REST_EPOCHS
    )
    
    if X_all is None or y_all is None:
        print("数据加载失败，请检查数据路径和格式。")
        return
    
    # 保存数据统计信息
    if SAVE_DATA_STATS:
        save_data_statistics(X_all, y_all, os.path.join(exp_dir, 'data_stats'))
    
    # 创建数据集
    print("创建数据集...")
    eeg_dataset = EEGDataset(X_all, y_all)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = eeg_dataset.get_loaders(batch_size=BATCH_SIZE)
    
    # 检查是否使用GPU
    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("使用CPU")
    
    # 计算类别权重（用于处理数据不平衡）
    if CLASS_WEIGHTS is None:
        # 计算类别权重（inverse frequency weighting）
        class_counts = np.bincount(y_all)
        total_samples = len(y_all)
        class_weights = {i: total_samples / (len(class_counts) * count) 
                        for i, count in enumerate(class_counts)}
        print("使用自动计算的类别权重:", class_weights)
    else:
        class_weights = CLASS_WEIGHTS
        print("使用指定的类别权重:", class_weights)
    
    # 创建模型
    print(f"创建模型: {MODEL}...")
    model_arch = create_model(
        MODEL,
        num_channels=NUM_CHANNELS,
        num_classes=NUM_CLASSES,
        seq_length=SEQ_LENGTH
    )
    print(model_arch)
    
    # 保存模型架构摘要
    if SAVE_MODEL_SUMMARY:
        save_model_architecture(
            model_arch, 
            (BATCH_SIZE, NUM_CHANNELS, SEQ_LENGTH),
            os.path.join(exp_dir, 'model_architecture')
        )
    
    # 准备回调
    callbacks = []
    
    # 早停回调
    if EARLY_STOPPING:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            verbose=True,
            mode='min'
        )
        callbacks.append(early_stopping)
    
    # 模型检查点回调
    if SAVE_MODEL:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(exp_dir, 'checkpoints'),
            filename='model-{epoch:02d}-{val_acc:.4f}',
            monitor='val_acc',
            save_top_k=SAVE_TOP_K_MODELS,
            mode='max',
            every_n_epochs=SAVE_MODEL_FREQUENCY,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
    
    # 学习率监控回调
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # 准备日志记录器
    loggers = []
    if SAVE_LOGS:
        if 'tensorboard' in LOG_TYPES:
            tb_logger = TensorBoardLogger(
                save_dir=os.path.join(exp_dir, 'logs'),
                name='tensorboard'
            )
            loggers.append(tb_logger)
        
        if 'csv' in LOG_TYPES:
            csv_logger = CSVLogger(
                save_dir=os.path.join(exp_dir, 'logs'),
                name='csv'
            )
            loggers.append(csv_logger)
    
    # 创建模型包装器
    model = ModelWrapper(
        arch=model_arch,
        dataset=eeg_dataset,
        batch_size=BATCH_SIZE,
        lr=LR,
        max_epoch=MAX_EPOCHS,
        num_classes=NUM_CLASSES,
        class_weights=class_weights
    )
    
    # 创建训练器
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        logger=loggers if loggers else False,
        log_every_n_steps=10,
        accelerator='gpu' if USE_GPU and torch.cuda.is_available() else 'cpu',
        deterministic=True
    )
    
    # 训练模型
    print("开始训练模型...")
    trainer.fit(model)
    
    # 测试模型
    print("开始测试模型...")
    trainer.test(model)
    
    # 保存测试预测结果
    if SAVE_TEST_PREDICTIONS:
        save_test_predictions(
            model.arch, 
            test_loader,
            os.path.join(exp_dir, 'test_results')
        )
    
    # 保存损失和准确率曲线
    if SAVE_LOSS_PLOTS and hasattr(model, 'train_loss') and hasattr(model, 'val_loss'):
        plots_dir = os.path.join(exp_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(model.train_loss, label='训练损失')
        plt.plot(model.val_loss, label='验证损失')
        plt.title('损失曲线')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'loss_curve.png'))
        plt.close()
        
        # 准确率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(model.train_acc, label='训练准确率')
        plt.plot(model.val_acc, label='验证准确率')
        plt.title('准确率曲线')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'accuracy_curve.png'))
        plt.close()
        
        print(f"损失和准确率曲线已保存到: {plots_dir}")
    
    # 打印训练结果摘要
    print(f"\n====== 训练结果摘要 ======")
    print(f"模型: {MODEL}")
    print(f"最佳验证准确率: {model.best_val_acc:.4f}")
    print(f"训练轮数: {trainer.current_epoch+1}/{MAX_EPOCHS}")
    print(f"实验目录: {exp_dir}")
    
    # 将参数保存为JSON文件，方便复现
    params = {
        "数据参数": {
            "DATA_DIR": DATA_DIR,
            "DATA_PATTERN": DATA_PATTERN,
            "MIN_REST_EPOCHS": MIN_REST_EPOCHS
        },
        "模型参数": {
            "MODEL": MODEL,
            "NUM_CHANNELS": NUM_CHANNELS,
            "NUM_CLASSES": NUM_CLASSES,
            "SEQ_LENGTH": SEQ_LENGTH
        },
        "训练参数": {
            "BATCH_SIZE": BATCH_SIZE,
            "LR": LR,
            "MAX_EPOCHS": MAX_EPOCHS,
            "EARLY_STOPPING": EARLY_STOPPING,
            "PATIENCE": PATIENCE,
            "USE_GPU": USE_GPU,
            "SEED": seed
        },
        "类别权重": class_weights,
        "保存参数": {
            "SAVE_MODEL": SAVE_MODEL,
            "SAVE_LOGS": SAVE_LOGS,
            "LOG_TYPES": LOG_TYPES,
            "SAVE_TEST_PREDICTIONS": SAVE_TEST_PREDICTIONS,
            "SAVE_CONFUSION_MATRIX": SAVE_CONFUSION_MATRIX,
            "SAVE_CLASSIFICATION_REPORT": SAVE_CLASSIFICATION_REPORT,
            "SAVE_MODEL_SUMMARY": SAVE_MODEL_SUMMARY,
            "SAVE_DATA_STATS": SAVE_DATA_STATS,
            "SAVE_LOSS_PLOTS": SAVE_LOSS_PLOTS
        }
    }
    
    with open(os.path.join(exp_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4, ensure_ascii=False)
    
    print(f"参数已保存到: {os.path.join(exp_dir, 'params.json')}")

if __name__ == "__main__":
    main()