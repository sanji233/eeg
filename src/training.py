"""
EEG训练模块 - 用于训练、验证和测试EEG分类模型

此模块提供了训练EEG分类模型所需的工具，包括：
- 移动平均计算器（用于记录损失和准确率）
- 模型封装器（PyTorch Lightning模块）
- 训练和评估功能
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torchmetrics.classification import Accuracy, MulticlassF1Score, ConfusionMatrix
from tqdm import tqdm
import os

try:
    from google.colab.patches import cv2_imshow
    import cv2
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    # 如果不在Colab中，提供替代的图像显示函数
    def cv2_imshow(img):
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

class AvgMeter(object):
    """
    移动平均计算器，用于记录和计算最近N个值的平均值
    """
    def __init__(self, num=40):
        """
        初始化移动平均计算器
        
        参数:
            num: 计算平均值时使用的最近值的数量
        """
        self.num = num
        self.reset()

    def reset(self):
        """重置计算器，清空所有记录的值"""
        self.losses = []

    def update(self, val):
        """
        添加新值到记录中
        
        参数:
            val: 要添加的新值
        """
        self.losses.append(val)

    def show(self):
        """
        计算并返回最近num个值的平均值
        
        返回:
            最近num个值的平均值
        """
        if not self.losses:
            return torch.tensor(0.0)
            
        out = torch.mean(
            torch.stack(
                self.losses[np.maximum(len(self.losses)-self.num, 0):]
            )
        )
        return out

class ModelWrapper(L.LightningModule):
    """
    模型封装器，用于训练、验证和测试EEG分类模型
    """
    def __init__(self, 
                 arch, 
                 dataset, 
                 batch_size=32, 
                 lr=0.001, 
                 max_epoch=100, 
                 num_classes=5,
                 class_weights=None):
        """
        初始化模型封装器
        
        参数:
            arch: 模型架构（nn.Module的子类实例）
            dataset: 数据集（应该是EEGDataset的实例）
            batch_size: 批量大小
            lr: 学习率
            max_epoch: 最大训练轮数
            num_classes: 分类类别数
            class_weights: 类别权重（用于处理类别不平衡）
        """
        super().__init__()

        self.arch = arch
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.max_epoch = max_epoch
        self.num_classes = num_classes
        self.class_weights = class_weights
        
        # 设置任务类型（多分类或二分类）
        if num_classes > 2:
            self.task = "multiclass"
            self.num_classes_arg = {"num_classes": num_classes}
        else:
            self.task = "binary"
            self.num_classes_arg = {}
        
        # 初始化评估指标
        self.train_accuracy = Accuracy(task=self.task, **self.num_classes_arg)
        self.val_accuracy = Accuracy(task=self.task, **self.num_classes_arg)
        self.test_accuracy = Accuracy(task=self.task, **self.num_classes_arg)
        self.test_f1 = MulticlassF1Score(num_classes=num_classes)
        self.test_confusion = ConfusionMatrix(task=self.task, **self.num_classes_arg)

        # 关闭自动优化，使用手动优化
        self.automatic_optimization = False

        # 初始化记录器
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.train_loss_recorder = AvgMeter()
        self.val_loss_recorder = AvgMeter()
        self.train_acc_recorder = AvgMeter()
        self.val_acc_recorder = AvgMeter()
        
        # 保存最佳验证精度
        self.best_val_acc = 0.0

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入数据
            
        返回:
            模型输出
        """
        return self.arch(x)

    def training_step(self, batch, batch_nb):
        """
        训练步骤
        
        参数:
            batch: 当前批次的数据
            batch_nb: 批次索引
        """
        x, y = batch
        y_hat = self(x)
        
        # 处理二分类和多分类情况
        if self.num_classes == 2:
            # 二分类
            y_onehot = F.one_hot(y, num_classes=2).float()
            loss = F.binary_cross_entropy_with_logits(y_hat, y_onehot)
        else:
            # 多分类 
            if self.class_weights is not None:
                weights = torch.tensor(self.class_weights).to(self.device)
                loss = F.cross_entropy(y_hat, y, weight=weights)
            else:
                loss = F.cross_entropy(y_hat, y)
        
        # 计算准确率
        self.train_accuracy.update(y_hat, y)
        acc = self.train_accuracy.compute().data.cpu()

        # 手动优化
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        # 更新记录
        self.train_loss_recorder.update(loss.data)
        self.train_acc_recorder.update(acc)

        # 记录指标
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

    def on_train_epoch_end(self):
        """每个训练轮结束时的操作"""
        # 更新学习率
        sch = self.lr_schedulers()
        sch.step()

        # 记录训练损失和准确率
        self.train_loss.append(self.train_loss_recorder.show().data.cpu().numpy())
        self.train_loss_recorder = AvgMeter()

        self.train_acc.append(self.train_acc_recorder.show().data.cpu().numpy())
        self.train_acc_recorder = AvgMeter()

    def validation_step(self, batch, batch_nb):
        """
        验证步骤
        
        参数:
            batch: 当前批次的数据
            batch_nb: 批次索引
        """
        x, y = batch
        y_hat = self(x)
        
        # 处理二分类和多分类情况
        if self.num_classes == 2:
            # 二分类
            y_onehot = F.one_hot(y, num_classes=2).float()
            loss = F.binary_cross_entropy_with_logits(y_hat, y_onehot)
        else:
            # 多分类
            loss = F.cross_entropy(y_hat, y)
        
        # 计算准确率
        self.val_accuracy.update(y_hat, y)
        acc = self.val_accuracy.compute().data.cpu()

        # 更新记录
        self.val_loss_recorder.update(loss.data)
        self.val_acc_recorder.update(acc)

        # 记录指标
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def on_validation_epoch_end(self):
        """每个验证轮结束时的操作"""
        # 记录验证损失和准确率
        val_loss = self.val_loss_recorder.show().data.cpu().numpy()
        self.val_loss.append(val_loss)
        self.val_loss_recorder = AvgMeter()

        val_acc = self.val_acc_recorder.show().data.cpu().numpy()
        self.val_acc.append(val_acc)
        self.val_acc_recorder = AvgMeter()
        
        # 记录最佳验证精度
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.log("best_val_acc", self.best_val_acc)

    def test_step(self, batch, batch_nb):
        """
        测试步骤
        
        参数:
            batch: 当前批次的数据
            batch_nb: 批次索引
        """
        x, y = batch
        y_hat = self(x)
        
        # 处理二分类和多分类情况
        if self.num_classes == 2:
            # 二分类
            y_onehot = F.one_hot(y, num_classes=2).float()
            loss = F.binary_cross_entropy_with_logits(y_hat, y_onehot)
        else:
            # 多分类
            loss = F.cross_entropy(y_hat, y)
        
        # 更新评估指标
        self.test_accuracy.update(y_hat, y)
        self.test_f1.update(y_hat, y)
        self.test_confusion.update(y_hat, y)

        # 记录指标
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_acc", self.test_accuracy.compute(), prog_bar=True, logger=True)

    def on_test_epoch_end(self):
        """测试结束时的操作"""
        # 计算并打印F1分数
        f1_score = self.test_f1.compute()
        print(f"测试F1分数: {f1_score}")
        
        # 获取混淆矩阵
        conf_matrix = self.test_confusion.compute()
        # 打印混淆矩阵
        print("混淆矩阵:")
        print(conf_matrix)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix.cpu(), interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混淆矩阵')
        plt.colorbar()
        
        classes = [f"类别 {i}" for i in range(self.num_classes)]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # 在混淆矩阵中标注数值
        thresh = conf_matrix.max() / 2.0
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j].item(), 'd'),
                        horizontalalignment="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        
        # 保存图像
        conf_matrix_file = "confusion_matrix.png"
        plt.savefig(conf_matrix_file)
        plt.close()
        
        # 如果在Colab中，显示图像
        if IN_COLAB and os.path.exists(conf_matrix_file):
            img = cv2.imread(conf_matrix_file)
            cv2_imshow(img)

    def on_train_end(self):
        """训练结束时的操作"""
        # 绘制损失曲线
        loss_img_file = "loss_plot.png"
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_loss, color='r', label='训练')
        plt.plot(self.val_loss, color='b', label='验证')
        plt.title("损失曲线")
        plt.xlabel("轮次")
        plt.ylabel("损失")
        plt.legend()
        plt.grid()
        plt.savefig(loss_img_file)
        plt.close()
        
        # 如果在Colab中，显示图像
        if IN_COLAB and os.path.exists(loss_img_file):
            img = cv2.imread(loss_img_file)
            cv2_imshow(img)

        # 绘制准确率曲线
        acc_img_file = "acc_plot.png"
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_acc, color='r', label='训练')
        plt.plot(self.val_acc, color='b', label='验证')
        plt.title("准确率曲线")
        plt.xlabel("轮次")
        plt.ylabel("准确率")
        plt.legend()
        plt.grid()
        plt.savefig(acc_img_file)
        plt.close()
        
        # 如果在Colab中，显示图像
        if IN_COLAB and os.path.exists(acc_img_file):
            img = cv2.imread(acc_img_file)
            cv2_imshow(img)
            
        print(f"最佳验证准确率: {self.best_val_acc:.4f}")

    def train_dataloader(self):
        """返回训练数据加载器"""
        return data.DataLoader(
            dataset=self.dataset.split("train"),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        """返回验证数据加载器"""
        return data.DataLoader(
            dataset=self.dataset.split("val"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def test_dataloader(self):
        """返回测试数据加载器"""
        return data.DataLoader(
            dataset=self.dataset.split("test"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
        )
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    int(self.max_epoch * 0.25),
                    int(self.max_epoch * 0.5),
                    int(self.max_epoch * 0.75),
                ],
                gamma=0.1
            ),
            "name": "lr_scheduler",
        }
        return [optimizer], [lr_scheduler]

def train_model(
    model_arch,
    dataset,
    batch_size=32,
    learning_rate=0.001,
    max_epochs=100,
    num_classes=5,
    class_weights=None,
    early_stopping=True,
    patience=10,
    save_model=True,
    save_dir="models"
):
    """
    训练EEG分类模型
    
    参数:
        model_arch: 模型架构（nn.Module的子类实例）
        dataset: 数据集（应该是EEGDataset的实例）
        batch_size: 批量大小
        learning_rate: 学习率
        max_epochs: 最大训练轮数
        num_classes: 分类类别数
        class_weights: 类别权重（用于处理类别不平衡）
        early_stopping: 是否使用早停
        patience: 早停的耐心值（多少轮验证损失没有改善后停止训练）
        save_model: 是否保存模型
        save_dir: 模型保存目录
        
    返回:
        训练好的模型封装器
    """
    # 初始化模型封装器
    model = ModelWrapper(
        arch=model_arch,
        dataset=dataset,
        batch_size=batch_size,
        lr=learning_rate,
        max_epoch=max_epochs,
        num_classes=num_classes,
        class_weights=class_weights
    )
    
    # 设置回调
    callbacks = []
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # 早停
    if early_stopping:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=True,
            mode='min'
        )
        callbacks.append(early_stop_callback)
    
    # 模型检查点
    if save_model:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        checkpoint_callback = ModelCheckpoint(
            dirpath=save_dir,
            filename='eeg-model-{epoch:02d}-{val_acc:.4f}',
            monitor='val_acc',
            mode='max',
            save_top_k=1,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
    
    # 初始化训练器
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        log_every_n_steps=10,
    )
    
    # 训练模型
    trainer.fit(model)
    
    # 测试模型
    trainer.test(model)
    
    return model

# 使用示例
if __name__ == "__main__":
    from torch import nn
    
    # 假设你已经加载了数据集
    # from eeg_data_processor import load_all_csv_files, EEGDataset
    # X_all, y_all = load_all_csv_files("your_data_folder")
    # dataset = EEGDataset(X_all, y_all)
    
    # 定义一个简单的EEG分类模型
    class SimpleEEGModel(nn.Module):
        def __init__(self, num_channels=8, num_classes=5):
            super().__init__()
            self.conv1 = nn.Conv1d(num_channels, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(2)
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Linear(32 * 62, 64)
            self.fc2 = nn.Linear(64, num_classes)
            
        def forward(self, x):
            # 输入形状: [batch_size, channels, time_points]
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.flatten(1)
            x = self.dropout(x)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # 设置类别权重（如果需要）
    # class_weights = [1.0, 1.0, 1.5, 1.0, 0.8]  # 增加"上"类别的权重
    
    # 训练模型
    # model_arch = SimpleEEGModel(num_channels=8, num_classes=5)
    # trained_model = train_model(
    #     model_arch=model_arch,
    #     dataset=dataset,
    #     batch_size=64,
    #     learning_rate=0.001,
    #     max_epochs=50,
    #     num_classes=5,
    #     class_weights=class_weights,
    #     early_stopping=True,
    #     patience=5,
    #     save_model=True
    # )