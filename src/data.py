"""
EEG数据处理模块 - 用于处理和准备EEG数据集

此模块提供了一整套工具，用于处理脑电图(EEG)数据，包括：
- 预处理CSV文件
- 创建MNE Raw对象
- 提取事件数据
- 处理EEG数据，使用完整的6秒epoch
- 加载和处理多个CSV文件
- 创建适用于PyTorch的EEG数据集
"""

from tqdm import tqdm 
import mne
import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings

# 设置常量
SAMPLING_FREQ = 250.0  # 采样频率
EPOCH_DURATION = 6.0   # 每个epoch的秒数 - 保持完整的6秒

# 标签映射
LABEL_MAP = {
    769: "OVTK_GDF_Left", 
    770: "OVTK_GDF_Right",
    780: "OVTK_GDF_Up",
    774: "OVTK_GDF_Down",
    32769: "OVTK_StimulationId_ExperimentStart",
    32775: "OVTK_StimulationId_BaselineStart",
    33026: "OVTK_GDF_Feedback_Continuous"
}

# 数值标签映射（用于分类）
NUMERIC_LABEL_MAP = {
    769: 0,  # 左
    770: 1,  # 右
    780: 2,  # 上
    774: 3,  # 下
    999: 4   # 静息
}

# 标签名称（用于显示）
LABEL_NAMES = ["左", "右", "上", "下", "静息"]

# EEG通道列
CHANNEL_COLS = [
    'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4',
    'Channel 5', 'Channel 6', 'Channel 7', 'Channel 8'
]

class LabelWeightAdjuster:
    """
    标签权重调整器，用于手动调整特定标签的样本数量
    """
    def __init__(self, weights=None):
        """
        初始化权重调整器
        
        参数:
            weights: 一个包含5个元素的列表或字典，表示每个标签的权重
                    如果是列表，顺序对应[左, 右, 上, 下, 静息]
                    如果是字典，键为标签编号(0-4)，值为权重
                    默认值为[1.0, 1.0, 1.0, 1.0, 1.0]表示不进行调整
        """
        # 设置默认权重
        self.weights = {i: 1.0 for i in range(5)} 
        
        # 处理输入的权重参数
        if weights is not None:
            if isinstance(weights, list) and len(weights) == 5:
                for i, w in enumerate(weights):
                    self.weights[i] = float(w)
            elif isinstance(weights, dict):
                for key, value in weights.items():
                    if key in range(5):
                        self.weights[key] = float(value)
            else:
                warnings.warn("权重格式不正确，将使用默认权重[1.0, 1.0, 1.0, 1.0, 1.0]")
    
    def adjust_samples(self, X, y):
        """
        根据权重调整样本数量
        
        参数:
            X: 形状为(n_samples, n_channels, n_times)的numpy数组
            y: 形状为(n_samples,)的numpy数组，类别标签
            
        返回:
            调整后的X和y
        """
        print("调整前样本统计:")
        for label in range(5):
            print(f"类别 {label} ({LABEL_NAMES[label]}): {np.sum(y == label)} 样本")
        
        # 对每个标签单独处理
        adjusted_X = []
        adjusted_y = []
        
        for label in range(5):
            # 获取当前标签的样本
            mask = (y == label)
            X_label = X[mask]
            y_label = y[mask]
            
            if len(X_label) == 0:
                continue
            
            weight = self.weights[label]
            
            # 如果权重大于1，则通过重复样本来增加数量
            if weight > 1.0:
                # 计算需要重复的次数
                repeat_count = int(weight)
                remainder = weight - repeat_count
                
                # 完整重复
                for _ in range(repeat_count):
                    adjusted_X.append(X_label)
                    adjusted_y.append(y_label)
                
                # 处理余数部分
                if remainder > 0:
                    samples_to_add = int(len(X_label) * remainder)
                    if samples_to_add > 0:
                        indices = np.random.choice(len(X_label), samples_to_add, replace=False)
                        adjusted_X.append(X_label[indices])
                        adjusted_y.append(y_label[indices])
            
            # 如果权重小于1，则随机选择部分样本
            elif weight < 1.0:
                samples_to_keep = max(int(len(X_label) * weight), 1)  # 至少保留一个样本
                indices = np.random.choice(len(X_label), samples_to_keep, replace=False)
                adjusted_X.append(X_label[indices])
                adjusted_y.append(y_label[indices])
            
            # 如果权重等于1，则保持不变
            else:
                adjusted_X.append(X_label)
                adjusted_y.append(y_label)
        
        # 合并所有标签的样本
        adjusted_X = np.vstack(adjusted_X) if adjusted_X else np.array([])
        adjusted_y = np.concatenate(adjusted_y) if adjusted_y else np.array([])
        
        print("调整后样本统计:")
        for label in range(5):
            print(f"类别 {label} ({LABEL_NAMES[label]}): {np.sum(adjusted_y == label)} 样本")
        
        return adjusted_X, adjusted_y

def preprocess_csv(file_path):
    """
    预处理单个CSV文件
    
    参数:
        file_path: CSV文件路径
        
    返回:
        预处理后的DataFrame
    """
    print(f"正在处理文件: {os.path.basename(file_path)}")
    
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 删除不需要的通道（如果存在）
    columns_to_drop = [col for col in ["Channel 9", "Channel 10", "Channel 11"] if col in df.columns]
    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True)
    
    # 处理事件ID、日期和持续时间列，保留冒号前的部分
    def keep_first_part(x):
        if not isinstance(x, str):
            x = "" if pd.isna(x) else str(x)
        return x.split(":")[0] if x else ""
    
    for col in ["Event Id", "Event Date", "Event Duration"]:
        if col in df.columns:
            df[col] = df[col].fillna("").apply(keep_first_part)
    
    # 清空重复事件行
    if "Event Id" in df.columns and "Event Date" in df.columns and "Event Duration" in df.columns:
        same_mask = df["Event Id"] == df["Event Id"].shift(-1)
        rows_to_blank = same_mask.index[same_mask]
        rows_to_blank = rows_to_blank + 1
        rows_to_blank = rows_to_blank[rows_to_blank < len(df)]
        df.loc[rows_to_blank, ["Event Id", "Event Date", "Event Duration"]] = ""
        
        # 转换为数值类型
        df["Event Id"] = pd.to_numeric(df["Event Id"], errors="coerce")
        df["Event Date"] = pd.to_numeric(df["Event Date"], errors="coerce")
    
    return df

def create_raw_from_dataframe(df):
    """
    从DataFrame创建MNE Raw对象
    
    参数:
        df: 包含EEG数据的DataFrame
        
    返回:
        MNE Raw对象
    """
    # 提取EEG数据
    data = df[CHANNEL_COLS].to_numpy().T
    
    # 创建MNE信息对象
    ch_names = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']
    ch_types = ['eeg'] * 8
    info = mne.create_info(ch_names=ch_names, sfreq=SAMPLING_FREQ, ch_types=ch_types)
    
    # 创建Raw对象
    raw = mne.io.RawArray(data, info)
    
    # 应用50Hz陷波滤波器（去除电源线噪声）
    raw.notch_filter(freqs=[50], picks='eeg')
    
    return raw

def extract_events_from_dataframe(df, raw):
    """
    从DataFrame提取事件数据
    
    参数:
        df: 包含事件信息的DataFrame
        raw: MNE Raw对象
        
    返回:
        numpy数组，形状为(n_events, 3)，包含事件信息
    """
    events = []
    
    if "Event Id" in df.columns and "Event Date" in df.columns:
        event_indices = df.index[~pd.isna(df["Event Id"]) & (df["Event Id"] != "")]
        
        for idx in event_indices:
            event_id = df.loc[idx, "Event Id"]
            event_date = df.loc[idx, "Event Date"]
            
            # 只保留任务相关事件
            if event_id in NUMERIC_LABEL_MAP.keys() and event_id != 999:  # 静息状态单独处理
                # 事件样本点位置
                sample = int(event_date)
                
                # 创建事件元组 [sample, 0, event_id]
                events.append([sample, 0, int(event_id)])
    
    return np.array(events, dtype=int) if events else np.empty((0, 3), dtype=int)

def process_eeg_data(raw, events, min_rest_epochs=None):
    """
    处理EEG数据，提取所有类别的完整epochs(不再切片)
    
    参数:
        raw: MNE Raw对象
        events: 事件数组
        min_rest_epochs: 最小静息epochs数量，如果为None，则自动判断
        
    返回:
        X_epochs: 形状为(n_epochs, n_channels, n_times)的numpy数组
        y_epochs: 形状为(n_epochs,)的numpy数组，标签
    """
    
    # 提取任务相关epochs（左、右、上、下）
    task_event_id = {
        'left': 769,
        'right': 770,
        'up': 780,
        'down': 774
    }
    
    task_epochs = mne.Epochs(
        raw,
        events=events,
        event_id=task_event_id,
        tmin=0.0,
        tmax=EPOCH_DURATION,
        baseline=None,
        picks='eeg',
        preload=True
    )
    
    X_task = (task_epochs.get_data() * 1e-6).astype(np.float32)
    y_task = np.array([NUMERIC_LABEL_MAP[val] for val in task_epochs.events[:, 2]], dtype=np.int64)
    
    print(f"任务相关epochs数量: {len(task_epochs)}")
    
    # 标记已被任务epochs占用的样本点
    used_samples = np.zeros(len(raw.times), dtype=bool)
    
    for idx in range(len(task_epochs)):
        event_sample = task_epochs.events[idx, 0]
        start_sample = event_sample
        end_sample = start_sample + int(EPOCH_DURATION * raw.info['sfreq'])
        if end_sample <= len(used_samples):
            used_samples[start_sample:end_sample] = True
    
    # 寻找未被占用的样本点作为静息epochs
    rest_events = []
    rest_length = int(EPOCH_DURATION * raw.info['sfreq'])
    
    # 确定需要提取的静息epochs数量
    if min_rest_epochs is None:
        # 默认尝试提取与任务epochs数量相当的静息epochs
        target_rest_epochs = len(task_epochs) // 4  # 每种任务类型的1/4
    else:
        target_rest_epochs = min_rest_epochs
    
    i = 0
    rest_count = 0
    while i < len(used_samples) and rest_count < target_rest_epochs:
        if not used_samples[i]:
            start_sample = i
            end_sample = min(start_sample + rest_length, len(used_samples))
            
            # 确保整个区间没有被占用
            if end_sample - start_sample >= rest_length and not any(used_samples[start_sample:end_sample]):
                rest_events.append([start_sample, 0, 999])  # 999为静息状态代码
                rest_count += 1
                i = end_sample
            else:
                i += 1
        else:
            i += 1
    
    rest_events = np.array(rest_events, dtype=int) if rest_events else np.empty((0, 3), dtype=int)
    
    # 创建静息epochs
    if len(rest_events) > 0:
        rest_event_id = {'rest': 999}
        rest_epochs = mne.Epochs(
            raw,
            events=rest_events,
            event_id=rest_event_id,
            tmin=0.0,
            tmax=EPOCH_DURATION,
            baseline=None,
            picks='eeg',
            preload=True
        )
        
        X_rest = (rest_epochs.get_data() * 1e-6).astype(np.float32)
        y_rest = np.ones(len(rest_epochs), dtype=np.int64) * NUMERIC_LABEL_MAP[999]
        
        print(f"静息epochs数量: {len(rest_epochs)}")
        
        # 合并任务和静息数据
        X_combined = np.vstack([X_task, X_rest])
        y_combined = np.concatenate([y_task, y_rest])
    else:
        print("未找到静息epochs")
        X_combined = X_task
        y_combined = y_task
    
    # 移除切片处理代码，直接返回完整的epochs数据
    X_epochs = X_combined
    y_epochs = y_combined
    
    print(f"epochs总数: {len(X_epochs)}")
    print(f"数据形状: {X_epochs.shape}")

    return X_epochs, y_epochs

def get_eeg_channels(raw):
    """
    获取EEG通道数
    
    参数:
        raw: MNE Raw对象
        
    返回:
        EEG通道数
    """
    eeg_channel_inds = mne.pick_types(
        raw.info,
        meg=False,
        eeg=True,
        stim=False,
        eog=False,
        exclude='bads',
    )
    return len(eeg_channel_inds)

def load_all_csv_files(folder_path, pattern="*.csv", label_weights=None, min_rest_epochs=None):
    """
    加载文件夹中所有符合模式的CSV文件，并处理它们
    
    参数:
        folder_path: 包含CSV文件的文件夹路径
        pattern: 文件匹配模式，默认为"*.csv"
        label_weights: 标签权重，用于调整样本数量
        min_rest_epochs: 最小静息epochs数量
        
    返回:
        X_all: 所有epochs的特征数据
        y_all: 所有epochs的标签
    """
    all_X_epochs = []
    all_y_epochs = []
    
    # 获取所有匹配的CSV文件
    csv_files = glob.glob(os.path.join(folder_path, pattern))
    
    if not csv_files:
        print(f"在 {folder_path} 中未找到符合 {pattern} 的CSV文件")
        return None, None
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 处理每个CSV文件
    for file_path in tqdm(csv_files, desc="处理文件"):
        try:
            # 预处理CSV
            df = preprocess_csv(file_path)
            
            # 创建Raw对象
            raw = create_raw_from_dataframe(df)
            
            # 提取事件
            events = extract_events_from_dataframe(df, raw)
            
            if len(events) > 0:
                # 处理EEG数据并获取epochs
                X_epochs, y_epochs = process_eeg_data(raw, events, min_rest_epochs)
                
                # 添加到总集合
                all_X_epochs.append(X_epochs)
                all_y_epochs.append(y_epochs)
            else:
                print(f"文件 {os.path.basename(file_path)} 中未找到有效事件")
        
        except Exception as e:
            print(f"处理文件 {os.path.basename(file_path)} 时出错: {str(e)}")
    
    # 合并所有数据
    if all_X_epochs and all_y_epochs:
        X_all = np.vstack(all_X_epochs)
        y_all = np.concatenate(all_y_epochs)
        
        print(f"所有文件处理完成!")
        print(f"总数据形状: {X_all.shape}")
        print(f"总标签形状: {y_all.shape}")
        
        # 显示每个类别的样本计数
        for i in range(5):
            count = np.sum(y_all == i)
            percent = count / len(y_all) * 100
            print(f"类别 {i} ({LABEL_NAMES[i]}): {count} 样本 ({percent:.2f}%)")
        
        # 应用标签权重调整
        if label_weights is not None:
            weight_adjuster = LabelWeightAdjuster(label_weights)
            X_all, y_all = weight_adjuster.adjust_samples(X_all, y_all)
        
        return X_all, y_all
    else:
        print("未能从任何文件中提取到有效数据")
        return None, None

class EEGDataset(Dataset):
    """
    处理EEG数据的Dataset类，支持五类分类（左、右、上、下、静息）
    并提供训练、验证和测试集的拆分功能
    """
    def __init__(self, x, y=None, inference=False, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
        """
        初始化EEG数据集
        
        参数:
            x: 形状为(n_samples, n_channels, n_times)的numpy数组
            y: 形状为(n_samples,)的numpy数组，类别标签(0-4对应左、右、上、下、静息)
            inference: 是否为推理模式（不需要标签）
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_state: 随机种子，用于数据集拆分
        """
        super().__init__()
        self.__split = None
        
        # 确保比例和为1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "数据集比例必须和为1"
        
        if not inference:
            # 使用sklearn的train_test_split进行更可靠的拆分
            # 首先分离出测试集
            X_temp, X_test, y_temp, y_test = train_test_split(
                x, y, test_size=test_ratio, random_state=random_state, stratify=y
            )
            
            # 再从剩余数据中分离出验证集
            val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)  # 调整验证集比例
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio_adjusted, 
                random_state=random_state, stratify=y_temp
            )
            
            self.train_ds = {
                'x': X_train,
                'y': y_train,
            }
            self.val_ds = {
                'x': X_val,
                'y': y_val,
            }
            self.test_ds = {
                'x': X_test,
                'y': y_test,
            }
            
            # 打印每个集合的样本数和类别分布
            print(f"训练集大小: {len(X_train)} 样本")
            print(f"验证集大小: {len(X_val)} 样本")
            print(f"测试集大小: {len(X_test)} 样本")
            
            for i in range(5):  # 五类: 0-左, 1-右, 2-上, 3-下, 4-静息
                print(f"类别 {i} ({LABEL_NAMES[i]}) 分布: 训练集 {sum(y_train == i)}, "
                      f"验证集 {sum(y_val == i)}, 测试集 {sum(y_test == i)}")
        else:
            self.inference_ds = {'x': x}
            print(f"推理数据集大小: {len(x)} 样本")

    def __len__(self):
        """返回当前拆分的数据集长度"""
        return len(self.dataset['x'])

    def __getitem__(self, idx):
        """获取指定索引的样本"""
        x_ = torch.tensor(self.dataset['x'][idx], dtype=torch.float32)  # shape=(n_channels, n_times)
        
        if self.__split != "inference":
            y_ = torch.tensor(self.dataset['y'][idx], dtype=torch.long)  # 使用long类型用于分类
            return x_, y_
        else:
            return x_

    def split(self, __split):
        """设置当前使用的数据集拆分"""
        self.__split = __split
        return self

    @property
    def dataset(self):
        """根据当前拆分返回相应的数据集"""
        assert self.__split is not None, "必须先指定数据集拆分(train/val/test/inference)!"
        
        if self.__split == "train":
            return self.train_ds
        elif self.__split == "val":
            return self.val_ds
        elif self.__split == "test":
            return self.test_ds
        elif self.__split == "inference":
            return self.inference_ds
        else:
            raise ValueError(f"未知的数据集拆分: {self.__split}")

    def get_loaders(self, batch_size=32, num_workers=4):
        """获取所有数据加载器"""
        train_loader = DataLoader(
            self.split("train"), batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            self.split("val"), batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )
        
        test_loader = DataLoader(
            self.split("test"), batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader



'''
from data import load_all_csv_files, EEGDataset

# 设置标签权重（可选）
label_weights = {0: 1.0, 1: 1.0, 2: 1.5, 3: 1.0, 4: 1.0}  # 增加"上"类别的权重

# 加载数据
data_folder = "你的数据文件夹路径"
X_all, y_all = load_all_csv_files(data_folder, label_weights=label_weights)

# 创建数据集
eeg_dataset = EEGDataset(X_all, y_all)

# 获取数据加载器
train_loader, val_loader, test_loader = eeg_dataset.get_loaders(batch_size=64)
'''



if __name__ == "__main__":
    # 设置标签权重 (示例: 增加"上"类别的权重为1.5)
    label_weights = {0: 1.0, 1: 1.0, 2: 1.5, 3: 1.0, 4: 1.0}
    
    # 加载数据
    data_folder = "eeg/OpenViBE/data"
    X_all, y_all = load_all_csv_files(data_folder, label_weights=label_weights)
    
    if X_all is not None and y_all is not None:
        # 创建数据集
        eeg_dataset = EEGDataset(X_all, y_all)
        
        # 获取数据加载器
        train_loader, val_loader, test_loader = eeg_dataset.get_loaders(batch_size=64)