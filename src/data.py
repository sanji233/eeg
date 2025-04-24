import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import mne
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split








class EEGConfig:
    """EEG数据处理的配置类"""
    
    # 采样频率
    SAMPLING_FREQ = 250.0
    
    # 事件ID到名称的映射
    LABEL_MAP = {
        769: "OVTK_GDF_Left", 
        770: "OVTK_GDF_Right",
        780: "OVTK_GDF_Up",
        774: "OVTK_GDF_Down",
        32769: "OVTK_StimulationId_ExperimentStart",
        32775: "OVTK_StimulationId_BaselineStart",
        33026: "OVTK_GDF_Feedback_Continuous"
    }
    
    # 定义静息状态的事件ID
    REST_STATE_EVENT_ID = 999
    
    # 事件ID到数值标签的映射
    NUMERIC_LABEL_MAP = {
        769: 0,  # 左
        770: 1,  # 右
        780: 2,  # 上
        774: 3,  # 下
    }
    
    # 静息状态标签
    REST_STATE_LABEL = 4
    
    # 定义标签名称（用于输出）
    LABEL_NAMES = ["左", "右", "上", "下", "静息"]
    
    # 通道列名
    CHANNEL_COLS = [
        'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4',
        'Channel 5', 'Channel 6', 'Channel 7', 'Channel 8'
    ]
    
    # 滤波器参数
    FILTER_L_FREQ = 0  # 低频截止，Hz
    FILTER_H_FREQ = 60  # 高频截止，Hz
    NOTCH_FREQ = 50  # 陷波频率（电源线），Hz
    NUM_CHANNELS = 8
    # 窗口参数默认值
    DEFAULT_WINDOW_DURATION = 2.0  # 秒
    DEFAULT_WINDOW_STEP = 0.2  # 秒
    DEFAULT_STATE_DURATION = 6.0  # 秒


class EEGPreprocessor:
    """EEG数据预处理类"""
    
    def __init__(self, config=None):
        """
        初始化预处理器
        
        参数:
            config: 配置对象，默认使用EEGConfig
        """
        self.config = config or EEGConfig()
    
    def preprocess_csv(self, file_path):
        """
        预处理单个CSV文件
        
        参数:
            file_path: CSV文件路径
            
        返回:
            预处理后的DataFrame
        """
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 删除不需要的通道（如果存在）
        columns_to_drop = [col for col in ["Channel 9", "Channel 10", "Channel 11"] if col in df.columns]
        if columns_to_drop:
            df.drop(columns=columns_to_drop, inplace=True)
        
        # 处理事件ID、日期和持续时间列，保留冒号前的部分
        def keep_first_part(x):
            try:
                if not isinstance(x, str):
                    x = "" if pd.isna(x) else str(x)
                return x.split(":")[0] if x and ":" in x else x
            except Exception:
                return "" if pd.isna(x) else str(x)
        
        for col in ["Event Id", "Event Date", "Event Duration"]:
            if col in df.columns:
                df[col] = df[col].fillna("").apply(keep_first_part)
        
        # 清空重复事件行
        if "Event Id" in df.columns and "Event Date" in df.columns and "Event Duration" in df.columns:
            try:
                same_mask = df["Event Id"] == df["Event Id"].shift(-1)
                rows_to_blank = same_mask.index[same_mask]
                rows_to_blank = rows_to_blank + 1
                rows_to_blank = rows_to_blank[rows_to_blank < len(df)]
                df.loc[rows_to_blank, ["Event Id", "Event Date", "Event Duration"]] = ""
            except Exception as e:
                print(f"处理重复事件行时出错: {str(e)}")
            
            try:
                df["Event Id"] = pd.to_numeric(df["Event Id"], errors="coerce")
                df["Event Date"] = pd.to_numeric(df["Event Date"], errors="coerce")
            except Exception as e:
                print(f"转换事件列为数值类型时出错: {str(e)}")
        
        return df
    
    def create_raw_from_dataframe(self, df):
        """
        从DataFrame创建MNE Raw对象
        
        参数:
            df: 包含EEG数据的DataFrame
            
        返回:
            MNE Raw对象
        """
        # 提取EEG数据
        data = df[self.config.CHANNEL_COLS].to_numpy().T
        
        # 创建MNE信息对象
        ch_names = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']
        ch_types = ['eeg'] * 8
        info = mne.create_info(ch_names=ch_names, sfreq=self.config.SAMPLING_FREQ, ch_types=ch_types)
        
        # 创建Raw对象
        raw = mne.io.RawArray(data, info)
        
        # 应用陷波滤波器（去除电源线噪声）
        raw.notch_filter(freqs=[self.config.NOTCH_FREQ], picks='eeg')
        
        return raw
    
    def extract_events_from_dataframe(self, df, raw):
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
            # 获取有效事件的索引
            event_indices = df.index[~pd.isna(df["Event Id"]) & (df["Event Id"] != "")]
            
            for idx in event_indices:
                try:
                    event_id = df.loc[idx, "Event Id"]
                    event_date = df.loc[idx, "Event Date"]
                        
                    # 只保留任务相关事件（包括静息状态）
                    if event_id in self.config.NUMERIC_LABEL_MAP.keys() or event_id == self.config.REST_STATE_EVENT_ID:
                        # 事件样本点位置
                        sample = int(float(event_date) * self.config.SAMPLING_FREQ)
                        
                        # 创建事件元组 [sample, 0, event_id]
                        events.append([sample, 0, int(event_id)])
                except Exception as e:
                    print(f"处理事件索引 {idx} 时出错: {str(e)}")
        
        return np.array(events, dtype=int) if events else np.empty((0, 3), dtype=int)
    
    def create_sliding_window_with_labels(self, raw, events, window_duration=None, window_step=None, state_duration=6):
        """
        使用滑动窗口切片EEG数据，并根据窗口是否与状态持续段重叠来标记

        参数:
            raw: MNE Raw对象
            events: 事件数组，形状为(n_events, 3)
            window_duration: 滑动窗口大小，单位为秒
            window_step: 滑动窗口步长，单位为秒
            state_duration: 从事件标记点开始的状态持续时间，单位为秒
            
        返回:
            X_windows: 窗口数据，形状为(n_windows, n_channels, n_times)
            y_labels: 窗口标签，形状为(n_windows,)
        """
        # 使用默认值（如果未指定）
        window_duration = window_duration or self.config.DEFAULT_WINDOW_DURATION
        window_step = window_step or self.config.DEFAULT_WINDOW_STEP
        state_duration = state_duration or self.config.DEFAULT_STATE_DURATION
        
        # 获取采样率
        sfreq = raw.info['sfreq']
        n_channels = len(raw.ch_names)
        
        # 计算窗口和状态持续时间的样本点数
        window_samples = int(window_duration * sfreq)
        step_samples = int(window_step * sfreq)
        state_samples = int(state_duration * sfreq)
        
        # 创建状态区间标记数组 (0表示未标记的背景状态，不同于REST_STATE_LABEL)
        n_samples = len(raw.times)
        state_markers = np.zeros(n_samples, dtype=np.int64)

        #DEBUG用
        print("处理的总事件数:", len(events))
        processed_count = 0
        #DEBUG用

        # 标记所有状态持续段
        for event in events:
            event_sample, _, event_id = event
    
            # 处理任务相关事件
            if event_id in self.config.NUMERIC_LABEL_MAP:
        # 计算状态持续段开始和结束点
                start_sample = event_sample
                end_sample = min(start_sample + state_samples, n_samples)
        
        # 用事件对应的数字标签标记该区间
                label = self.config.NUMERIC_LABEL_MAP[event_id]
                state_markers[start_sample:end_sample] = label
                processed_count += 1
        # 提取原始EEG数据
        eeg_data = raw.get_data() * 1e-6  # 转换为μV
        
        # 创建滑动窗口
        X_windows = []
        y_labels = []
        
        # 遍历所有可能的窗口位置
        for start_sample in range(0, n_samples - window_samples + 1, step_samples):
            end_sample = start_sample + window_samples
            
            # 提取当前窗口的数据
            window_data = eeg_data[:, start_sample:end_sample]
            
            # 确定窗口标签
            window_states = state_markers[start_sample:end_sample]
            unique_states, state_counts = np.unique(window_states, return_counts=True)
            
            # 如果窗口中有标记的状态，选择占比最大的状态作为标签
            # 排除未标记状态(0)，除非所有状态都是未标记状态
            valid_states = unique_states[unique_states != 0] if 0 in unique_states else unique_states
            valid_counts = state_counts[unique_states != 0] if 0 in unique_states else state_counts
            
            if len(valid_states) > 0:
                # 找出样本点数量最多的状态
                max_idx = np.argmax(valid_counts)
                label = valid_states[max_idx]
            else:
                # 所有状态都是未标记状态，将其视为静息
                label = self.config.REST_STATE_LABEL
            
            X_windows.append(window_data)
            y_labels.append(label)
        
        # 转换为numpy数组
        X_windows = np.array(X_windows, dtype=np.float32)
        y_labels = np.array(y_labels, dtype=np.int64)
        
        print(f"创建了 {len(X_windows)} 个滑动窗口")
        print(f"窗口数据形状: {X_windows.shape}")
        
        # 输出各类别数量
        for i in range(5):  # 5个类别：0-左, 1-右, 2-上, 3-下, 4-静息
            count = np.sum(y_labels == i)
            print(f"类别 {i} ({self.config.LABEL_NAMES[i]}): {count} 窗口")
        
        return X_windows, y_labels
    
    def process_file(self, file_path, window_duration=None, window_step=None, state_duration=None):
        """
        处理单个EEG文件
        
        参数:
            file_path: CSV文件路径
            window_duration: 滑动窗口大小，单位为秒
            window_step: 滑动窗口步长，单位为秒
            state_duration: 从事件标记点开始的状态持续时间，单位为秒
            
        返回:
            X_windows: 窗口数据，形状为(n_windows, n_channels, n_times)
            y_labels: 窗口标签，形状为(n_windows,)
        """
        try:
            # 预处理CSV
            df = self.preprocess_csv(file_path)
            
            # 创建Raw对象
            raw = self.create_raw_from_dataframe(df)
            
            # 应用带通滤波器
            raw.filter(l_freq=self.config.FILTER_L_FREQ, h_freq=self.config.FILTER_H_FREQ, picks='eeg')
            
            # 提取事件
            events = self.extract_events_from_dataframe(df, raw)
            
            if len(events) > 0:
                # 使用滑动窗口切片并标记
                X_windows, y_labels = self.create_sliding_window_with_labels(
                    raw, events, 
                    window_duration=window_duration, 
                    window_step=window_step,
                    state_duration=state_duration
                )
                
                return X_windows, y_labels
            else:
                print(f"文件 {os.path.basename(file_path)} 中未找到有效事件")
                return None, None
        
        except Exception as e:
            print(f"处理文件 {os.path.basename(file_path)} 时出错: {str(e)}")
            return None, None
    
    def process_folder(self, folder_path, window_duration=None, window_step=None, state_duration=None):
        """
        处理文件夹中的所有EEG文件
        
        参数:
            folder_path: 包含CSV文件的文件夹路径
            window_duration: 滑动窗口大小，单位为秒
            window_step: 滑动窗口步长，单位为秒
            state_duration: 从事件标记点开始的状态持续时间，单位为秒
            
        返回:
            X_all: 所有窗口数据，形状为(n_windows_total, n_channels, n_times)
            y_all: 所有窗口标签，形状为(n_windows_total,)
        """
        # 获取所有匹配的CSV文件
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        if not csv_files:
            print(f"在 {folder_path} 中未找到CSV文件")
            return None, None
        
        print(f"找到 {len(csv_files)} 个CSV文件")
        
        all_X_windows = []
        all_y_labels = []
        
        # 处理每个CSV文件
        for file_path in tqdm(csv_files, desc="处理文件"):
            X_windows, y_labels = self.process_file(
                file_path, 
                window_duration=window_duration, 
                window_step=window_step,
                state_duration=state_duration
            )
            
            if X_windows is not None and y_labels is not None:
                all_X_windows.append(X_windows)
                all_y_labels.append(y_labels)
        
        # 合并所有数据
        if all_X_windows and all_y_labels:
            X_all = np.vstack(all_X_windows)
            y_all = np.concatenate(all_y_labels)
            
            print(f"所有文件处理完成!")
            print(f"总数据形状: {X_all.shape}")
            print(f"总标签形状: {y_all.shape}")
            
            # 显示每个类别的样本计数
            for i in range(5):  # 5个类别
                count = np.sum(y_all == i)
                percent = count / len(y_all) * 100
                print(f"类别 {i} ({self.config.LABEL_NAMES[i]}): {count} 样本 ({percent:.2f}%)")
            
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
        self.config = EEGConfig()
        
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
            
            # 设置默认拆分为训练集
            self.__split = "train"
            
            # 打印每个集合的样本数和类别分布
            print(f"训练集大小: {len(X_train)} 样本")
            print(f"验证集大小: {len(X_val)} 样本")
            print(f"测试集大小: {len(X_test)} 样本")
            
            for i in range(5):  # 五类: 0-左, 1-右, 2-上, 3-下, 4-静息
                print(f"类别 {i} ({self.config.LABEL_NAMES[i]}) 分布: 训练集 {sum(y_train == i)}, "
                      f"验证集 {sum(y_val == i)}, 测试集 {sum(y_test == i)}")
        else:
            self.inference_ds = {'x': x}
            self.__split = "inference"
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
        assert __split in ["train", "val", "test", "inference"], f"未知的数据集拆分: {__split}"
        if __split == "inference" and not hasattr(self, "inference_ds"):
            raise ValueError("当前数据集不支持推理模式，请以inference=True初始化")
            
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


class EEGProcessor:
    """
    EEG数据处理的主类，集成预处理和数据集创建
    """
    def __init__(self, config=None):
        """
        初始化EEG处理器
        
        参数:
            config: 配置对象，默认使用EEGConfig
        """
        self.config = config or EEGConfig()
        self.preprocessor = EEGPreprocessor(config=self.config)
    
    def prepare_dataset(self, folder_path, window_duration=None, window_step=None, state_duration=None, 
                    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
        """
        准备用于训练的EEG数据集
        
        参数:
            folder_path: 包含CSV文件的文件夹路径
            window_duration: 滑动窗口大小，单位为秒
            window_step: 滑动窗口步长，单位为秒
            state_duration: 从事件标记点开始的状态持续时间，单位为秒
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_state: 随机种子，用于数据集拆分
            
        返回:
            EEGDataset对象
        """
        # 处理文件夹中的所有EEG文件
        X_all, y_all = self.preprocessor.process_folder(
            folder_path,
            window_duration=window_duration,
            window_step=window_step,
            state_duration=state_duration
        )
        
        if X_all is not None and y_all is not None:
            # 减少静息样本到10%
            REST_STATE_LABEL = 4  # 静息状态标签
            rest_mask = y_all == REST_STATE_LABEL
            rest_indices = np.where(rest_mask)[0]
            non_rest_indices = np.where(~rest_mask)[0]
            
            # 计算要保留的静息样本数量（10%）
            keep_rest_count = int(len(rest_indices) * 0.15)
            
            # 随机选择要保留的静息样本
            if keep_rest_count > 0:
                # 设置随机种子确保可重复性
                np.random.seed(random_state)
                keep_rest_indices = np.random.choice(rest_indices, size=keep_rest_count, replace=False)
                keep_indices = np.concatenate([non_rest_indices, keep_rest_indices])
                
                # 减少数据集
                X_all = X_all[keep_indices]
                y_all = y_all[keep_indices]
                
                print(f"静息样本减少: 从 {len(rest_indices)} 减少到 {keep_rest_count} (10%)")
                print(f"总样本量: 从 {len(rest_mask)} 减少到 {len(y_all)}")
            
            # 创建数据集
            eeg_dataset = EEGDataset(
                x=X_all, 
                y=y_all, 
                inference=False,
                train_ratio=train_ratio, 
                val_ratio=val_ratio, 
                test_ratio=test_ratio, 
                random_state=random_state
            )
            return eeg_dataset
        else:
            return None