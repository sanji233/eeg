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

SAMPLING_FREQ = 250.0

LABEL_MAP = {
    769: "OVTK_GDF_Left", 
    770: "OVTK_GDF_Right",
    780: "OVTK_GDF_Up",
    774: "OVTK_GDF_Down",
    32769: "OVTK_StimulationId_ExperimentStart",
    32775: "OVTK_StimulationId_BaselineStart",
    33026: "OVTK_GDF_Feedback_Continuous"
}

NUMERIC_LABEL_MAP = {
    769: 0,  # 左
    770: 1,  # 右
    780: 2,  # 上
    774: 3,  # 下
}

CHANNEL_COLS = [
    'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4',
    'Channel 5', 'Channel 6', 'Channel 7', 'Channel 8'
]
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

import time
if __name__ == "__main__":
    # 读取CSV文件
    df = preprocess_csv('D:\\data\\code\\eeg\\OpenViBE\\data\\TEST\\motor-imagery-1-[2025.04.20-12.31.29].csv')
    raw = create_raw_from_dataframe(df)
    raw.plot(duration=60, n_channels=8, scalings='auto')
    plt.show()
