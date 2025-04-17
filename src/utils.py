import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch

def preprocess_eeg_data(file_path):
    """Preprocess EEG data from CSV file"""
    # Read and clean the data
    df = pd.read_csv(file_path)
    df.drop(columns=["Channel 9", "Channel 10", "Channel 11"], inplace=True)
    
    def keep_first_part(x):
        if not isinstance(x, str):
            x = "" if pd.isna(x) else str(x)
        return x.split(":")[0] if x else ""
    
    for col in ["Event Id", "Event Date", "Event Duration"]:
        df[col] = df[col].fillna("").apply(keep_first_part)
    
    same_mask = df["Event Id"] == df["Event Id"].shift(-1)
    rows_to_blank = same_mask.index[same_mask]
    rows_to_blank = rows_to_blank + 1
    rows_to_blank = rows_to_blank[rows_to_blank < len(df)]
    df.loc[rows_to_blank, ["Event Id", "Event Date", "Event Duration"]] = ""
    df["Event Id"] = pd.to_numeric(df["Event Id"], errors="coerce")
    df["Event Date"] = pd.to_numeric(df["Event Date"], errors="coerce")
    
    return df

def convert_to_mne_format(df):
    """Convert DataFrame to MNE format and apply filters"""
    channel_cols = [
        'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4',
        'Channel 5', 'Channel 6', 'Channel 7', 'Channel 8'
    ]
    
    data = df[channel_cols].to_numpy().T
    ch_names = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']
    ch_types = ['eeg'] * 8
    sfreq = 250.0
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    raw.notch_filter(freqs=[50], picks='eeg')
    
    return raw, channel_cols

def extract_epochs(raw, df, event_id_dict):
    """Extract epochs from raw EEG data"""
    df['Event Id'] = df['Event Id'].fillna(0).astype(int)
    event_rows = df.index[df['Event Id'] != 0]
    event_codes = df.loc[event_rows, 'Event Id'].values
    events = np.column_stack((
        event_rows,            # First column: sample points
        np.zeros_like(event_rows),  # Second column: zeros
        event_codes           # Third column: event codes
    ))
    
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id_dict,
        tmin=0.0,
        tmax=3.0,
        baseline=None,
        picks='eeg',
        preload=True
    )
    
    return epochs

def prepare_data_for_training(epochs, label_map):
    """Prepare data for training"""
    X = epochs.get_data() * 1e-6  # Convert to more manageable units
    X = X.astype(np.float32)
    raw_labels = epochs.events[:, 2]
    y = np.array([label_map[val] for val in raw_labels], dtype=np.int64)
    
    return X, y

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed: {seed}")

def plot_eeg_data(data, channel_idx, epoch_range):
    """Plot EEG data for visualization"""
    data_for_plot = data[epoch_range, channel_idx, :].T
    
    plt.figure(figsize=(10, 5))
    plt.plot(data_for_plot)
    plt.title(f"Channel {channel_idx}, multiple epochs {epoch_range}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend([f"Ep {ep}" for ep in epoch_range])
    plt.show()
    plt.clf()