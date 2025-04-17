import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping

from data import EEGDataset
from models import EEGClassificationModel
from training import ModelWrapper
from utils import (
    preprocess_eeg_data, 
    convert_to_mne_format, 
    extract_epochs, 
    prepare_data_for_training,
    set_seed,
    plot_eeg_data
)

def main():
    # Configuration
    MAX_EPOCH = 600
    BATCH_SIZE = 10
    LR = 5e-4
    EEG_CHANNEL = 8
    CHECKPOINT_DIR = os.getcwd()
    SEED = int(np.random.randint(2147483647))
    set_seed(SEED)
    
    # Define event mapping
    label_map = {769: 0, 770: 1, 780: 2, 774: 3}
    event_id_dict = {
        'left': 769,
        'right': 770,
        'up': 780,
        'down': 774
    }
    
    # Process data
    try:
        # Load and preprocess your data - replace with your actual file path
        file_path = 'path/to/your/eeg_data.csv'
        df = preprocess_eeg_data(file_path)
        raw, channel_cols = convert_to_mne_format(df)
        epochs = extract_epochs(raw, df, event_id_dict)
        X, y = prepare_data_for_training(epochs, label_map)
        
        print(f"Data loaded successfully: {X.shape}")
        
        # Create dataset
        eeg_dataset = EEGDataset(X, y)
        
        # Create model
        base_model = EEGClassificationModel(eeg_channel=EEG_CHANNEL, dropout=0.125)
        
        # Wrap model for training
        wrapper = ModelWrapper(
            arch=base_model,
            dataset=eeg_dataset,
            batch_size=BATCH_SIZE,
            lr=LR,
            max_epoch=MAX_EPOCH
        )
        
        # Create trainer and train
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=30)
        ]
        
        trainer = L.Trainer(
            max_epochs=MAX_EPOCH,
            accelerator="auto",
            callbacks=callbacks
        )
        
        trainer.fit(wrapper)
        results = trainer.test(wrapper)
        
        # Save the model
        torch.save(base_model.state_dict(), "eeg_model.pth")
        
        print(f"Training complete. Test results: {results}")
        
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()