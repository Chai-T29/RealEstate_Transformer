import os
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
from dateutil.relativedelta import relativedelta


def custom_collate_fn(batch):
    """Collate function.
    
    Args:
        batch (list): List of sample dicts.
            Each sample contains:
            - "housing_x": [T, F, P] torch.Tensor
            - "rental_x": [T] torch.Tensor
            - "housing_y": [pred_window, P] torch.Tensor or None
            - "rental_y": [pred_window] torch.Tensor
            - "zipcode": identifier
            - "zipcode_idx": int
            - "time": sequence of timestamps
    Returns:
        dict: Batch dictionary with:
            - housing_x: [B, T, F, P]
            - rental_x: [B, T]
            - housing_y: [B, pred_window, P] or None
            - rental_y: [B, pred_window]
            - zipcode: list of identifiers
            - zipcode_idx: [B] torch.Tensor
            - time: same as in sample
    """
    housing_x = torch.stack([sample["housing_x"] for sample in batch])
    rental_x = torch.stack([sample["rental_x"] for sample in batch])
    housing_y = torch.stack([sample["housing_y"] for sample in batch]) if batch[0]["housing_y"] is not None else None
    rental_y = torch.stack([sample["rental_y"] for sample in batch])
    batch_zip = [sample["zipcode"] for sample in batch]
    batch_zip_idx = torch.tensor([sample["zipcode_idx"] for sample in batch], dtype=torch.long)
    batch_time = batch[0]["time"]
    return {
        "housing_x": housing_x,
        "rental_x": rental_x,
        "housing_y": housing_y,
        "rental_y": rental_y,
        "zipcode": batch_zip,
        "zipcode_idx": batch_zip_idx,
        "time": batch_time
    }


class HousingWindowDataset(Dataset):
    """Dataset for training/validation housing & rental data.
    
    Args:
        data (np.ndarray): 4D array [num_zip, T, F, P].
        zipcodes (np.ndarray): Array of zipcode identifiers.
        zipcode_idx (np.ndarray): Array of zipcode indices.
        times (array-like): Timestamps.
        features (np.ndarray): Feature names.
        window_size (int): Input window length (T_window).
        prediction_window (list): List of prediction offsets.
        rental_data (np.ndarray): 2D array [num_zip, T].
    """
    def __init__(self, data, zipcodes, zipcode_idx, times, features, window_size, prediction_window, rental_data):
        self.data = data
        self.zipcodes = zipcodes
        self.zipcode_idx = zipcode_idx
        self.times = times
        self.features = features
        self.window_size = window_size
        self.prediction_window = prediction_window
        self.rental_data = rental_data
        self.y_ind = np.where(self.features == "median_sale_price")[0]
        self.valid_indices = self._build_index()

    def _build_index(self):
        """Build valid (zipcode, time) index pairs.
        
        Returns:
            list: List of (i, t) pairs.
        """
        valid_pairs = []
        pmin = min(self.prediction_window)
        num_zip, T, _, _ = self.data.shape
        for i in range(num_zip):
            for t in range(T - self.window_size):
                if (t + self.window_size - 1 + pmin) < T:
                    valid_pairs.append((i, t))
        return valid_pairs

    def __len__(self):
        """Length of dataset.
        
        Returns:
            int: Number of valid indices.
        """
        return len(self.valid_indices)

    def __getitem__(self, index):
        """Get sample.
        
        Args:
            index (int): Sample index.
        Returns:
            dict: Sample with keys:
                - "housing_x": [window_size, F, P] torch.Tensor
                - "rental_x": [window_size] torch.Tensor
                - "housing_y": [pred_window, P] torch.Tensor (NaNs replaced by 0)
                - "rental_y": [pred_window] torch.Tensor (NaNs replaced by 0)
                - "zipcode": zipcode identifier
                - "zipcode_idx": index (int)
                - "time": [window_size] torch.Tensor of timestamps
        """
        i, t = self.valid_indices[index]
        housing_X = self.data[i, t:t+self.window_size, :, :].astype(np.float32)
        rental_X = self.rental_data[i, t:t+self.window_size].astype(np.float32)
        
        housing_targets = []
        rental_targets = []
        for p in self.prediction_window:
            target_time = t + self.window_size - 1 + p
            if target_time < self.data.shape[1]:
                housing_slice = self.data[i, target_time, self.y_ind, :].astype(np.float32)
                housing_targets.append(torch.from_numpy(housing_slice))
                rental_value = self.rental_data[i, target_time].astype(np.float32)
                rental_targets.append(torch.tensor(rental_value))
            else:
                shape = (len(self.y_ind), self.data.shape[-1])
                housing_targets.append(torch.from_numpy(np.full(shape, np.nan, dtype=np.float32)))
                rental_targets.append(torch.tensor(np.nan, dtype=torch.float32))
        housing_y = torch.nan_to_num(torch.stack(housing_targets), nan=0.0)
        rental_y = torch.nan_to_num(torch.stack(rental_targets), nan=0.0)
        
        time_seq = torch.tensor([tm.timestamp() for tm in self.times[t:t+self.window_size]], dtype=torch.float32)
        return {
            "housing_x": torch.from_numpy(housing_X),
            "rental_x": torch.from_numpy(rental_X),
            "housing_y": housing_y,
            "rental_y": rental_y,
            "zipcode": self.zipcodes[i],
            "zipcode_idx": self.zipcode_idx[i],
            "time": time_seq
        }


class HousingWindowTestDataset(Dataset):
    """Test dataset for housing & rental data.
    
    Args:
        (Same as HousingWindowDataset)
    """
    def __init__(self, data, zipcodes, zipcode_idx, times, features, window_size, prediction_window, rental_data):
        self.data = data
        self.zipcodes = zipcodes
        self.zipcode_idx = zipcode_idx
        self.times = times
        self.features = features
        self.window_size = window_size
        self.prediction_window = prediction_window
        self.rental_data = rental_data
        self.valid_indices = self._build_index()

    def _build_index(self):
        """Build valid test index pairs.
        
        Returns:
            list: List of (i, t) pairs.
        """
        valid_pairs = []
        pmin = min(self.prediction_window)
        num_zip, T, _, _ = self.data.shape
        for i in range(num_zip):
            for t in range(T - self.window_size + 1):
                if (t + self.window_size - 1 + pmin) >= T:
                    valid_pairs.append((i, t))
        return valid_pairs

    def __len__(self):
        """Dataset length.
        
        Returns:
            int: Number of valid indices.
        """
        return len(self.valid_indices)

    def __getitem__(self, index):
        """Get test sample.
        
        Args:
            index (int): Sample index.
        Returns:
            dict: Sample with keys:
                - "housing_x": [window_size, F, P] torch.Tensor
                - "rental_x": [window_size] torch.Tensor
                - "housing_y": None (ground truth unavailable)
                - "rental_y": [pred_window] torch.Tensor (NaNs replaced by 0)
                - "zipcode": identifier
                - "zipcode_idx": int
                - "time": [window_size] torch.Tensor of timestamps
        """
        i, t = self.valid_indices[index]
        housing_X = self.data[i, t:t+self.window_size, :, :].astype(np.float32)
        rental_X = self.rental_data[i, t:t+self.window_size].astype(np.float32)
        
        rental_targets = []
        for p in self.prediction_window:
            target_time = t + self.window_size - 1 + p
            if target_time < self.data.shape[1]:
                rental_value = self.rental_data[i, target_time].astype(np.float32)
                rental_targets.append(torch.tensor(rental_value))
            else:
                rental_targets.append(torch.tensor(np.nan, dtype=torch.float32))
        rental_y = torch.nan_to_num(torch.stack(rental_targets), nan=0.0)
        
        time_seq = torch.tensor([tm.timestamp() for tm in self.times[t:t+self.window_size]], dtype=torch.float32)
        return {
            "housing_x": torch.from_numpy(housing_X),
            "rental_x": torch.from_numpy(rental_X),
            "housing_y": None,
            "rental_y": rental_y,
            "zipcode": self.zipcodes[i],
            "zipcode_idx": self.zipcode_idx[i],
            "time": time_seq
        }


class HousingDataModule(pl.LightningDataModule):
    """Lightning DataModule for housing & rental data.
    
    Args:
        data_filepath (str): Path to housing data (4D np.ndarray).
        zip_filepath (str): Path to zipcode data.
        time_filepath (str): Path to timestamps.
        features_filepath (str): Path to feature names.
        property_types_filepath (str): Path to property types.
        rental_filepath (str): Path to rental data (2D np.ndarray).
        window_size (int): Input window length.
        prediction_window (list): Prediction offsets.
        batch_size (int): Batch size.
        val_split (float): Validation split fraction.
        num_workers (int): Number of dataloader workers.
        prefetch_factor (int): Prefetch factor.
        pin_memory (bool): Pin memory flag.
    """
    def __init__(
        self,
        data_filepath, zip_filepath, time_filepath, features_filepath, property_types_filepath, rental_filepath,
        window_size=24,
        prediction_window=[1, 3, 12],
        batch_size=32,
        val_split=0.1,
        num_workers=4,
        prefetch_factor=4,
        pin_memory=True,
    ):
        super().__init__()
        self.data_filepath = data_filepath
        self.zip_filepath = zip_filepath
        self.time_filepath = time_filepath
        self.features_filepath = features_filepath
        self.property_types_filepath = property_types_filepath
        self.rental_filepath = rental_filepath
        self.window_size = window_size
        self.prediction_window = prediction_window
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.features = np.load(self.features_filepath)
        self.property_types = np.load(self.property_types_filepath)

    def setup(self, stage=None):
        """Setup datasets.
        
        Args:
            stage (str, optional): Stage indicator.
        Returns:
            None.
        """
        data = np.load(self.data_filepath, mmap_mode='r+')
        zipcodes = np.load(self.zip_filepath)
        zipcode_idx = np.arange(len(zipcodes))
        times = np.load(self.time_filepath, allow_pickle=True)
        rental_data = np.load(self.rental_filepath, mmap_mode='r+')

        complete_dataset = HousingWindowDataset(
            data, zipcodes, zipcode_idx, times, self.features,
            self.window_size, self.prediction_window, rental_data=rental_data
        )
        total_len = len(complete_dataset)
        val_size = int(total_len * self.val_split)
        train_size = total_len - val_size
        self.train_ds, self.val_ds = random_split(complete_dataset, [train_size, val_size])

        self.test_ds = HousingWindowTestDataset(
            data, zipcodes, zipcode_idx, times, self.features,
            self.window_size, self.prediction_window, rental_data=rental_data
        )

    def train_dataloader(self):
        """Training dataloader.
        
        Returns:
            DataLoader: Training dataloader.
        """
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 1,
            shuffle=True,
            pin_memory=self.pin_memory,
            collate_fn=custom_collate_fn,
            drop_last=False
        )

    def val_dataloader(self):
        """Validation dataloader.
        
        Returns:
            DataLoader: Validation dataloader.
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 1,
            shuffle=False,
            pin_memory=self.pin_memory,
            collate_fn=custom_collate_fn,
            drop_last=True
        )

    def test_dataloader(self):
        """Test dataloader.
        
        Returns:
            DataLoader: Test dataloader.
        """
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 1,
            shuffle=False,
            pin_memory=self.pin_memory,
            collate_fn=custom_collate_fn,
            drop_last=False
        )