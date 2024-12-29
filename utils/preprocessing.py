import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class ReshapeAndScale:
    """
    Custom transform to reshape a flat pixel string into a 2D array, 
    scale it to [0, 1], and convert it to a PyTorch tensor.
    """

    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def __call__(self, input_data):
        # If input is a string, process as pixel string
        if isinstance(input_data, str):
            pixel_vals = np.array([float(val) for val in input_data.split()], dtype=np.float32)
            image = pixel_vals.reshape(self.n_rows, self.n_cols) / 255.0
            return torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        # If input is a PIL image, convert to tensor
        elif isinstance(input_data, Image.Image):
            return transforms.ToTensor()(input_data)

        raise TypeError(f"Unsupported input type: {type(input_data)}")


def load_data(data_path):
    """
    Load the data from the specified path.

    Parameters
    ----------
    data_path : str
        The path to the data file.

    Returns
    -------
    DataFrame
        The loaded data.
    """

    data = pd.read_csv(data_path)

    return data


def oversample_data(x_data, y_data):
    """
    Oversample the minority class in the data using RandomOverSampler.

    Parameters
    ----------
    data : DataFrame
        The data to be oversampled.

    Returns
    -------
    DataFrame
        The oversampled data.
    """
    oversampler = RandomOverSampler(sampling_strategy='auto')

    x_data, y_data = oversampler.fit_resample(x_data.values.reshape(-1,1), y_data)

    # Convert oversampled data back to a DataFrame for compatibility with the dataset
    x_data = pd.DataFrame(x_data, columns=["pixels"])
    y_data = pd.Series(y_data, name="emotion")
    oversampled_data = pd.concat([x_data, y_data], axis=1)

    # Shuffle the oversampled dataset to mix the examples
    oversampled_data = shuffle(oversampled_data)

    return oversampled_data

def split_data(df, train_split=0.8):
    """
    Split the data into training and validation sets using train_test_split.

    Parameters
    ----------
    df : pd.DataFrame
        The data to split, should contain 'pixels' and 'emotion' columns.
    train_split : float
        The proportion of the data to use for training. Default is 0.8.

    Returns
    -------
    tuple
        A tuple containing the training and validation sets as DataFrames.
    """

    # Split the DataFrame into train and validation sets
    train_df, val_df = train_test_split(df, train_size=train_split, stratify=df['emotion'], random_state=42)

    return train_df, val_df


def create_dataloaders(train_dataset, val_dataset, batch_size):
    """
    Creates DataLoaders.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to split and create loaders for.
    batch_size : int
        Batch size for the DataLoaders.
    train_split : float
        Proportion of the dataset to use for training. Default is 0.8.

    Returns
    -------
    tuple
        A tuple containing the training and validation DataLoaders.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

