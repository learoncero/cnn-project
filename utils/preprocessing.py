import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.utils import shuffle

class ReshapeAndScale:
    """
    Custom transform to reshape a flat pixel string into a 2D array, 
    scale it to [0, 1], and convert it to a PyTorch tensor.
    """

    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def __call__(self, pixel_string):
        # Reshape the flat pixel string into a 2D array
        pixel_vals = np.array([float(val) for val in pixel_string.split()], dtype=np.float32)
        image = pixel_vals.reshape(self.n_rows, self.n_cols)

        # Scale the image to [0, 1]
        image = image / 255.0

        # Convert to a PyTorch tensor
        return torch.tensor(image, dtype=torch.float32).unsqueeze(0) # add a channel dimension
    

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


def create_dataloaders(dataset, batch_size, train_split=0.8):
    """
    Splits the dataset into training and validation sets and creates DataLoaders.

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
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader