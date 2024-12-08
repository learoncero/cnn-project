import torch
from torch.utils.data import Dataset

class Fer2013Dataset(Dataset):
    def __init__(self, data_frame, labels, transform=None):
        """
        Custom Dataset for the FER-2013 dataset.
        
        Args:
            data_frame (pd.DataFrame): DataFrame containing image pixel data.
            labels (pd.Series): Series containing labels for each image.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = data_frame
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        """
        Retrieve a single data item and its corresponding label.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: (image, label) where image is the transformed image data,
                   and label is the corresponding label tensor.
        """
        pixel_string = self.data_frame.iloc[idx]["pixels"]
        image = self.transform(pixel_string) if self.transform else pixel_string

        label = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        return image, label

    def __len__(self):
        """
        Returns:
            int: Total number of items in the dataset.
        """
        return len(self.data_frame)
