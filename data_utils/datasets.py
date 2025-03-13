from monai.transforms import Compose, SpatialPad
from torch.utils.data import Dataset, DataLoader
from hydra.utils import instantiate
from PIL import Image
import pandas as pd
import numpy as np
import torch
import cv2
import os


class DynamicSquarePad:
    def __call__(self, img):
        # Find the largest spatial dimension
        spatial_shape = img.shape[1:]  # Assuming channel-first data
        max_dim = max(spatial_shape)
        
        # Create a spatial pad transform with square dimensions
        pad = SpatialPad(spatial_size=[max_dim] * len(spatial_shape))
        return pad(img)


class Processor:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Initialize CLAHE processor.
        
        Args:
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
    def __call__(self, img):
        """
        Apply CLAHE to an image.
        
        Args:
            img: NumPy array in HWC format (Height, Width, Channels) or grayscale
            
        Returns:
            Processed image in the same format as input
        """
        res = self.clahe.apply(img[:, :, 0].numpy().astype(np.uint8)).T
        res = np.stack((res, res, res), axis=0).astype(float) / 255
        return torch.from_numpy(res).double()


class ImageClassificationDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.entries = os.listdir(img_dir)
        self.transform = transform
    
    
    def __len__(self):
        return len(self.entries)
    
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.entries[idx])
        erosion_score, jsn_score = os.path.splitext(self.entries[idx])[0].split("_")[-2:]

        img = self.transform(img_path)
        return img, int(erosion_score), int(jsn_score)


class EvalImageDataset(Dataset):
    def __init__(self, image_dir, bbox_csv, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.bbox_df = pd.read_csv(bbox_csv)
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for _, row in self.bbox_df.iterrows():
            patient_id = row["patient_id"]
            joint_id = row["joint_id"]
            image_name = f"{int(patient_id)}_{int(joint_id)}.jpeg"
            image_path = os.path.join(self.image_dir, image_name)
            if os.path.exists(image_path):
                samples.append((image_path, patient_id, joint_id, row["xcenter"], row["ycenter"], row["dx"], row["dy"]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, patient_id, joint_id, xcenter, ycenter, dx, dy = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "patient_id": patient_id,
            "joint_id": joint_id,
            "xcenter": xcenter,
            "ycenter": ycenter,
            "dx": dx,
            "dy": dy,
        }


def initialize_data(cfg):
    train_dataset = instantiate(cfg.train_dataset)
    val_dataset = instantiate(cfg.val_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )
    return train_loader, val_loader