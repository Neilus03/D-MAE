import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import yaml

# Load configuration
config_path = '/home/ndelafuente/Desktop/D-MAE/config/config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

class CustomDataset(Dataset):
    def __init__(self, image_dir, depth_dir, transform_image=None, transform_depth=None):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.depth_files = sorted(os.listdir(depth_dir))
        self.transform_image = transform_image
        self.transform_depth = transform_depth

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])

        image = Image.open(image_path).convert('RGB')
        depth = np.load(depth_path)
        depth = Image.fromarray(depth)

        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_depth:
            depth = self.transform_depth(depth)

        # Concatenate the depth map as the 4th channel
        image_depth = torch.cat((image, depth), dim=0)

        return image_depth

def get_dataloaders(train_image_dir, train_depth_dir, val_image_dir, val_depth_dir, batch_size): # 
    transform_image = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_depth = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = CustomDataset(train_image_dir, train_depth_dir, transform_image=transform_image, transform_depth=transform_depth)
    val_dataset = CustomDataset(val_image_dir, val_depth_dir, transform_image=transform_image, transform_depth=transform_depth)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

if __name__ == "__main__":
    train_image_dir = config['data']['train_dir'] + '/images'
    train_depth_dir = config['data']['train_dir'] + '/depth'
    val_image_dir = config['data']['val_dir'] + '/images'
    val_depth_dir = config['data']['val_dir'] + '/depth'
    batch_size = config['training']['batch_size']

    train_loader, val_loader = get_dataloaders(train_image_dir, train_depth_dir, val_image_dir, val_depth_dir, batch_size) 

    for images in train_loader:
        print(images.shape)
        break
