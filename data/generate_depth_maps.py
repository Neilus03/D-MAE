import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../depth_anything_v2'))

import cv2
import torch
import yaml
import numpy as np
from datasets import load_dataset
from depth_anything_v2.dpt import DepthAnythingV2

# Load configuration
with open('../config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model_configs = {
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
}

model = DepthAnythingV2(**model_configs['vitb'])
model.load_state_dict(torch.load(config['data']['depth_model_checkpoint'], map_location='cpu'))
model = model.to(DEVICE).eval()

def get_depth_map(image):
    raw_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    depth = model.infer_image(raw_img)
    return depth

def process_images(dataset, output_dir_images, output_dir_depth):
    if not os.path.exists(output_dir_images):
        os.makedirs(output_dir_images)
    if not os.path.exists(output_dir_depth):
        os.makedirs(output_dir_depth)

    for item in dataset:
        image_id = item['image_id']
        image = item['image']
        output_image_path = os.path.join(output_dir_images, f'{image_id}.jpg')
        cv2.imwrite(output_image_path, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

        # Generate and save the depth map as a numpy array (this is showed on the depth anything v2 github: https://github.com/DepthAnything/Depth-Anything-V2)
        depth_map = get_depth_map(image)
        depth_path = os.path.join(output_dir_depth, f'{image_id}.npy')
        np.save(depth_path, depth_map)

# Load the MSCOCO dataset
train_dataset = load_dataset('rafaelpadilla/coco2017', split='train')
val_dataset = load_dataset('rafaelpadilla/coco2017', split='val')

# Process training and validation images
process_images(train_dataset, os.path.join('./data/train', 'images'), os.path.join('./data/train', 'depth'))
process_images(val_dataset, os.path.join('./data/val', 'images'), os.path.join('./data/val', 'depth'))