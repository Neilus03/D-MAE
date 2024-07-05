import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# Import custom modules
from data.dataloader import get_dataloaders
from model.dmae import MAE, mae_loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Load configuration
config_path = '/home/ndelafuente/Desktop/D-MAE/config/config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

def visualize_reconstruction(original, reconstructed, mask, batch_idx):
    """
    Visualize original, reconstructed, and masked images for logging to wandb.
    """
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Reshape mask to match original image dimensions
    B, C, H, W = original.shape
    P = int(H / 16)  # Assuming 16x16 patches
    mask_reshaped = mask.reshape(B, P, P).unsqueeze(1).repeat(1, C, 16, 16)
    mask_reshaped = mask_reshaped.reshape(B, C, H, W)
    
    masked = original * (1 - mask_reshaped)
    
    for i in range(4):
        # Original
        axes[0, i].imshow(original[i, :3].permute(1, 2, 0).cpu().detach().numpy())  
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')
        
        # Reconstructed
        recon_img = reconstructed[i].reshape(P, P, 16, 16, C).permute(0, 2, 1, 3, 4).reshape(H, W, C)
        axes[1, i].imshow(recon_img[:, :, :3].cpu().detach().numpy())
        axes[1, i].set_title(f"Reconstructed {i+1}")
        axes[1, i].axis('off')
        
        # Masked
        axes[2, i].imshow(masked[i, :3].permute(1, 2, 0).cpu().detach().numpy())
        axes[2, i].set_title(f"Masked {i+1}")
        axes[2, i].axis('off')
    
    plt.tight_layout()
    return fig


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0
    total_rgb_loss = 0
    total_depth_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        pred, mask = model(batch)
        loss, rgb_loss, depth_loss = mae_loss(pred, batch, mask)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_rgb_loss += rgb_loss.item()
        total_depth_loss += depth_loss.item()
        
        if batch_idx % 100 == 0:
            wandb.log({
                "train_loss": loss.item(),
                "train_rgb_loss": rgb_loss.item(),
                "train_depth_loss": depth_loss.item(),
                "train_reconstructions": visualize_reconstruction(batch, pred, mask, batch_idx)
            })
    
    return total_loss / len(dataloader), total_rgb_loss / len(dataloader), total_depth_loss / len(dataloader)

def validate(model, dataloader, device):
    """
    Validate the model on the validation set.
    """
    model.eval()
    total_loss = 0
    total_rgb_loss = 0
    total_depth_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
            batch = batch.to(device)
            
            pred, mask = model(batch)
            loss, rgb_loss, depth_loss = mae_loss(pred, batch, mask)
            
            total_loss += loss.item()
            total_rgb_loss += rgb_loss.item()
            total_depth_loss += depth_loss.item()
            
            if batch_idx % 50 == 0:
                wandb.log({
                    "val_reconstructions": visualize_reconstruction(batch, pred, batch * (1 - mask.unsqueeze(-1)), batch_idx)
                })
    
    return total_loss / len(dataloader), total_rgb_loss / len(dataloader), total_depth_loss / len(dataloader)

def main():
    # Initialize wandb
    wandb.init(project=config['logging']['wandb_project'], entity=config['logging']['wandb_entity'])
    wandb.config.update(config)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get dataloaders
    train_image_dir = os.path.join(config['data']['train_dir'], 'images')
    train_depth_dir = os.path.join(config['data']['train_dir'], 'depth')
    val_image_dir = os.path.join(config['data']['val_dir'], 'images')
    val_depth_dir = os.path.join(config['data']['val_dir'], 'depth')
    
    train_loader, val_loader = get_dataloaders(
        train_image_dir, train_depth_dir, 
        val_image_dir, val_depth_dir, 
        config['training']['batch_size']
    )

    # Initialize model
    model = MAE(
        d_model=config['model']['d_model'],
        img_size=config['model']['image_size'],
        patch_size=config['model']['patch_size'],
        n_channels=config['model']['n_channels'],
        n_heads=config['model']['num_heads_encoder'],
        n_layers=config['model']['num_layers_encoder'],
        mask_ratio=config['model']['mask_ratio']
    ).to(device)

    # Print and log the number of parameters
    num_params = count_parameters(model)
    print(f"Number of trainable parameters: {num_params:,}")
    wandb.log({"num_parameters": num_params})
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Training loop
    for epoch in range(config['training']['epochs']):
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        train_loss, train_rgb_loss, train_depth_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_rgb_loss, val_depth_loss = validate(model, val_loader, device)
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_rgb_loss": train_rgb_loss,
            "train_depth_loss": train_depth_loss,
            "val_loss": val_loss,
            "val_rgb_loss": val_rgb_loss,
            "val_depth_loss": val_depth_loss
        })
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save model checkpoint
        torch.save(model.state_dict(), f"/home/ndelafuente/Desktop/D-MAE/checkpoints/dmae_epoch_{epoch+1}.pth")

    wandb.finish()

if __name__ == "__main__":
    main()
