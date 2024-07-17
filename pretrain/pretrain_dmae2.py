'''
Training and validation loops for the D-MAE pretraining.
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# Import custom modules
from data.dataloader import get_dataloaders, denormalize_RGB
from model.dmae import MAE, mae_loss

def count_parameters(model):
    '''
    Counts the number of trainable parameters in the given model
    Args:
        model: PyTorch model 
    Returns:
        Total number of trainable parameters in the model
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Load configuration
config_path = '/home/ndelafuente/Desktop/D-MAE/config/config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Initialize Weights (Xavier Initialization)
def initialize_weights(m):
    '''
    Initializes the weights of the model using Xavier uniform initialization for linear
    layers and Kaiming Normal initialization for convolutional layers (cool thing is that kaiming he is the author of MAE paper :D)
    Args:
        m: PyTorch module
    '''
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

# Main training function
def pretrain_dmae():
    '''
    Main function for pretraining the D-MAE model.
    '''
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize WandB
    wandb.init(project=config['logging']['wandb_project'], 
               entity=config['logging']['wandb_entity'])

    # Data Loaders
    train_loader, val_loader = get_dataloaders(config)

    # Model
    model = MAE(d_model=config['model']['d_model'],
                img_size=config['model']['image_size'],
                patch_size=config['model']['patch_size'],
                n_channels=config['model']['n_channels'],
                n_heads_encoder=config['model']['num_heads_encoder'],
                n_layers_encoder=config['model']['num_layers_encoder'],
                n_heads_decoder=config['model']['num_heads_decoder'],
                n_layers_decoder=config['model']['num_layers_decoder'],
                mask_ratio=config['training']['mask_ratio']).to(device)

    # Apply weight initialization
    model.apply(initialize_weights)

    wandb.watch(model)
    print(f"Model has {count_parameters(model):,} trainable parameters")

    # Optimizer 
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])

    # Training and validation loops
    best_val_loss = float('inf')
    for epoch in range(config['training']['epochs']):
        # --- Training --- 
        model.train()
        train_loss = 0.0
        train_loss_rgb = 0.0
        train_loss_depth = 0.0

        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")):
            data = data.to(device)
            optimizer.zero_grad()
            pred, mask, ids_restore = model(data) # Unpack all three values
            loss, loss_rgb, loss_depth = mae_loss(pred, data, mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss_rgb += loss_rgb.item()
            train_loss_depth += loss_depth.item()

            if batch_idx % 100 == 0:
                # Log to WandB
                wandb.log({"train/loss": loss.item(),
                           "train/loss_rgb": loss_rgb.item(),
                           "train/loss_depth": loss_depth.item()})

                # Visualize reconstruction (first image in batch)
                visualize_reconstruction(data[0], pred[0], mask[0], epoch, batch_idx, device)

        train_loss /= len(train_loader)
        train_loss_rgb /= len(train_loader)
        train_loss_depth /= len(train_loader)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, RGB Loss: {train_loss_rgb:.4f}, Depth Loss: {train_loss_depth:.4f}")

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_loss_rgb = 0.0
        val_loss_depth = 0.0

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(val_loader, desc=f"Validation")):
                data = data.to(device)
                pred, mask, ids_restore = model(data) # Unpack all three values
                loss, loss_rgb, loss_depth = mae_loss(pred, data, mask)
                val_loss += loss.item()
                val_loss_rgb += loss_rgb.item()
                val_loss_depth += loss_depth.item()

                if batch_idx % 100 == 0:
                    # Visualize reconstruction (first image in batch)
                    visualize_reconstruction(data[0], pred[0], mask[0], epoch, batch_idx, device, mode='validation')

        val_loss /= len(val_loader)
        val_loss_rgb /= len(val_loader)
        val_loss_depth /= len(val_loader)

        # Log to WandB
        wandb.log({"val/loss": val_loss,
                   "val/loss_rgb": val_loss_rgb,
                   "val/loss_depth": val_loss_depth})

        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, RGB Loss: {val_loss_rgb:.4f}, Depth Loss: {val_loss_depth:.4f}")

        # --- Save the model ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = os.path.join(config['logging']['model_save_dir'], 'best_model.pth') # Add a model_save_dir to your config.yaml
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path}")

def visualize_reconstruction(original, pred, mask, epoch, batch_idx, device, mode='training'):
    """Visualize original, masked, and reconstructed images."""

    # Reshape pred to image dimensions
    B, L, D = pred.shape
    H = W = int((L)**0.5) * 16 # Assuming 16x16 patches
    C = D // (16*16)  # Number of channels
    pred = pred.reshape(B, H, W, C).permute(0, 3, 1, 2)

    # Apply inverse transforms for visualization 
    original_image = denormalize_RGB(original[:3]).cpu()  # Denormalize only RGB channels
    masked_image = original_image * mask.view(1, 1, H, W).cpu()  # Apply mask to original image
    reconstructed_image = denormalize_RGB(pred[:3]).cpu() # Denormalize only RGB channels

    # Create a plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original_image.permute(1, 2, 0)) 
    axs[0].set_title('Original')
    axs[1].imshow(masked_image.permute(1, 2, 0))
    axs[1].set_title('Masked')
    axs[2].imshow(reconstructed_image.permute(1, 2, 0))
    axs[2].set_title('Reconstructed')
    plt.tight_layout()

    # Log to WandB
    wandb.log({f"images/{mode}_reconstruction_{epoch}_{batch_idx}": wandb.Image(plt)})
    plt.close(fig)

if __name__ == "__main__":
    pretrain_dmae()
