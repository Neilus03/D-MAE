import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import wandb
import sys
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# Import custom modules
from data.dataloader import get_dataloaders, denormalize_RGB
from model.dmae import MAE, mae_loss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config_path = '/home/ndelafuente/D-MAE/config/config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    print(config)

def initialize_weights(m):
    '''
    Initializes the weights of the model using Xavier uniform initialization for linear
    layers and Kaiming Normal initialization for convolutional layers
    '''
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

def count_parameters(model):
    '''
    Counts the number of trainable parameters in the given model
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_rgb_loss = 0
    total_depth_loss = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        optimizer.zero_grad()
        
        # Move batch to device
        batch = batch.to(device)
        
        # Forward pass
        pred, mask, ids_restore, reconstructed_image, reconstructed_depth, x_patched = model(batch)
        
        # Create target
        target = model.decoder.decoder_pred(x_patched)

        # Visualize the first batch in each epoch
        if batch_idx % 10 == 0:
            visualize_reconstruction(batch[0], reconstructed_image[0], reconstructed_depth[0], mask, epoch=0)


        # Calculate loss
        loss, rgb_loss, depth_loss = mae_loss(
            pred, 
            target,
            mask, 
            alpha=config['training']['alpha'], 
            beta=config['training']['beta']
        )
        
        # Log the batch loss to wandb
        wandb.log({"general batch_loss": loss.item()})
        wandb.log({"rgb batch_loss": rgb_loss.item()})
        wandb.log({"depth batch_loss": depth_loss.item()})
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_rgb_loss += rgb_loss.item()
        total_depth_loss += depth_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_rgb_loss = total_rgb_loss / len(dataloader)
    avg_depth_loss = total_depth_loss / len(dataloader)
    
    return avg_loss, avg_rgb_loss, avg_depth_loss


def visualize_reconstruction(original, reconstructed_image, reconstructed_depth, mask, epoch):
    '''
    Visualizes the original, masked, and reconstructed images and depth maps and logs them to WandB.
    
    Args:
        original (torch.Tensor): The original input image tensor, which includes RGB and depth channels.
        reconstructed_image (torch.Tensor): The reconstructed RGB image tensor output from the model.
        reconstructed_depth (torch.Tensor): The reconstructed depth map tensor output from the model.
        mask (torch.Tensor): The mask tensor indicating which patches were masked during training.
        epoch (int): The current epoch number, used for logging in WandB.
    '''
    # Create a 2x2 grid of subplots for displaying the images
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Check the dimensionality of the original input tensor
    # If it is 4D (batch of images), extract the first image for visualization
    if original.dim() == 4:
        original_rgb = denormalize_RGB(original[:, :3, :, :].cpu()).permute(1, 2, 0).numpy()  # Extract and denormalize RGB channels
        original_depth = original[:, 3, :, :].cpu().numpy()  # Extract the depth channel
    elif original.dim() == 3:
        # If the tensor is 3D, directly process it as a single image
        original_rgb = denormalize_RGB(original[:3, :, :].cpu()).permute(1, 2, 0).numpy()  # Extract and denormalize RGB channels
        original_depth = original[3, :, :].cpu().numpy()  # Extract the depth channel

    # Process the reconstructed images
    # Denormalize the reconstructed RGB image and move the channels to the last dimension
    reconstructed_rgb = denormalize_RGB(reconstructed_image.detach().cpu()).permute(1, 2, 0).numpy()
    
    # Squeeze the reconstructed depth map to remove unnecessary dimensions
    reconstructed_depth = reconstructed_depth.detach().cpu().squeeze().numpy()  # Squeeze depth to (224, 224)

    # Plot the original RGB image
    axs[0, 0].imshow(original_rgb)
    axs[0, 0].set_title("Original RGB Image")
    axs[0, 0].axis('off')  # Turn off the axis for better visualization

    # Plot the original depth map
    axs[1, 0].imshow(original_depth, cmap='viridis')  # Use a colormap for depth visualization
    axs[1, 0].set_title("Original Depth Map")
    axs[1, 0].axis('off')  # Turn off the axis for better visualization

    # Plot the reconstructed RGB image
    axs[0, 1].imshow(reconstructed_rgb)
    axs[0, 1].set_title("Reconstructed RGB Image")
    axs[0, 1].axis('off')  # Turn off the axis for better visualization

    # Plot the reconstructed depth map
    axs[1, 1].imshow(reconstructed_depth, cmap='viridis')  # Use a colormap for depth visualization
    axs[1, 1].set_title("Reconstructed Depth Map")
    axs[1, 1].axis('off')  # Turn off the axis for better visualization

    # Adjust layout to avoid overlapping and improve spacing
    plt.tight_layout()

    # Log the entire figure as an image to WandB for visualization and tracking
    wandb.log({f"Reconstruction at Epoch {epoch}": wandb.Image(fig)})

    # Close the figure to free up memory and avoid display issues
    plt.close(fig)

    
    
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_rgb_loss = 0
    total_depth_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            batch = batch.to(device)
            
            pred, mask, ids_restore, reconstructed_image, reconstructed_depth, x_patched = model(batch)
            
            # Create target
            target = model.decoder.decoder_pred(x_patched)
            
            loss, rgb_loss, depth_loss = mae_loss(
                pred, 
                target,
                mask, 
                alpha=config['training']['alpha'], 
                beta=config['training']['beta']
            )
            
            total_loss += loss.item()
            total_rgb_loss += rgb_loss.item()
            total_depth_loss += depth_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_rgb_loss = total_rgb_loss / len(dataloader)
    avg_depth_loss = total_depth_loss / len(dataloader)
    
    return avg_loss, avg_rgb_loss, avg_depth_loss, batch[0], reconstructed_image[0]

def main():
    # Initialize wandb
    wandb.init(project=config['logging']['wandb_project'], entity=config['logging']['wandb_entity'])
    wandb.config.update(config)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    model = MAE(
        d_model=config['model']['d_model'],
        img_size=tuple(config['model']['image_size']),
        patch_size=config['model']['patch_size'],
        n_channels=config['model']['n_channels'],
        n_heads_encoder=config['model']['num_heads_encoder'],
        n_layers_encoder=config['model']['num_layers_encoder'],
        n_heads_decoder=config['model']['num_heads_decoder'],
        n_layers_decoder=config['model']['num_layers_decoder'],
        mask_ratio=config['training']['mask_ratio']
    ).to(device)

    # Initialize weights
    model.apply(initialize_weights)

    # Count and log number of parameters
    num_params = count_parameters(model)
    logger.info(f"Number of trainable parameters: {num_params}")
    wandb.log({"num_parameters": num_params})

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(config)

    '''# Create directory for saving reconstructions
    reconstruction_dir = os.path.join(config['logging']['model_save_dir'], 'reconstructions')
    os.makedirs(reconstruction_dir, exist_ok=True)'''

    # Training loop
    for epoch in range(config['training']['epochs']):
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        # Train
        train_loss, train_rgb_loss, train_depth_loss = train_one_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss, val_rgb_loss, val_depth_loss, original_image, reconstructed_image = validate(model, val_loader, device)
        
        # Visualize reconstruction
        visualize_reconstruction(original_image, reconstructed_image, epoch+1)
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_rgb_loss": train_rgb_loss,
            "train_depth_loss": train_depth_loss,
            "val_loss": val_loss,
            "val_rgb_loss": val_rgb_loss,
            "val_depth_loss": val_depth_loss,
            #"reconstruction": wandb.Image(os.path.join(reconstruction_dir, f'reconstruction_epoch_{epoch+1}.png'))
        })
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:  # Save every 10 epochs
            checkpoint_path = os.path.join(config['logging']['model_save_dir'], f"dmae_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")

    wandb.finish()

if __name__ == "__main__":
    main()
