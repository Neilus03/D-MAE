import torch
import torch.nn as nn
import numpy as np
from vit import ViTFeatureExtractor, PatchEmbedding, TransformerEncoder, PositionalEncoding


class RandomMasking(nn.Module):
    '''
    Random Masking class:
    This class is used to randomly mask patches from the input
    '''
    def __init__(self, mask_ratio=0.75):
        '''
        __init__ function:
        This function is used to initialize the RandomMasking class
        Args:
            mask_ratio: Ratio of patches to be masked
        '''
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        '''
        forward function:
        This function is used to randomly mask patches
        Args:
            x: Input patches (B, L, D)
        Returns:
            x_masked: Masked patches (B, L_visible, D)
            mask: Binary mask (B, L)
            ids_restore: Indices to restore original order (B, L)
        '''
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first len_keep tokens
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

class MAEEncoder(nn.Module):
    '''
    MAE Encoder class:
    This class is used to encode the visible patches
    '''
    def __init__(self, d_model, img_size, patch_size, n_channels, n_heads, n_layers):
        '''
        __init__ function:
        This function is used to initialize the MAEEncoder class
        Args:
            d_model: Model dimension
            img_size: Image size
            patch_size: Patch size
            n_channels: Number of channels
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
        '''
        super().__init__()
        self.patch_embed = PatchEmbedding(d_model, img_size, patch_size, n_channels)
        self.positional_encoding = PositionalEncoding(d_model, (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]))
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(d_model, n_heads, n_layers) for _ in range(n_layers)])

    def forward(self, x):
        '''
        forward function:
        This function is used to encode the visible patches
        Args:
            x: Input visible patches (B, C, H, W)
        Returns:
            x: Encoded patches (B, L, D)
        '''
        print(f"MAEEncoder input shape: {x.shape}")
        x = self.patch_embed(x)
        print(f"After patch embedding: {x.shape}")
        x = self.positional_encoding(x)
        print(f"After positional encoding: {x.shape}")
        x = self.transformer_encoder(x)
        print(f"After transformer encoder: {x.shape}")
        return x

class MAEDecoder(nn.Module):
    '''
    MAE Decoder class:
    This class is used to decode the encoded patches and reconstruct the full image
    '''
    def __init__(self, d_model, num_patches, patch_size, num_channels=4, n_layers=8):
        '''
        __init__ function:
        This function is used to initialize the MAEDecoder class
        Args:
            d_model: Model dimension
            num_patches: Number of patches
            patch_size: Patch size
            num_channels: Number of channels (4 for RGB + depth)
            n_layers: Number of transformer layers
        '''
        super().__init__()
        self.decoder_embed = nn.Linear(d_model, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.decoder_blocks = nn.ModuleList([
            TransformerEncoder(d_model, n_heads=8, n_layers=8)
            for _ in range(n_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)
        self.decoder_pred = nn.Linear(d_model, patch_size[0] * patch_size[1] * num_channels)

    def forward(self, x, ids_restore):
        '''
        forward function:
        This function is used to decode the encoded patches and reconstruct the full image
        Args:
            x: Encoded patches (B, L_visible, D)
            ids_restore: Indices to restore original order (B, L)
        Returns:
            x: Reconstructed patches (B, L, patch_size*patch_size*num_channels)
        '''
        print(f"MAEDecoder input shape: {x.shape}")
        x = self.decoder_embed(x)
        print(f"MAEDecoder embedded shape: {x.shape}")
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        print(f"MAEDecoder mask tokens shape: {mask_tokens.shape}")
        x_ = torch.cat([x, mask_tokens], dim=1)
        print(f"MAEDecoder concatenated shape: {x_.shape}")
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        print(f"MAEDecoder gathered shape: {x_.shape}")

        for block in self.decoder_blocks:
            x_ = block(x_)
        x_ = self.decoder_norm(x_)
        x_ = self.decoder_pred(x_)
        print(f"MAEDecoder prediction shape: {x_.shape}")

        return x_

class MAE(nn.Module):
    '''
    MAE class:
    This class implements the full Masked Autoencoder
    '''
    def __init__(self, d_model, img_size, patch_size, n_channels=4, n_heads=16, n_layers=12, mask_ratio=0.75):
        '''
        __init__ function:
        This function is used to initialize the MAE class
        Args:
            d_model: Model dimension
            img_size: Image size
            patch_size: Patch size
            n_channels: Number of channels (4 for RGB + depth)
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            mask_ratio: Ratio of patches to be masked
        '''
        super().__init__()
        self.random_masking = RandomMasking(mask_ratio)
        self.encoder = MAEEncoder(d_model, img_size, patch_size, n_channels, n_heads, n_layers)
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.decoder = MAEDecoder(d_model, num_patches, patch_size, n_channels)

    def forward(self, x):
        '''
        forward function:
        This function implements the forward pass of the MAE
        Args:
            x: Input image (B, C, H, W)
        Returns:
            pred: Reconstructed patches (B, L, patch_size*patch_size*num_channels)
            mask: Binary mask (B, L)
        '''
        # Encode
        x = self.encoder(x)
        # Random masking
        x, mask, ids_restore = self.random_masking(x)
        # Decode patches
        pred = self.decoder(x, ids_restore)
        return pred, mask

def mae_loss(pred, target, mask):
    '''
    MAE loss function:
    This function computes the reconstruction loss for the MAE
    Args:
        pred: Reconstructed patches (B, L, patch_size*patch_size*num_channels)
        target: Original patches (B, L, patch_size*patch_size*num_channels)
        mask: Binary mask (B, L)
    Returns:
        loss: Total reconstruction loss
        loss_rgb: RGB reconstruction loss
        loss_depth: Depth reconstruction loss
    '''
    pred_rgb = pred[:, :, :-1]
    pred_depth = pred[:, :, -1:]
    target_rgb = target[:, :, :-1]
    target_depth = target[:, :, -1:]

    loss_rgb = (pred_rgb - target_rgb) ** 2
    loss_depth = (pred_depth - target_depth) ** 2

    loss_rgb = loss_rgb.mean(dim=-1)  # [N, L], mean loss per patch
    loss_depth = loss_depth.mean(dim=-1)  # [N, L], mean loss per patch

    loss_rgb = (loss_rgb * mask).sum() / mask.sum()  # Mean loss on removed patches
    loss_depth = (loss_depth * mask).sum() / mask.sum()  # Mean loss on removed patches

    loss = loss_rgb + loss_depth

    return loss, loss_rgb, loss_depth

if __name__ == "__main__":
    import yaml
    # Load configuration
    config_path = '/home/ndelafuente/Desktop/D-MAE/config/config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Mock example to test shapes
    batch_size = config['training']['batch_size']
    img_size = config['model']['image_size']
    patch_size = config['model']['patch_size']
    n_channels = config['model']['n_channels'] # RGB + depth
    d_model = config['model']['d_model']

    # Create random input tensor
    x = torch.randn(batch_size, n_channels, img_size[0], img_size[1])

    # Initialize MAE model
    mae_model = MAE(d_model, img_size, patch_size, n_channels)

    # Forward pass
    pred, mask = mae_model(x)

    # Calculate loss
    target = torch.randn_like(pred)  # Mock target
    loss, loss_rgb, loss_depth = mae_loss(pred, target, mask)

    # Print shapes for verification
    print(f"Input shape: {x.shape}")
    print(f"Prediction shape: {pred.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"RGB Loss: {loss_rgb.item():.4f}")
    print(f"Depth Loss: {loss_depth.item():.4f}")
