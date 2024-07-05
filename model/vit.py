import torch
import torch.nn as nn
import numpy as np


#Creating the PatchEmbedding class
class PatchEmbedding(nn.Module):
    '''
    Patch Embedding class:
        This class is used to convert the image into patches and then flatten them to
        feed into the transformer model
    '''
    def __init__(self, d_model, img_size, patch_size, n_channels):
        '''
        __init__ function:
            This function is used to initialize the PatchEmbedding class
        Args:
            d_model: Model dimension (number of neurons in the hidden layer)s
            img_size: Image size 
            patch_size: Patch size
            n_channels: Number of channels in the image (1 for mnist as it is grayscale)
        '''
        super().__init__()
        self.d_model = d_model 
        self.img_size = img_size
        self.patch_size = patch_size 
        self.n_channels = n_channels 
        
        #creating the patch embedding layer, which is a convolutional layer that converts the image into patches by sliding a window over the image
        self.linear_projection = nn.Conv2d(in_channels=self.n_channels, out_channels=self.d_model, kernel_size=self.patch_size, stride=self.patch_size)
        
    def forward(self, x):
        '''
        forward function:
            This function is used to pass the input through the patch embedding layer
        Args:
            x: Input image (B, C, H, W) #B: Batch size, C: Number of channels, H: Height, W: Width
        Returns:
            x: Output of the patch embedding layer (B, d_model, P_col, P_row) #d_model: Model dimension, P_col: Patch column, P_row: Patch row
        '''
        #projection of the image into patches
        x = self.linear_projection(x) # (B, C, H, W) -> (B, d_model, P_col, P_row) 
        #flatten the patches
        x = x.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P_col*P_row) 
        #rearrange the patches to (B, P_col*P_row, d_model)
        x = x.transpose(1, 2) # (B, d_model, P_col*P_row) -> (B, P_col*P_row, d_model)
        #return the new tensor
        return x  #(B, P_col*P_row, d_model)
        

#Creating the PositionalEncoding class
class PositionalEncoding(nn.Module):
    '''
    Positional Encoding class:
        This class is used to add positional encoding to the patches
    '''
    def __init__(self, d_model, max_seq_len, include_cls_token=False):
        '''
        __init__ function:
            This function is used to initialize the PositionalEncoding class
        Args:
            d_model: Model dimension (number of neurons in the hidden layer)
            max_seq_len: Maximum sequence length
        '''
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.include_cls_token = include_cls_token #control whether to include the cls token or not
        
        if include_cls_token:
            #cls token to be added to the patches
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
            max_seq_len += 1 #add 1 to the maximum sequence length to include the cls token
        
        #positional encoding matrix
        pe = torch.zeros(self.max_seq_len, self.d_model) #initialize the positional encoding matrix with zeros
        
        for pos in range(self.max_seq_len):
            for i in range(self.d_model):
                #positional encoding with sine and cosine functions
                if i % 2 == 0:
                    #even index, we update the even indices of the pe matrix with sine function
                    pe[pos, i] = np.sin(pos / 10000 ** (i / self.d_model)) 
                else:
                    #odd index, we update the odd indices of the pe matrix with cosine function
                    pe[pos, i] = np.cos(pos / 10000 ** ((i-1) / self.d_model)) 
       
        #register the positional encoding matrix as a buffer (not a parameter), this is because we don't want to update the positional encoding matrix during training
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_seq_len, d_model)
        
    def forward(self, x):
        '''
        forward function:
            This function is used to add positional encoding to the patches
        Args:
            x: Input patches (B, P_col*P_row, d_model)
        Returns:
            x: Output of the positional encoding layer (B, P_col*P_row+1, d_model)
        '''
        
        #if the cls token is included, add the cls token to the patches
        if self.include_cls_token:
            #add the cls token to the patches by expanding it to the batch size, create class tokens for every image in the batch.
            tokens_batch = self.cls_token.expand(x.size(0), -1, -1) # (B, 1, d_model)
            #concatenate the cls token at the beginning of each the patch embeddings
            x = torch.cat((tokens_batch, x), dim=1)

        #add the positional encoding before outputting the patches tensor
        x = x + self.pe 
  
        #return the new tensor
        return x #(B, (P_col*P_row)+1, d_model)

#Creating the AttentionHead class
class AttentionHead(nn.Module):
    '''
    Attention Head class:
        This class is used to perform multi-head self-attention
    '''
    def __init__(self, d_model, head_size):
        '''
        __init__ function:
            This function is used to initialize the AttentionHead class
        Args:
            d_model: Model dimension (number of neurons in the hidden layer)
            head_size: Model dimension // number of heads 
        '''
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        
        #query, key, value linear layers 
        self.query = nn.Linear(self.d_model, self.d_model)
        self.key = nn.Linear(self.d_model, self.d_model)
        self.value = nn.Linear(self.d_model, self.d_model)
    
    def forward(self, x):
        '''
        forward function:
            This function is used to perform multi-head self-attention
        Args:
            x: Input patches (B, P_col*P_row+1, d_model)
        Returns:
            x: Output of the multi-head self-attention layer (B, P_col*P_row+1, d_model)
        '''
        #query, key, value matrices
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        #scaled dot-product attention dot(Q, K) / sqrt(d_k)
        attention = Q @ K.transpose(-2, -1) / np.sqrt(self.head_size)
        
        #softmax function
        attention = torch.softmax(attention, dim=-1) #getting attention map
        
        #dot product of "softmaxed" attention and value matrices
        attention = attention @ V
        
        #return the new tensor
        return attention #(B, P_col*P_row+1, d_model)
 
   
#Creating the MultiHeadAttention class
class MultiHeadAttention(nn.Module):
    '''
    Multi-Head Attention class:
        This class is used to perform multi-head self-attention
    '''
    def __init__(self, d_model, n_heads):
        '''
        __init__ function:
            This function is used to initialize the MultiHeadAttention class
        Args:
            d_model: Model dimension (number of neurons in the hidden layer)
            n_heads: Number of heads
        '''
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        #head size is the model dimension divided by the number of heads
        self.head_size = self.d_model // self.n_heads
        
        #attention heads, using nn.ModuleList to store the attention heads in a list
        self.attention_heads = nn.ModuleList([AttentionHead(self.d_model, self.head_size) for _ in range(self.n_heads)])
        
        #linear layer to project the concatenated attention heads
        self.linear = nn.Linear(self.d_model*n_heads, self.d_model) # (B, P_col*P_row+1, d_model*n_heads) -> (B, P_col*P_row+1, d_model)
        
        #dropout layer
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        '''
        forward function:
            This function is used to combine the attention heads 
        Args:
            x: Input patches (B, P_col*P_row+1, d_model)
        '''
        #combine the attention heads by concatenating them along the last dimension (d_model)
        attention_heads = torch.cat([head(x) for head in self.attention_heads], dim=-1) # (B, P_col*P_row+1, d_model)
 
        #project the concatenated attention heads
        x = self.linear(attention_heads)
        
        #apply dropout
        x = self.dropout(x)
        
        #return the new tensor
        return x
  

#Creating the TransformerEncoder class
class TransformerEncoder(nn.Module):
    '''
    Transformer Encoder class:
        This class is used to encode the patches using the transformer encoder
    '''
    def __init__(self, d_model, n_heads, n_layers=4):
        '''
        __init__ function:
            This function is used to initialize the TransformerEncoder class
        Args:
            d_model: Model dimension (number of neurons in the hidden layer)
            n_heads: Number of heads
            n_layers: Number of layers
        '''
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # SubLayer 1 normalization, used to normalize the output of the multi-head attention layer
        self.ln1 = nn.LayerNorm(d_model)
        
        #Multi-head attention layer
        self.multi_head_attention = MultiHeadAttention(d_model, n_heads)
        
        #SubLayer 2 normalization, used to normalize the output of the mlp layer
        self.ln2 = nn.LayerNorm(d_model)
        
        #MLP layer
        self.mlp = nn.Sequential(
            #Linear layer 1
            nn.Linear(d_model, n_layers*d_model), # (B, P_col*P_row+1, d_model) -> (B, P_col*P_row+1, n_layers*d_model)
            #ReLU activation function (could use GELU as well, im not too worried about the non-differentiability at 0)
            nn.ReLU(), 
            #Linear layer 2
            nn.Linear(n_layers*d_model, d_model)  # (B, P_col*P_row+1, n_layers*d_model) -> (B, P_col*P_row+1, d_model)
        )
        
    def forward(self, x):
        '''
        forward function:
            This function is used to encode the patches using the transformer encoder
        Args:
            x: Input patches (B, P_col*P_row+1, d_model)
        Returns:
            x: Output of the transformer encoder (B, P_col*P_row+1, d_model)
        '''
        #Store the input tensor for residual connection
        x_res = x
        #Layer normalization after sublayer 1
        x = self.ln1(x)
        #Multi-head attention layer
        x = self.multi_head_attention(x)
        #Add the residual connection after sublayer 1
        x += x_res
        
        #This could be synthesized to the following line:
        #out = x + self.multi_head_attention(self.ln1(x))
        
        #Store the input tensor for residual connection
        x_res = x
        #Layer normalization after sublayer 2
        x = self.ln2(x)
        #MLP layer
        x = self.mlp(x)
        #Add the residual connection after sublayer 2
        x += x_res
        
        #This could be synthesized to the following line:
        #out = out + self.mlp(self.ln2(x))
        
        #return the new tensor
        return x

#Creating the VisionTransformer class
class ViTFeatureExtractor(nn.Module):
    '''
    ViTFeatureExtractor class:
        This class is used to build the vision transformer model ready to extract features
    '''
    def __init__(self, d_model, img_size, patch_size, n_channels, n_heads, n_layers, include_cls_token=False):
        '''
        __init__ function:
            This function is used to initialize the ViTFeatureExtractor class
        Args:
            d_model: Model dimension (number of neurons in the hidden layer)
            img_size: Image size
            patch_size: Patch size
            n_channels: Number of channels in the image
            n_heads: Number of heads
            n_layers: Number of layers
        '''
        super().__init__()
        
        #image size must be divisible by the patch size (patch size is the size of the window that slides over the image)
        assert img_size[0] % patch_size[0] == 0
        assert img_size[1] % patch_size[1] == 0
        
        self.d_model = d_model
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        #number of patches in the image
        self.n_patches = (img_size[0] * img_size[1]) // (patch_size[0] * patch_size[1])
        
        #set max sequence length to the number of patches + 1 (cls token) if cls token is included
        if include_cls_token:
            self.max_seq_len = self.n_patches + 1
        else:
            self.max_seq_len = self.n_patches
        #patch embedding layer
        self.patch_embedding = PatchEmbedding(self.d_model, img_size, patch_size, n_channels)
        
        #positional encoding layer
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_seq_len)
        
        #transformer encoder layers (n transformer encoder layers)
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoder(self.d_model, self.n_heads, self.n_layers) for _ in range(self.n_layers)]
            )
        
    def forward(self, x):
        '''
        forward function:
            This function is used to pass the input through the vision transformer model
        Args:
            x: Input image (B, C, H, W)
        Returns:
            x: Output of the vision transformer model (B, n_classes)
        '''
        
        print(f"ViTFeatureExtractor input shape: {x.shape}")
        
        #patch embedding layer
        x = self.patch_embedding(x)
        
        print(f"PatchEmbedding output shape: {x.shape}")
        
        #positional encoding layer
        x = self.positional_encoding(x)
        
        print(f"PositionalEncoding output shape: {x.shape}")
        
        #transformer encoder layers
        x = self.transformer_encoder(x)
        
        print(f"TransformerEncoder output shape: {x.shape}")
        
        #return the new tensor
        return x
  
  
#Checking the ViTFeatureExtractor class
if __name__ == "__main__":
    #creating a random image tensor
    x = torch.randn(1, 3, 224, 224)
    #creating the ViTFeatureExtractor model
    vit_extractor = ViTFeatureExtractor(d_model=512, img_size=(224, 224), patch_size=(16, 16), n_channels=3, n_heads=8, n_layers=8, include_cls_token=False)
    #passing the image through the model
    output = vit_extractor(x)
    #printing the output shape
    print(f"ViTFeatureExtractor output shape: {output.shape}")
