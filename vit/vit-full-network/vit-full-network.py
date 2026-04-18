import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def layer_norm(x, eps=1e-6):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

class VisionTransformer:
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 num_classes: int = 1000, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0,
                 W_patch=None, cls_token=None, pos_embed=None,
                 encoder_weights=None, W_head=None):
        """
        Initialize Vision Transformer. If weight arrays are provided, use them;
        otherwise initialize randomly.
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        
        # Patch Projection Initialization
        patch_dim = (patch_size ** 2) * 3
        self.W_patch = W_patch if W_patch is not None else np.random.randn(patch_dim, embed_dim) * 0.02

        # [CLS] Token Initialization
        self.cls_token = cls_token if cls_token is not None else np.random.randn(1, 1, embed_dim) * 0.02

        # Position Embeddings Initialization
        self.pos_embed = pos_embed if pos_embed is not None else np.random.randn(1, self.num_patches + 1, embed_dim) * 0.02

        # Encoder Weights
        if encoder_weights is not None:
            self.layers = encoder_weights
        else:
            self.layers = []
            hidden_dim = int(embed_dim * mlp_ratio)
            for _ in range(depth):
                self.layers.append({
                    'Wq': np.random.randn(embed_dim, embed_dim) * 0.02,
                    'Wk': np.random.randn(embed_dim, embed_dim) * 0.02,
                    'Wv': np.random.randn(embed_dim, embed_dim) * 0.02,
                    'Wo': np.random.randn(embed_dim, embed_dim) * 0.02,
                    'W1': np.random.randn(embed_dim, hidden_dim) * 0.02,
                    'W2': np.random.randn(hidden_dim, embed_dim) * 0.02,
                })

        # Classification Head Initialization
        self.W_head = W_head if W_head is not None else np.random.randn(embed_dim, num_classes) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        """
        B, H, W, C = x.shape
        P, D, h = self.patch_size, self.embed_dim, self.num_heads
        d_k = D // h

        # Reshape to (B, n_h, P, n_w, P, C) -> Transpose -> Flatten to (B, N, patch_dim)
        n_h, n_w = H // P, W // P
        patches = x.reshape(B, n_h, P, n_w, P, C).transpose(0, 1, 3, 2, 4, 5).reshape(B, self.num_patches, -1)
        z = patches @ self.W_patch

        # Prepend [CLS] token
        cls_tokens = np.tile(self.cls_token, (B, 1, 1))
        z = np.concatenate([cls_tokens, z], axis=1) # (B, N+1, D)

        # Add Pos Embeddings
        z = z + self.pos_embed

        for layer in self.layers:
            res_attn = z
            z_norm = layer_norm(z)

            q = (z_norm @ layer['Wq']).reshape(B, -1, h, d_k).transpose(0, 2, 1, 3)
            k = (z_norm @ layer['Wk']).reshape(B, -1, h, d_k).transpose(0, 2, 1, 3)
            v = (z_norm @ layer['Wv']).reshape(B, -1, h, d_k).transpose(0, 2, 1, 3)

            scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
            attn_weights = softmax(scores, axis=-1)

            # Merge heads: (B, N+1, D)
            msa_out = (attn_weights @ v).transpose(0, 2, 1, 3).reshape(B, -1, D)
            z = res_attn + (msa_out @ layer['Wo'])

            res_mlp = z
            z_norm = layer_norm(z)
            mlp_out = gelu(z_norm @ layer['W1']) @ layer['W2']
            z = res_mlp + mlp_out

        cls_final = layer_norm(z[:, 0, :]) 
        logits = cls_final @ self.W_head

        return logits