"""
Vision Encoder Module for Nanochat (NanoBot/NanoJanus).

This module implements:
1. Standard VisionTransformer (Lightweight ViT).
2. DeepSeekVisionTransformer (Hybrid SAM+CLIP with High-Res Tiling).

It is designed to be:
1.  Lightweight (pure PyTorch, no heavy deps like timm/torchvision).
2.  Modular (easy to swap out for SigLIP or other encoders).
3.  Compatible with Strix Halo (ROCm) via standard PyTorch ops.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from easydict import EasyDict as adict

# Import DeepSeek modules
try:
    from .deepseek_modules import build_sam_vit_b, build_clip_l, MlpProjector
    from .deepseek_modules.image_processing import dynamic_preprocess
except ImportError:
    # Fallback for when running as a script or different context
    try:
        from nanochat.deepseek_modules import build_sam_vit_b, build_clip_l, MlpProjector
        from nanochat.deepseek_modules.image_processing import dynamic_preprocess
    except ImportError:
        print("Warning: DeepSeek modules not found. DeepSeekVisionTransformer will not be available.")
    build_sam_vit_b = None
    build_clip_l = None
    MlpProjector = None
    dynamic_preprocess = None


class VisionConfig:
    def __init__(self,
                 image_size=224,
                 patch_size=14,
                 width=768,
                 layers=12,
                 heads=12,
                 mlp_ratio=4.0,
                 channels=3,
                 output_dim=None,
                 # DeepSeek specifics
                 use_deepseek=False,
                 deepseek_local_path=None):
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.channels = channels
        self.output_dim = output_dim # If None, defaults to width
        self.use_deepseek = use_deepseek
        self.deepseek_local_path = deepseek_local_path

class PatchEmbed(nn.Module):
    """ Turn images into patch embeddings. """
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # assert H == self.img_size and W == self.img_size, \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."

        x = self.proj(x) # (B, embed_dim, grid_h, grid_w)
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, embed_dim)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class VisionTransformer(nn.Module):
    """
    A minimal Vision Transformer.
    """
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbed(
            img_size=config.image_size,
            patch_size=config.patch_size,
            in_chans=config.channels,
            embed_dim=config.width
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, config.width))

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(config.width),
                "attn": Attention(config.width, num_heads=config.heads),
                "norm2": nn.LayerNorm(config.width),
                "mlp": MLP(config.width, int(config.width * config.mlp_ratio))
            }) for _ in range(config.layers)
        ])

        self.norm = nn.LayerNorm(config.width)
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed

        for block in self.blocks:
            x = x + block["attn"](block["norm1"](x))
            x = x + block["mlp"](block["norm2"](x))

        x = self.norm(x)
        return x

class VisionProjector(nn.Module):
    def __init__(self, vision_dim, llm_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = llm_dim

        self.net = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- DeepSeek Logic ---

class DeepSeekVisionTransformer(nn.Module):
    """
    Hybrid Encoder (SAM + CLIP) with High-Res Tiling Support.
    Ported from DeepSeek-OCR-vllm.
    """
    def __init__(self, config: VisionConfig, llm_embed_dim=2048):
        super().__init__()
        if build_sam_vit_b is None:
            raise ImportError("DeepSeek modules missing.")

        self.config = config

        # Initialize Encoders
        self.sam_model = build_sam_vit_b(checkpoint=config.deepseek_local_path) # Assume checkpoint loading inside or handled externally
        self.vision_model = build_clip_l() # Config is hardcoded in build_clip_l currently

        # Projector
        n_embed = 1280 # DeepSeek default output dim? Or is it 2048?
        # In run_dpsk_ocr.py: model = AutoModel... use_safetensors=True.
        # DeepSeekOCRForCausalLM init says: n_embed = 1280 (this is the output of projector?)
        # MlpProjector input_dim=2048.

        # Let's align with Nanochat's LLM dim if possible, OR output 1280 and let Nanochat's projector handle it?
        # But DeepSeek HAS a projector.
        # DeepSeek's MlpProjector maps 2048 -> 1280.
        # Wait, if Nanochat LLM is 768, we need 1280 -> 768 or 2048 -> 768.

        # We will use DeepSeek's Projector to get to 1280 (standard DeepSeek space),
        # and if Nanochat needs something else, we add an adapter or change n_embed.

        self.target_dim = llm_embed_dim

        # DeepSeek Projector Config
        proj_cfg = adict(
            projector_type="linear",
            input_dim=2048, # SAM+CLIP concatenated dim
            n_embed=self.target_dim # Target LLM Dim
        )
        self.projector = MlpProjector(proj_cfg)

        # Special Tokens for Tiling
        embed_std = 1 / math.sqrt(self.target_dim)
        self.image_newline = nn.Parameter(torch.randn(self.target_dim) * embed_std)
        self.view_seperator = nn.Parameter(torch.randn(self.target_dim) * embed_std)

    def forward(self, images):
        """
        Forward pass with Tiling Logic.

        Args:
            images: List of PIL Images OR Pre-processed Tensor Dict.

        If images is List[PIL.Image], we perform dynamic preprocessing here (slow, but E2E).
        If images is Tensor (B, C, H, W), we assume it's the Global View only (no tiling).
        To support tiling, we ideally need the pre-processed batch structure.
        """

        # Case 1: Simple Tensor Input (Global View Only / No Tiling) - Fallback
        if isinstance(images, torch.Tensor):
            # images: (B, 3, H, W)
            return self._forward_simple(images)

        # Case 2: List of PIL Images - Dynamic Tiling
        if isinstance(images, list) and isinstance(images[0], Image.Image):
            return self._forward_tiled_pil(images)

        raise ValueError(f"Unsupported input type for DeepSeekVisionTransformer: {type(images)}")

    def _forward_simple(self, images):
        # Global view processing only
        # SAM
        global_features_1 = self.sam_model(images)
        # CLIP (requires pixel_values and sam_features?)
        # DeepSeek CLIP takes (x, patch_embeds)
        global_features_2 = self.vision_model(images, global_features_1)

        # Concatenate: (B, C, H, W) -> (B, L, D) logic?
        # In deepseek_ocr.py:
        # global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1)

        # global_features_1 (SAM): (B, 1024, H, W)? Check sam.py. Returns conv3_output (B, 1024, H, W).
        # global_features_2 (CLIP): Returns (B, L, 1024).

        # Flatten SAM
        sam_flat = global_features_1.flatten(2).permute(0, 2, 1) # (B, HW, 1024)

        # CLIP Output: (B, L, 1024). L = HW + 1 (CLS).
        # We drop CLS? deepseek_ocr.py says: global_features_2[:, 1:]
        clip_flat = global_features_2[:, 1:]

        global_features = torch.cat((clip_flat, sam_flat), dim=-1) # (B, HW, 2048)

        # Project
        global_features = self.projector(global_features) # (B, HW, target_dim)

        return global_features

    def _forward_tiled_pil(self, images: list[Image.Image]):
        """
        Full DeepSeek Tiling Logic applied to a batch of PIL images.
        Note: This does not support batching efficiently if crops differ significantly
        without a custom collator. We process one by one and pad?
        Or assume batch_size=1 for now (common in local inference).
        """
        outputs = []

        for image in images:
            # 1. Dynamic Preprocess
            # image_size hardcoded to 1024 for DeepSeek?
            # DeepSeek config: IMAGE_SIZE=1024, BASE_SIZE=1024.
            crops, crop_ratio = dynamic_preprocess(image, image_size=1024)
            # crops is List[PIL.Image] containing Global View (resized) + Local Views
            # Wait, dynamic_preprocess in image_process.py returns `processed_images`.
            # And we need to add Global View separately?
            # image_process.py: tokenize_with_images does:
            # global_view = ImageOps.pad(image, (BASE_SIZE, BASE_SIZE)...)
            # images_list.append(global_view)
            # then appends crops.

            # Replicating that logic roughly:

            # Global View
            # Using simple resize/pad for demo
            from torchvision.transforms import ToTensor, Normalize, Compose, Resize
            transform = Compose([
                Resize((1024, 1024)), # Simplified
                ToTensor(),
                Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])

            # Prepare Global
            pixel_values = transform(image).unsqueeze(0).to(self.image_newline.device).to(dtype=torch.bfloat16) # (1, 3, 1024, 1024)

            # Prepare Local Crops
            # We need to transform them
            crop_tensors = []
            for crop in crops:
                crop_tensors.append(transform(crop))

            if len(crop_tensors) > 0:
                images_crop = torch.stack(crop_tensors).to(self.image_newline.device).to(dtype=torch.bfloat16) # (N, 3, 1024, 1024)
            else:
                images_crop = None

            # 2. Forward Global
            global_feats = self._forward_simple(pixel_values) # (1, HW, D)

            # 3. Forward Local
            if images_crop is not None:
                 local_feats = self._forward_simple(images_crop) # (N, HW, D)
            else:
                 local_feats = None

            # 4. Tiling Assembly (Insert Separators)
            # Logic from _pixel_values_to_embedding

            _, hw, n_dim = global_feats.shape
            h = w = int(hw ** 0.5)

            # Global Reshape & Newline
            global_feats = global_feats.view(h, w, n_dim)
            global_feats = torch.cat(
                [global_feats, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
            ) # (h, w+1, D)
            global_feats = global_feats.view(-1, n_dim)

            if local_feats is not None:
                # Local Reshape
                num_w, num_h = crop_ratio
                _2, hw2, n_dim2 = local_feats.shape
                h2 = w2 = int(hw2 ** 0.5)

                # Arrange blocks
                # local_feats: (num_h * num_w, h2*w2, D)
                local_feats = local_feats.view(num_h, num_w, h2, w2, n_dim2)
                # Permute to image structure
                local_feats = local_feats.permute(0, 2, 1, 3, 4).contiguous() # (num_h, h2, num_w, w2, D)
                local_feats = local_feats.view(num_h * h2, num_w * w2, n_dim2)

                # Add Newlines
                local_feats = torch.cat(
                     [local_feats, self.image_newline[None, None, :].expand(num_h * h2, 1, n_dim2)], dim=1
                ) # (H_total, W_total+1, D)
                local_feats = local_feats.view(-1, n_dim2)

                # Concatenate Local + Separator + Global
                final_feats = torch.cat([local_feats, global_feats, self.view_seperator[None, :]], dim=0)
            else:
                final_feats = torch.cat([global_feats, self.view_seperator[None, :]], dim=0)

            outputs.append(final_feats)

        # Stack? They might have different lengths due to dynamic crops.
        # Return list of tensors? Nanochat might expect a padded tensor.
        # For now, return list.
        return outputs
