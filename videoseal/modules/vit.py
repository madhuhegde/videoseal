# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import LayerNorm, MLPBlock


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: tuple[int, ...] = (),
        temporal_attention: bool = False,
        max_temporal_length: int = 32,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: nn.Parameter = None
        self.pos_embed_temporal: nn.Parameter = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )
            if temporal_attention:
                self.pos_embed_temporal = nn.Parameter(
                    torch.zeros(max_temporal_length, 1, 1, embed_dim)
                )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.temp_att = temporal_attention
        if self.temp_att:
            self.temp_blocks = nn.ModuleList()
            for i in range(depth):
                block = TemporalBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    use_rel_pos=use_rel_pos,
                    video_len=max_temporal_length,
                )
                self.temp_blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm(out_chans, data_format="channels_first"),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm(out_chans, data_format="channels_first"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        if self.pos_embed_temporal is not None:
            x = x + self.pos_embed_temporal[:len(x)]

        if self.temp_att:
            for blk, tblk in zip(self.blocks, self.temp_blocks):
                x = blk(x)
                x = tblk(x)
        else:
            for blk in self.blocks:
                x = blk(x)  # -> b h/16 h/16 d

        x = self.neck(x.permute(0, 3, 1, 2).contiguous())  # b h/16 w/16 d -> b out_chans h/16 w/16

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: tuple[int, int] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class TemporalBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        use_rel_pos: bool = False,
        video_len: int = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TemporalAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            video_len=video_len,
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.mlp(self.norm2(x))
        return shortcut + x


class TemporalAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        video_len: int = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                video_len is not None
            ), "Video length must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos = nn.Parameter(torch.zeros(2 * video_len - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, H * W, nHead, B, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 1, 3, 0, 4)
        # q, k, v with shape (H * W * nHead, B, C)
        q, k, v = qkv.reshape(3, H * W * self.num_heads, B, -1).unbind(0)

        # (H * W * nHead, B, B)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos_temporal(attn, q, self.rel_pos, B, B)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(H, W, self.num_heads, B, -1).permute(3, 0, 1, 2, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: tuple[int, int] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        head_dim = C // self.num_heads
        
        # Compute qkv: (B, H*W, 3*C) -> split into q, k, v without creating 5D tensor
        qkv = self.qkv(x)  # (B, H*W, 3*C)
        # Split into q, k, v: (B, H*W, C) each
        qkv = qkv.reshape(B, H * W, 3, C)
        qkv = qkv.permute(2, 0, 1, 3)  # (3, B, H*W, C) - 4D max
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, H*W, C)
        
        # Reshape to separate heads: (B, H*W, num_heads, head_dim)
        q = q.reshape(B, H * W, self.num_heads, head_dim)
        k = k.reshape(B, H * W, self.num_heads, head_dim)
        v = v.reshape(B, H * W, self.num_heads, head_dim)
        
        # Transpose to (B, num_heads, H*W, head_dim) for easier computation
        q = q.permute(0, 2, 1, 3)  # (B, num_heads, H*W, head_dim)
        k = k.permute(0, 2, 1, 3)  # (B, num_heads, H*W, head_dim)
        v = v.permute(0, 2, 1, 3)  # (B, num_heads, H*W, head_dim)
        
        # Flatten batch and heads: (B*num_heads, H*W, head_dim)
        q_flat = q.reshape(B * self.num_heads, H * W, head_dim)
        k_flat = k.reshape(B * self.num_heads, H * W, head_dim)
        v_flat = v.reshape(B * self.num_heads, H * W, head_dim)

        attn = (q_flat * self.scale) @ k_flat.transpose(-2, -1)  # (B*num_heads, H*W, H*W)

        if self.use_rel_pos:
            # Process each head separately for relative position embeddings
            # Reshape attn from (B * num_heads, H*W, H*W) to (B, num_heads, H*W, H*W)
            attn = attn.reshape(B, self.num_heads, H * W, H * W)
            # Reshape q from (B, num_heads, H*W, head_dim) - already in this shape
            q_for_rel = q.reshape(B, self.num_heads, H * W, head_dim)
            
            # Process each head
            attn_heads = []
            for h in range(self.num_heads):
                attn_h = add_decomposed_rel_pos(
                    attn[:, h],  # (B, H*W, H*W)
                    q_for_rel[:, h],  # (B, H*W, head_dim)
                    self.rel_pos_h,
                    self.rel_pos_w,
                    (H, W),
                    (H, W)
                )
                attn_heads.append(attn_h)
            
            # Stack heads: (B, num_heads, H*W, H*W) - still 4D
            attn = torch.stack(attn_heads, dim=1)
            # Reshape back to (B * num_heads, H*W, H*W) for matmul with v
            attn = attn.reshape(B * self.num_heads, H * W, H * W)

        attn = attn.softmax(dim=-1)
        
        # x (output) calculation, avoiding 5D and 6D tensors
        # Using UVQ-style approach: break down operations into 4D-only steps
        x = (attn @ v_flat)  # (B * num_heads, H*W, head_dim)
        
        # Reshape to (B, num_heads, H*W, head_dim) - 4D
        x = x.reshape(B, self.num_heads, H * W, head_dim)
        
        # Permute heads and spatial: (B, H*W, num_heads, head_dim) - 4D
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, H*W, num_heads, head_dim)
        
        # Reshape to (B, H, W, C) - 4D
        x = x.reshape(B, H, W, C)
        
        # Apply projection using UVQ-style 4D-only approach to avoid 6D tensors
        # Flatten batch and spatial dimensions to 2D for linear layer
        # This ensures TFLite converter doesn't create intermediate 6D tensors
        B_orig, H_orig, W_orig, C_orig = x.shape
        x_flat = x.reshape(B_orig * H_orig * W_orig, C_orig)  # (B*H*W, C) - 2D
        x_flat = self.proj(x_flat)  # (B*H*W, C) - 2D, linear layer on 2D tensor
        x = x_flat.reshape(B_orig, H_orig, W_orig, C_orig)  # (B, H, W, C) - 4D

        return x


def window_partition(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    TFLite-friendly version that avoids 6D tensors.
    
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    # TFLite-friendly: Use reshape instead of view to avoid 6D tensors
    # Original: view(B, Hp//ws, ws, Wp//ws, ws, C) -> 6D
    # New approach: Use 4D operations throughout
    
    num_windows_h = Hp // window_size
    num_windows_w = Wp // window_size

    # Reshape to (B * num_windows_h, window_size, Wp, C)
    x = x.reshape(B * num_windows_h, window_size, Wp, C)

    # Permute to (B * num_windows_h, Wp, window_size, C)
    x = x.permute(0, 2, 1, 3).contiguous()

    # Reshape to (B * num_windows_h * num_windows_w, window_size, window_size, C)
    x = x.reshape(B * num_windows_h * num_windows_w, window_size, window_size, C)
    
    return x, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: tuple[int, int], hw: tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    TFLite-friendly version that avoids 5D and 6D tensors using pure 4D operations.
    
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (tuple): padded height and width (Hp, Wp).
        hw (tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    C = windows.shape[-1]
    
    num_windows_h = Hp // window_size
    num_windows_w = Wp // window_size
    B = windows.shape[0] // (num_windows_h * num_windows_w)

    # Pure 4D approach: Process windows to avoid 5D/6D tensors entirely
    # Input: (B * num_windows_h * num_windows_w, window_size, window_size, C)
    # Goal: (B, Hp, Wp, C) where Hp = num_windows_h * window_size, Wp = num_windows_w * window_size
    
    # Strategy: Use only 4D operations by processing windows in a way that never creates 5D tensors
    # The key is to use view/reshape operations that directly map to 4D shapes
    
    # Step 1: Reshape to group by batch and window rows, flattening window columns
    # From (B * num_windows_h * num_windows_w, window_size, window_size, C)
    # To (B * num_windows_h, num_windows_w * window_size, window_size, C)
    # This groups windows in each row and flattens them horizontally - all 4D
    # We achieve this by viewing the data as rows of concatenated windows
    total_windows = B * num_windows_h * num_windows_w
    x = windows.reshape(total_windows, window_size * window_size, C)  # Flatten spatial dims: 3D
    x = x.reshape(B * num_windows_h, num_windows_w, window_size * window_size, C)  # Group by rows: 4D
    x = x.reshape(B * num_windows_h, num_windows_w * window_size, window_size, C)  # Flatten window columns: 4D
    
    # Step 2: Permute to interleave window rows vertically: (B * num_windows_h, window_size, num_windows_w * window_size, C)
    x = x.permute(0, 2, 1, 3).contiguous()
    
    # Step 3: Reshape to final dimensions: (B, Hp, Wp, C)
    # Combine B * num_windows_h and window_size to get Hp = num_windows_h * window_size
    x = x.reshape(B, num_windows_h * window_size, num_windows_w * window_size, C)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: tuple[int, int],
    k_size: tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    TFLite-friendly version that avoids 5D tensors.
    
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map with shape (B, q_h * q_w, k_h * k_w).
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)  # (q_h, k_h, C)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)  # (q_w, k_w, C)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)  # (B, q_h, q_w, C)
    
    # Compute relative position biases using einsum
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)  # (B, q_h, q_w, k_h)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)  # (B, q_h, q_w, k_w)

    # TFLite-friendly: Avoid 5D tensors by using 4D operations
    # Original: attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:,:,:,:,None] + rel_w[:,:,:,None,:]
    # New approach: Expand and add in 4D space
    
    # Reshape attn from (B, q_h*q_w, k_h*k_w) to (B*q_h, q_w, k_h, k_w)
    attn = attn.reshape(B * q_h, q_w, k_h, k_w)
    
    # Reshape rel_h from (B, q_h, q_w, k_h) to (B*q_h, q_w, k_h, 1) and expand
    rel_h = rel_h.reshape(B * q_h, q_w, k_h, 1).expand(-1, -1, -1, k_w)
    
    # Reshape rel_w from (B, q_h, q_w, k_w) to (B*q_h, q_w, 1, k_w) and expand
    rel_w = rel_w.reshape(B * q_h, q_w, 1, k_w).expand(-1, -1, k_h, -1)
    
    # Add relative position biases
    attn = attn + rel_h + rel_w
    
    # Reshape back to (B, q_h*q_w, k_h*k_w)
    attn = attn.reshape(B, q_h * q_w, k_h * k_w)

    return attn


def add_decomposed_rel_pos_temporal(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos: torch.Tensor,
    q_size: int,
    k_size: int,
) -> torch.Tensor:
    R = get_rel_pos(q_size, k_size, rel_pos)
    rel = torch.einsum("bhc,hkc->bhk", q, R)
    attn = attn + rel
    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] = (16, 16),
        stride: tuple[int, int] = (16, 16),
        padding: tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (tuple): kernel size of the projection layer.
            stride (tuple): stride of the projection layer.
            padding (tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x
