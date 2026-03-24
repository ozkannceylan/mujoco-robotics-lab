"""Lab 9 — Dual-camera vision encoder (NumPy implementation).

A simplified CNN that processes two camera images (wrist + head) and
produces a 1024-dim visual embedding (512 per camera). The architecture
mirrors PyTorch's nn.Module pattern for easy porting.

Architecture per camera:
    Conv2D(3, 32, 8, stride=4) -> ReLU
    Conv2D(32, 64, 4, stride=2) -> ReLU
    Conv2D(64, 64, 3, stride=1) -> ReLU
    Flatten -> Linear(*, 512) -> ReLU

Two camera streams are concatenated to produce a 1024-dim output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _he_init(fan_in: int, fan_out: int, shape: tuple) -> np.ndarray:
    """He (Kaiming) initialization for ReLU networks.

    Args:
        fan_in: Number of input units.
        fan_out: Number of output units (unused, kept for API consistency).
        shape: Shape of the weight tensor.

    Returns:
        Initialized weight array.
    """
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape).astype(np.float32) * std


class Conv2D:
    """2D convolution layer (NumPy, forward-only with gradient support).

    Implements valid-padding convolution with stride.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        fan_in = in_channels * kernel_size * kernel_size
        self.weight = _he_init(
            fan_in, out_channels,
            (out_channels, in_channels, kernel_size, kernel_size),
        )
        self.bias = np.zeros(out_channels, dtype=np.float32)

        # Gradient storage
        self.grad_weight: Optional[np.ndarray] = None
        self.grad_bias: Optional[np.ndarray] = None
        self._input_cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.

        Args:
            x: Input of shape (batch, in_channels, H, W).

        Returns:
            Output of shape (batch, out_channels, H', W').
        """
        self._input_cache = x
        batch, c_in, h_in, w_in = x.shape
        k = self.kernel_size
        s = self.stride
        h_out = (h_in - k) // s + 1
        w_out = (w_in - k) // s + 1

        # im2col approach for efficiency
        cols = np.zeros(
            (batch, c_in, k, k, h_out, w_out), dtype=x.dtype
        )
        for i in range(k):
            i_max = i + s * h_out
            for j in range(k):
                j_max = j + s * w_out
                cols[:, :, i, j, :, :] = x[:, :, i:i_max:s, j:j_max:s]

        # Reshape for matmul: (batch, h_out*w_out, c_in*k*k)
        cols_flat = cols.reshape(batch, c_in * k * k, h_out * w_out)
        # weight: (c_out, c_in*k*k)
        w_flat = self.weight.reshape(self.out_channels, -1)
        # output: (batch, c_out, h_out*w_out)
        out = np.einsum("oi,bio->boh", w_flat, cols_flat, optimize=True)
        out = out.reshape(batch, self.out_channels, h_out, w_out)
        out += self.bias[None, :, None, None]
        return out

    def parameters(self) -> list[np.ndarray]:
        """Return list of parameter arrays."""
        return [self.weight, self.bias]

    def param_count(self) -> int:
        """Total number of parameters."""
        return self.weight.size + self.bias.size


class Linear:
    """Fully connected layer (NumPy implementation)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.weight = _he_init(
            in_features, out_features, (out_features, in_features)
        )
        self.bias = np.zeros(out_features, dtype=np.float32)

        self.grad_weight: Optional[np.ndarray] = None
        self.grad_bias: Optional[np.ndarray] = None
        self._input_cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: y = x @ W^T + b.

        Args:
            x: Input of shape (batch, in_features).

        Returns:
            Output of shape (batch, out_features).
        """
        self._input_cache = x
        return x @ self.weight.T + self.bias[None, :]

    def parameters(self) -> list[np.ndarray]:
        """Return list of parameter arrays."""
        return [self.weight, self.bias]

    def param_count(self) -> int:
        """Total number of parameters."""
        return self.weight.size + self.bias.size


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function.

    Args:
        x: Input array.

    Returns:
        Element-wise max(0, x).
    """
    return np.maximum(x, 0)


class SingleCameraEncoder:
    """CNN encoder for a single camera image.

    Architecture:
        Conv2D(3, 32, 8, stride=4) -> ReLU
        Conv2D(32, 64, 4, stride=2) -> ReLU
        Conv2D(64, 64, 3, stride=1) -> ReLU
        Flatten -> Linear(flat_dim, 512) -> ReLU

    Input: (batch, 3, H, W) normalized to [0, 1]
    Output: (batch, 512)
    """

    def __init__(self, input_height: int = 120, input_width: int = 160) -> None:
        self.input_height = input_height
        self.input_width = input_width

        self.conv1 = Conv2D(3, 32, kernel_size=8, stride=4)
        self.conv2 = Conv2D(32, 64, kernel_size=4, stride=2)
        self.conv3 = Conv2D(64, 64, kernel_size=3, stride=1)

        # Compute flattened dimension after convolutions
        h = (input_height - 8) // 4 + 1
        h = (h - 4) // 2 + 1
        h = (h - 3) // 1 + 1
        w = (input_width - 8) // 4 + 1
        w = (w - 4) // 2 + 1
        w = (w - 3) // 1 + 1
        self.flat_dim = 64 * h * w

        self.fc = Linear(self.flat_dim, 512)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.

        Args:
            x: Image tensor of shape (batch, 3, H, W), float32 in [0, 1].

        Returns:
            Embedding of shape (batch, 512).
        """
        h = relu(self.conv1.forward(x))
        h = relu(self.conv2.forward(h))
        h = relu(self.conv3.forward(h))
        h = h.reshape(h.shape[0], -1)  # flatten
        h = relu(self.fc.forward(h))
        return h

    def parameters(self) -> list[np.ndarray]:
        """All trainable parameters."""
        params = []
        for layer in [self.conv1, self.conv2, self.conv3, self.fc]:
            params.extend(layer.parameters())
        return params

    def param_count(self) -> int:
        """Total parameter count."""
        return sum(
            layer.param_count()
            for layer in [self.conv1, self.conv2, self.conv3, self.fc]
        )


class DualCameraVisionEncoder:
    """Dual-camera vision encoder producing a 1024-dim visual embedding.

    Processes wrist and head camera images through separate CNN encoders,
    then concatenates the 512-dim embeddings to produce a 1024-dim output.

    This mirrors the PyTorch nn.Module pattern:
    - forward() for inference
    - parameters() to get all trainable weights
    - save() / load() for checkpointing
    """

    def __init__(
        self,
        input_height: int = 120,
        input_width: int = 160,
    ) -> None:
        """Initialize dual-camera encoder.

        Args:
            input_height: Camera image height.
            input_width: Camera image width.
        """
        self.wrist_encoder = SingleCameraEncoder(input_height, input_width)
        self.head_encoder = SingleCameraEncoder(input_height, input_width)
        self.output_dim = 1024

    def forward(
        self,
        wrist_images: np.ndarray,
        head_images: np.ndarray,
    ) -> np.ndarray:
        """Encode dual camera images into a visual embedding.

        Args:
            wrist_images: (batch, 3, H, W) float32 in [0, 1].
            head_images: (batch, 3, H, W) float32 in [0, 1].

        Returns:
            Visual embedding of shape (batch, 1024).
        """
        wrist_emb = self.wrist_encoder.forward(wrist_images)  # (B, 512)
        head_emb = self.head_encoder.forward(head_images)      # (B, 512)
        return np.concatenate([wrist_emb, head_emb], axis=1)   # (B, 1024)

    def parameters(self) -> list[np.ndarray]:
        """Return all trainable parameter arrays."""
        return self.wrist_encoder.parameters() + self.head_encoder.parameters()

    def param_count(self) -> int:
        """Total number of parameters."""
        return self.wrist_encoder.param_count() + self.head_encoder.param_count()

    def save(self, filepath: Path) -> None:
        """Save encoder parameters to npz file.

        Args:
            filepath: Destination path.
        """
        params = {f"param_{i}": p for i, p in enumerate(self.parameters())}
        np.savez(filepath, **params)

    def load(self, filepath: Path) -> None:
        """Load encoder parameters from npz file.

        Args:
            filepath: Source path.
        """
        data = np.load(filepath)
        params = self.parameters()
        for i, p in enumerate(params):
            loaded = data[f"param_{i}"]
            p[:] = loaded


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Convert HWC uint8 image to CHW float32 in [0, 1].

    Args:
        image: (H, W, 3) or (batch, H, W, 3) uint8 image.

    Returns:
        (3, H, W) or (batch, 3, H, W) float32 in [0, 1].
    """
    x = image.astype(np.float32) / 255.0
    if x.ndim == 3:
        return np.transpose(x, (2, 0, 1))
    elif x.ndim == 4:
        return np.transpose(x, (0, 3, 1, 2))
    else:
        raise ValueError(f"Expected 3D or 4D image, got {x.ndim}D")
