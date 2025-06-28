"""
HiFi-GAN vocoder for converting mel spectrograms to audio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ResBlock1(nn.Module):
    """Residual block with dilated convolutions."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.utils.weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=(kernel_size * dilation[0] - dilation[0]) // 2,
                    )
                ),
                nn.utils.weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=(kernel_size * dilation[1] - dilation[1]) // 2,
                    )
                ),
                nn.utils.weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=(kernel_size * dilation[2] - dilation[2]) // 2,
                    )
                ),
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.utils.weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=(kernel_size - 1) // 2,
                    )
                )
                for _ in range(3)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(nn.Module):
    """Residual block with multiple dilations."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.utils.weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=(kernel_size * d - d) // 2,
                    )
                )
                for d in dilation
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for c in self.convs:
            xt = F.leaky_relu(x, 0.1)
            xt = c(xt)
            x = xt + x
        return x


class Generator(nn.Module):
    """HiFi-GAN generator."""

    def __init__(
        self,
        in_channels: int = 80,
        resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ),
        upsample_rates: Tuple[int, ...] = (8, 8, 2, 2),
        upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 4, 4),
        upsample_initial_channel: int = 512,
        resblock: str = "1",
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # Initial convolution
        self.conv_pre = nn.utils.weight_norm(
            nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)
        )

        # Upsampling layers
        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()

        # Initial upsampling
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.utils.weight_norm(
                        nn.ConvTranspose1d(
                            upsample_initial_channel // (2**i),
                            upsample_initial_channel // (2 ** (i + 1)),
                            k,
                            u,
                            padding=(k - u) // 2,
                        )
                    ),
                )
            )

            # Noise injection
            stride = upsample_rates[i]
            kernel_size = 2 * stride
            padding = (kernel_size - stride) // 2
            self.noise_convs.append(
                nn.utils.weight_norm(
                    nn.Conv1d(
                        1,
                        upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size,
                        stride,
                        padding=padding,
                    )
                )
            )

        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                if resblock == "1":
                    self.resblocks.append(ResBlock1(ch, k, d))
                elif resblock == "2":
                    self.resblocks.append(ResBlock2(ch, k, d))

        # Output convolution
        self.conv_post = nn.utils.weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for better training stability."""
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input mel spectrogram of shape [batch_size, n_mels, time]

        Returns:
            torch.Tensor: Generated waveform of shape [batch_size, 1, time * hop_length]
        """
        # Initial convolution
        x = self.conv_pre(x)

        # Upsampling + residual blocks
        for i in range(self.num_upsamples):
            # Upsample
            x = self.ups[i](x)

            # Add noise
            noise = torch.randn(x.size(0), 1, x.size(2) * self.ups[i][1].stride[0]).to(
                x.device
            )
            noise = self.noise_convs[i](noise)
            x = x + noise

            # Apply residual blocks
            for j in range(self.num_kernels):
                x = self.resblocks[i * self.num_kernels + j](x)

        # Output convolution
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class DiscriminatorP(nn.Module):
    """Period discriminator for HiFi-GAN."""

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period

        # Layers
        self.convs = nn.ModuleList(
            [
                nn.utils.weight_norm(
                    nn.Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                nn.utils.weight_norm(
                    nn.Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                nn.utils.weight_norm(
                    nn.Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                nn.utils.weight_norm(
                    nn.Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                nn.utils.weight_norm(
                    nn.Conv2d(
                        1024, 1024, (kernel_size, 1), 1, padding=(kernel_size // 2, 0)
                    )
                ),
            ]
        )

        # Output layers
        self.conv_post = nn.utils.weight_norm(
            nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input waveform of shape [batch_size, 1, time]

        Returns:
            tuple: (feature_maps, scores)
        """
        # Reshape for periodicity
        if x.size(1) == 1:
            x = x.unsqueeze(1)  # Add channel dim if needed

        # Pad the input
        if x.size(-1) % self.period != 0:
            pad = self.period - (x.size(-1) % self.period)
            x = F.pad(x, (0, pad), "reflect")

        # Reshape to [batch, 1, period, time/period]
        x = x.view(x.size(0), 1, self.period, -1)

        # Forward through conv layers
        feature_maps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feature_maps.append(x)

        # Final convolution
        scores = self.conv_post(x)
        feature_maps.append(scores)

        # Flatten
        scores = torch.flatten(scores, 1, -1)

        return scores, feature_maps


class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator for HiFi-GAN."""

    def __init__(self, periods: Tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorP(period) for period in periods]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input waveform of shape [batch_size, 1, time]

        Returns:
            tuple: (scores, feature_maps)
        """
        scores = []
        feature_maps = []

        for disc in self.discriminators:
            score, fm = disc(x)
            scores.append(score)
            feature_maps.extend(fm)

        return torch.cat(scores, dim=1), feature_maps


class HiFiGAN(nn.Module):
    """HiFi-GAN vocoder."""

    def __init__(
        self,
        in_channels: int = 80,
        resblock: str = "1",
        upsample_rates: Tuple[int, ...] = (8, 8, 2, 2),
        upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 4, 4),
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ),
        periods: Tuple[int, ...] = (2, 3, 5, 7, 11),
    ):
        super().__init__()

        # Generator
        self.generator = Generator(
            in_channels=in_channels,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            upsample_initial_channel=upsample_initial_channel,
            resblock=resblock,
        )

        # Multi-period discriminator
        self.mpd = MultiPeriodDiscriminator(periods=periods)

        # Store device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate waveform from mel spectrogram.

        Args:
            mel: Input mel spectrogram of shape [batch_size, n_mels, time]

        Returns:
            torch.Tensor: Generated waveform of shape [batch_size, 1, time * hop_length]
        """
        return self.generator(mel)

    @torch.no_grad()
    def generate(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate waveform from mel spectrogram (inference mode).

        Args:
            mel: Input mel spectrogram of shape [batch_size, n_mels, time] or [n_mels, time]

        Returns:
            torch.Tensor: Generated waveform of shape [batch_size, 1, time * hop_length] or [1, time * hop_length]
        """
        self.generator.eval()

        # Add batch dimension if needed
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        # Move to device if not already
        if mel.device != self.device:
            mel = mel.to(self.device)

        # Generate
        with torch.no_grad():
            audio = self.generator(mel)

        return audio.squeeze(0) if audio.size(0) == 1 else audio

    def to(self, device):
        """Move model to device and store device."""
        self.device = device
        self.generator = self.generator.to(device)
        self.mpd = self.mpd.to(device)
        return self


# For testing
if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = HiFiGAN().to(device)

    # Test forward pass
    mel = torch.randn(4, 80, 100).to(device)
    audio = model(mel)
    print(f"Generated audio shape: {audio.shape}")

    # Test discriminator
    scores, _ = model.mpd(audio)
    print(f"Discriminator scores shape: {scores.shape}")

    # Test generation
    with torch.no_grad():
        generated = model.generate(mel[0])
        print(f"Generated single audio shape: {generated.shape}")
