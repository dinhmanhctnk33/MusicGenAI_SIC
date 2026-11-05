"""Neural audio codec tu dinh nghia va huan luyen tu dau."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        kernel = 2 * stride - 1 if stride % 2 else 2 * stride
        padding = (kernel - stride) // 2
        self.net = nn.Sequential(
            ResidualBlock(in_channels, 1), ResidualBlock(in_channels, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        )

    def forward(self, x):
        return self.net(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        kernel = 2 * stride - 1 if stride % 2 else 2 * stride
        padding = (kernel - stride) // 2
        self.net = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(in_channels, out_channels, kernel, stride=stride, padding=padding),
            ResidualBlock(out_channels, 1), ResidualBlock(out_channels, 3),
        )

    def forward(self, x):
        return self.net(x)


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, dim: int, num_codebooks: int, codebook_size: int, beta: float = 0.25):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.beta = beta
        self.codebooks = nn.ModuleList([nn.Embedding(codebook_size, dim) for _ in range(num_codebooks)])
        for table in self.codebooks:
            # Khoi tao rong de cac code khong co cung nearest neighbour luc dau.
            nn.init.normal_(table.weight, mean=0.0, std=dim ** -0.5)

    def forward(self, z):
        # z: [B, D, T]
        residual = z.transpose(1, 2)
        quantized_sum = torch.zeros_like(residual)
        indices, loss = [], z.new_zeros(())
        for table in self.codebooks:
            flat = residual.reshape(-1, residual.size(-1))
            weight = table.weight
            distance = flat.square().sum(1, keepdim=True) - 2 * flat @ weight.t() + weight.square().sum(1)
            index = distance.argmin(1).view(residual.shape[:-1])
            quantized = table(index)
            loss = loss + F.mse_loss(quantized, residual.detach()) + self.beta * F.mse_loss(residual, quantized.detach())
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized.detach()
            indices.append(index)
        straight_through = z + (quantized_sum.transpose(1, 2) - z).detach()
        return straight_through, torch.stack(indices, dim=1), loss / self.num_codebooks

    def decode(self, tokens):
        if tokens.ndim != 3 or tokens.size(1) != self.num_codebooks:
            raise ValueError("tokens phai co shape [batch, codebooks, frames].")
        quantized = 0
        for k, table in enumerate(self.codebooks):
            quantized = quantized + table(tokens[:, k])
        return quantized.transpose(1, 2)


class NeuralAudioCodec(nn.Module):
    STRIDES = (5, 4, 4, 4, 2)

    def __init__(self, latent_dim=128, num_codebooks=4, codebook_size=1024,
                 base_channels=32):
        super().__init__()
        channels = [base_channels, 64, 96, 128, 192, latent_dim]
        encoder = [nn.Conv1d(1, channels[0], 7, padding=3)]
        for index, stride in enumerate(self.STRIDES):
            encoder.append(DownsampleBlock(channels[index], channels[index + 1], stride))
        self.encoder = nn.Sequential(*encoder)
        self.quantizer = ResidualVectorQuantizer(latent_dim, num_codebooks, codebook_size)
        decoder = []
        reversed_channels = list(reversed(channels))
        for index, stride in enumerate(reversed(self.STRIDES)):
            decoder.append(UpsampleBlock(reversed_channels[index], reversed_channels[index + 1], stride))
        decoder.extend([nn.LeakyReLU(0.2), nn.Conv1d(base_channels, 1, 7, padding=3), nn.Tanh()])
        self.decoder = nn.Sequential(*decoder)

    def forward(self, waveform):
        latent = self.encoder(waveform)
        quantized, tokens, quantizer_loss = self.quantizer(latent)
        reconstruction = self.decoder(quantized)
        return reconstruction[..., :waveform.size(-1)], tokens, quantizer_loss

    @torch.no_grad()
    def encode(self, waveform):
        _, tokens, _ = self.quantizer(self.encoder(waveform))
        return tokens

    @torch.no_grad()
    def decode(self, tokens):
        return self.decoder(self.quantizer.decode(tokens))


def load_codec_checkpoint(path, device, config):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = NeuralAudioCodec(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def multi_resolution_stft_loss(prediction, target, fft_sizes=(512, 1024, 2048)):
    loss = prediction.new_zeros(())
    prediction = prediction.squeeze(1)
    target = target.squeeze(1)
    for fft_size in fft_sizes:
        hop = fft_size // 4
        window = torch.hann_window(fft_size, device=prediction.device)
        pred_mag = torch.stft(prediction, fft_size, hop, fft_size, window, return_complex=True).abs()
        target_mag = torch.stft(target, fft_size, hop, fft_size, window, return_complex=True).abs()
        spectral_convergence = (target_mag - pred_mag).norm() / target_mag.norm().clamp_min(1e-7)
        log_magnitude = F.l1_loss(torch.log1p(pred_mag), torch.log1p(target_mag))
        loss = loss + spectral_convergence + log_magnitude
    return loss / len(fft_sizes)
