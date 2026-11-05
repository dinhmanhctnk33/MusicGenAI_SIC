"""Huan luyen neural audio codec tu dau truoc khi train MusicGen."""
from __future__ import annotations

import math
import os
import random
import time

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from audio_codec import NeuralAudioCodec, multi_resolution_stft_loss
from config import (CODEC_CHECKPOINT_DIR, CODEC_CONFIG, CODEC_TRAIN_CONFIG,
                    DATA_CONFIG)
from data import FMAAudioTextDataset, download_and_extract_fma


class CodecWaveformDataset(Dataset):
    """Moi track xuat hien mot lan/epoch voi crop 2 giay ngau nhien."""
    def __init__(self, source):
        self.source = source

    def __len__(self):
        return len(self.source.track_ids)

    def __getitem__(self, index):
        for offset in range(20):
            track_id = self.source.track_ids[(index + offset) % len(self)]
            try:
                waveform = self.source._read_audio(self.source._audio_path(track_id))
                segment = random.randrange(self.source.segments_per_track)
                return self.source._crop_waveform(waveform, segment, True)
            except Exception as exc:
                if offset == 0:
                    print(f"[BO QUA CODEC] Track {track_id}: {exc}")
        raise RuntimeError("Khong tim duoc track audio hop le sau 20 lan thu.")


def _losses(model, waveform, config):
    reconstruction, _, quantizer_loss = model(waveform)
    waveform_loss = F.l1_loss(reconstruction, waveform)
    stft_loss = multi_resolution_stft_loss(reconstruction.float(), waveform.float())
    total = (config["waveform_weight"] * waveform_loss
             + config["stft_weight"] * stft_loss
             + config["commitment_weight"] * quantizer_loss)
    return total, waveform_loss.detach(), stft_loss.detach(), quantizer_loss.detach(), reconstruction


@torch.no_grad()
def validate(model, loader, device, config):
    model.eval()
    totals = torch.zeros(4, dtype=torch.float64)
    count = 0
    example = None
    for waveform in loader:
        waveform = waveform.to(device, non_blocking=True)
        values = _losses(model, waveform, config)
        totals += torch.tensor([values[i].item() for i in range(4)])
        count += 1
        if example is None:
            example = (
                waveform[0].detach().cpu(),
                values[4][0].detach().cpu(),
            )
    return totals / max(count, 1), example


def train_codec():
    config = CODEC_TRAIN_CONFIG
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"Thiet bi train codec: {device}")

    dataset_root = DATA_CONFIG["dataset_dir"]
    download_and_extract_fma(dataset_root, DATA_CONFIG["subset"])
    source = FMAAudioTextDataset(
        os.path.join(dataset_root, "fma_metadata"),
        os.path.join(dataset_root, f"fma_{DATA_CONFIG['subset']}"),
        sample_rate=DATA_CONFIG["sample_rate"],
        audio_duration=config["audio_duration"],
        subset=DATA_CONFIG["subset"], use_cache=False,
    )
    dataset = CodecWaveformDataset(source)
    validation_size = max(1, round(len(dataset) * config["validation_ratio"]))
    train_size = len(dataset) - validation_size
    train_set, validation_set = random_split(
        dataset, [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )
    common = {"batch_size": config["batch_size"], "num_workers": 0,
              "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, shuffle=True, **common)
    validation_loader = DataLoader(validation_set, shuffle=False, **common)

    model = NeuralAudioCodec(**CODEC_CONFIG).to(device)
    optimizer_kwargs = {"lr": config["learning_rate"], "weight_decay": config["weight_decay"]}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = AdamW(model.parameters(), **optimizer_kwargs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    CODEC_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    latest_path = CODEC_CHECKPOINT_DIR / "latest_codec.pt"
    best_path = CODEC_CHECKPOINT_DIR / "best_codec.pt"
    best_validation = float("inf")
    stale_epochs = 0
    start_epoch = 1
    if latest_path.is_file() and os.environ.get("MUSICGEN_CODEC_RESUME", "auto").lower() != "none":
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation = checkpoint.get("best_validation_loss", checkpoint.get("validation_loss", float("inf")))
        print(f"Tiep tuc codec tu epoch {checkpoint['epoch']}: {latest_path}")

    for epoch in range(start_epoch, config["epochs"] + 1):
        model.train()
        started = time.time()
        totals = torch.zeros(4, dtype=torch.float64)
        batches = 0
        for batch_index, waveform in enumerate(train_loader, 1):
            waveform = waveform.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                reconstruction, _, quantizer_loss = model(waveform)
                waveform_loss = F.l1_loss(reconstruction, waveform)
            stft_loss = multi_resolution_stft_loss(reconstruction.float(), waveform.float())
            loss = (config["waveform_weight"] * waveform_loss
                    + config["stft_weight"] * stft_loss
                    + config["commitment_weight"] * quantizer_loss.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            totals += torch.tensor([loss.item(), waveform_loss.item(), stft_loss.item(), quantizer_loss.item()])
            batches += 1
            if batch_index % 100 == 0:
                print(f"Epoch {epoch} [{batch_index}/{len(train_loader)}] loss={loss.item():.4f}")

        train_metrics = totals / max(batches, 1)
        validation_metrics, example = validate(model, validation_loader, device, config)
        print(
            f"Epoch {epoch}/{config['epochs']} - {time.time()-started:.0f}s | "
            f"train={train_metrics[0]:.4f} | val={validation_metrics[0]:.4f} | "
            f"wave={validation_metrics[1]:.4f} | stft={validation_metrics[2]:.4f} | "
            f"rvq={validation_metrics[3]:.4f}"
        )
        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "validation_loss": validation_metrics[0].item(),
                      "best_validation_loss": min(best_validation, validation_metrics[0].item()),
                      "codec_config": CODEC_CONFIG, "train_config": config}
        torch.save(checkpoint, latest_path)
        if example is not None:
            sf.write(CODEC_CHECKPOINT_DIR / "reconstruction_latest.wav",
                     example[1].squeeze(0).numpy(), DATA_CONFIG["sample_rate"])
            sf.write(CODEC_CHECKPOINT_DIR / "original_latest.wav",
                     example[0].squeeze(0).numpy(), DATA_CONFIG["sample_rate"])
        if validation_metrics[0] < best_validation:
            best_validation = validation_metrics[0].item()
            stale_epochs = 0
            torch.save(checkpoint, best_path)
            if example is not None:
                sf.write(CODEC_CHECKPOINT_DIR / "reconstruction_best.wav",
                         example[1].squeeze(0).numpy(), DATA_CONFIG["sample_rate"])
                sf.write(CODEC_CHECKPOINT_DIR / "original_best.wav",
                         example[0].squeeze(0).numpy(), DATA_CONFIG["sample_rate"])
            print(f"[BEST CODEC] {best_path}")
        else:
            stale_epochs += 1
            if stale_epochs >= config["early_stopping_patience"]:
                print("Early stopping codec.")
                break


if __name__ == "__main__":
    train_codec()
