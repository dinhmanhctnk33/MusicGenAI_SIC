import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from config import (
    ARCHITECTURE_KEYS,
    CHECKPOINT_DIR,
    MODEL_CONFIG,
    build_run_config,
)
from data import create_dataloader
from model import MusicGenTransformer


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def _compute_loss(model, batch, criterion, device, use_amp, cfg_drop_prob=0.0):
    audio_tokens_cpu = batch["audio_tokens"].long()
    token_min = int(audio_tokens_cpu.min())
    token_max = int(audio_tokens_cpu.max())
    if token_min < 0 or token_max >= model.codebook_size:
        raise ValueError(
            f"Audio token nằm ngoài miền [0, {model.codebook_size - 1}]: "
            f"min={token_min}, max={token_max}."
        )
    audio_tokens = audio_tokens_cpu.to(device, non_blocking=True)
    text_embeddings = batch["text_token_ids"].to(device, non_blocking=True)
    text_attention_mask = batch["text_attention_mask"].to(device, non_blocking=True)

    # Classifier-free guidance dropout. Giữ attention mask gốc để tránh một hàng
    # memory bị mask hoàn toàn và sinh NaN trong cross-attention.
    if cfg_drop_prob > 0:
        drop_mask = (
            torch.rand(text_embeddings.shape[0], 1, device=device)
            < cfg_drop_prob
        )
        text_embeddings = text_embeddings.masked_fill(drop_mask, 0)

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=use_amp,
    ):
        # Model tự thêm BOS và dịch chuỗi sang phải cho teacher forcing.
        logits = model(
            audio_tokens,
            text_embeddings,
            text_attention_mask=text_attention_mask,
        )
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            audio_tokens.reshape(-1),
        )
        codebook_losses = torch.stack([
            criterion(
                logits[:, k].reshape(-1, logits.size(-1)),
                audio_tokens[:, k].reshape(-1),
            ).detach()
            for k in range(logits.size(1))
        ])

    predictions = logits.detach().argmax(dim=-1)
    codebook_correct = (predictions == audio_tokens).sum(dim=(0, 2)).cpu()
    tokens_per_codebook = audio_tokens.size(0) * audio_tokens.size(2)

    return loss, codebook_losses, codebook_correct, tokens_per_codebook


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    scaler,
    device,
    accumulation_steps=8,
    cfg_drop_prob=0.1,
    gradient_clip_norm=1.0,
    log_every_batches=10,
):
    model.train()
    total_loss = 0.0
    total_codebook_losses = torch.zeros(model.num_codebooks, dtype=torch.float64)
    total_codebook_correct = torch.zeros(model.num_codebooks, dtype=torch.long)
    total_tokens_per_codebook = 0
    optimizer.zero_grad(set_to_none=True)
    use_amp = device.type == "cuda"

    for batch_idx, batch in enumerate(dataloader):
        loss, codebook_losses, codebook_correct, token_count = _compute_loss(
            model,
            batch,
            criterion,
            device,
            use_amp,
            cfg_drop_prob=cfg_drop_prob,
        )
        group_start = (batch_idx // accumulation_steps) * accumulation_steps
        actual_group_size = min(accumulation_steps, len(dataloader) - group_start)
        scaler.scale(loss / actual_group_size).backward()

        is_update_step = (batch_idx + 1) % accumulation_steps == 0
        is_last_batch = (batch_idx + 1) == len(dataloader)
        if is_update_step or is_last_batch:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.detach().item()
        total_codebook_losses += codebook_losses.double().cpu()
        total_codebook_correct += codebook_correct
        total_tokens_per_codebook += token_count
        if (batch_idx + 1) % log_every_batches == 0 or is_last_batch:
            running_accuracy = (
                total_codebook_correct.sum().item()
                / max(total_tokens_per_codebook * model.num_codebooks, 1)
            )
            print(
                f"   Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"- Loss: {loss.item():.4f} "
                f"- Running Accuracy: {running_accuracy * 100:.2f}%"
            )

    denominator = max(len(dataloader), 1)
    codebook_accuracy = total_codebook_correct.double() / max(total_tokens_per_codebook, 1)
    accuracy = codebook_accuracy.mean().item()
    return (
        total_loss / denominator,
        total_codebook_losses / denominator,
        accuracy,
        codebook_accuracy,
    )


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_codebook_losses = torch.zeros(model.num_codebooks, dtype=torch.float64)
    total_codebook_correct = torch.zeros(model.num_codebooks, dtype=torch.long)
    total_tokens_per_codebook = 0
    use_amp = device.type == "cuda"

    for batch in dataloader:
        loss, codebook_losses, codebook_correct, token_count = _compute_loss(
            model,
            batch,
            criterion,
            device,
            use_amp,
            cfg_drop_prob=0.0,
        )
        total_loss += loss.item()
        total_codebook_losses += codebook_losses.double().cpu()
        total_codebook_correct += codebook_correct
        total_tokens_per_codebook += token_count

    denominator = max(len(dataloader), 1)
    codebook_accuracy = total_codebook_correct.double() / max(total_tokens_per_codebook, 1)
    accuracy = codebook_accuracy.mean().item()
    return (
        total_loss / denominator,
        total_codebook_losses / denominator,
        accuracy,
        codebook_accuracy,
    )


class CropModeDataset(Dataset):
    """Subset view chọn crop ngẫu nhiên cho train, crop cố định cho eval."""

    def __init__(self, subset, random_crop):
        self.subset = subset
        self.random_crop = random_crop

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        source_idx = self.subset.indices[idx]
        return self.subset.dataset.get_item(
            source_idx, random_crop=self.random_crop
        )


def _create_data_loaders(config):
    full_loader = create_dataloader(
        batch_size=config["batch_size"],
        audio_duration=config["audio_duration"],
        shuffle=False,
        auto_download=True,
        num_workers=0,
        device=config["encoder_device"],
        use_cache=config["use_encoded_cache"],
    )
    dataset = full_loader.dataset
    if config["precompute_cache"]:
        dataset.populate_cache()
        dataset.release_encoders()

    # Chia theo track trước, sau đó mới mở rộng sang các segment.
    # Như vậy các crop của cùng bài không thể rò sang test set.
    track_count = len(dataset.track_ids)
    validation_tracks = max(1, int(track_count * config["validation_ratio"]))
    test_tracks = max(1, int(track_count * config["test_ratio"]))
    train_tracks = track_count - validation_tracks - test_tracks
    if train_tracks < 1:
        raise ValueError("Dataset quá nhỏ để chia train/validation/test.")
    generator = torch.Generator().manual_seed(config["random_seed"])
    shuffled_track_indices = torch.randperm(track_count, generator=generator).tolist()
    train_track_ids = {
        dataset.track_ids[i] for i in shuffled_track_indices[:train_tracks]
    }
    validation_track_ids = {
        dataset.track_ids[i]
        for i in shuffled_track_indices[train_tracks:train_tracks + validation_tracks]
    }
    test_track_ids = {
        dataset.track_ids[i] for i in shuffled_track_indices[train_tracks + validation_tracks:]
    }
    train_indices, validation_indices, test_indices = [], [], []
    for sample_idx in range(len(dataset)):
        track_id = dataset.sample_track_id(sample_idx)
        if track_id in train_track_ids:
            train_indices.append(sample_idx)
        elif track_id in validation_track_ids:
            validation_indices.append(sample_idx)
        elif track_id in test_track_ids:
            test_indices.append(sample_idx)

    train_subset = Subset(dataset, train_indices)
    validation_subset = Subset(dataset, validation_indices)
    test_subset = Subset(dataset, test_indices)
    cache_is_precomputed = (
        config["use_encoded_cache"] and config["precompute_cache"]
    )
    train_dataset = CropModeDataset(
        train_subset, random_crop=not cache_is_precomputed
    )
    validation_dataset = CropModeDataset(validation_subset, random_crop=False)
    test_dataset = CropModeDataset(test_subset, random_crop=False)

    common = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }
    if config["num_workers"] > 0:
        common.update({
            "persistent_workers": True,
            "prefetch_factor": config["prefetch_factor"],
        })
    genre_counts = {}
    train_genres = []
    for source_idx in train_indices:
        genre = dataset.sample_genre(source_idx)
        train_genres.append(genre)
        genre_counts[genre] = genre_counts.get(genre, 0) + 1
    sample_weights = [1.0 / genre_counts[genre] for genre in train_genres]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(config["random_seed"]),
    )
    train_loader = DataLoader(train_dataset, sampler=sampler, **common)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, validation_loader, test_loader


def _load_checkpoint(path, model, optimizer, scheduler, scaler, device, config):
    if not path or not os.path.isfile(path):
        return 1, float("inf"), float("inf")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved_config = checkpoint.get("config", {})
    mismatches = [
        f"{key}: checkpoint={saved_config.get(key)!r}, current={config[key]!r}"
        for key in ARCHITECTURE_KEYS
        if saved_config.get(key) != config[key]
    ]
    if mismatches:
        raise ValueError(
            "Checkpoint không tương thích với model hiện tại:\n- "
            + "\n- ".join(mismatches)
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if saved_config.get("epochs") == config["epochs"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        for group in optimizer.param_groups:
            group["lr"] = config["learning_rate"]
        print("Số epoch đã thay đổi; khởi tạo lại LR scheduler.")
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    last_loss = checkpoint.get("training_loss", float("inf"))
    print(f"Đã tiếp tục từ checkpoint: {path} (epoch {checkpoint['epoch']})")
    best_validation_loss = checkpoint.get("best_validation_loss", float("inf"))
    return start_epoch, last_loss, best_validation_loss


def train_model():
    config = build_run_config()
    _set_random_seed(config["random_seed"])
    print("=" * 60)
    print("      MUSIC GENAI - HUẤN LUYỆN MÔ HÌNH (TRAIN.PY)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị huấn luyện: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Cảnh báo: CUDA không khả dụng; training sẽ rất chậm trên CPU.")

    checkpoint_dir = str(CHECKPOINT_DIR / config["run_name"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "latest_musicgen_model.pt")
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_musicgen_model.pt")

    print("\n[1/4] Đang nạp dữ liệu và chia train/validation/test...")
    train_loader, validation_loader, test_loader = _create_data_loaders(config)
    print(
        f"Train: {len(train_loader.dataset)} | "
        f"Validation: {len(validation_loader.dataset)} | "
        f"Test: {len(test_loader.dataset)}"
    )
    print(
        f"DataLoader workers: {config['num_workers']} | "
        f"Prefetch/worker: {config['prefetch_factor']} | "
        f"Encoded cache: {'bat' if config['use_encoded_cache'] else 'tat'}"
    )
    print("Text encoder va audio codec: tu xay dung, khong pretrained.")

    print("\n[2/4] Đang khởi tạo MusicGen Transformer...")
    model = MusicGenTransformer(**MODEL_CONFIG).to(device)

    optimizer_kwargs = {
        "lr": config["learning_rate"],
        "weight_decay": config["weight_decay"],
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = AdamW(model.parameters(), **optimizer_kwargs)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=config["min_learning_rate"],
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Mặc định tiếp tục latest checkpoint nếu có. Có thể ép train
    # lại từ đầu bằng MUSICGEN_RESUME=none, hoặc chỉ định checkpoint khác.
    resume_setting = os.environ.get("MUSICGEN_RESUME", "auto").strip()
    if resume_setting.lower() == "auto":
        resume_path = checkpoint_path if os.path.isfile(checkpoint_path) else None
    elif resume_setting.lower() in {"none", "off", "false", "0"}:
        resume_path = None
    else:
        resume_path = resume_setting
    start_epoch, last_loss, best_validation_loss = _load_checkpoint(
        resume_path, model, optimizer, scheduler, scaler, device, config
    )
    epochs_without_improvement = 0

    print(f"\n[3/4] Bắt đầu huấn luyện {config['epochs']} epoch...\n")
    for epoch in range(start_epoch, config["epochs"] + 1):
        start_time = time.time()
        print(f"--- Epoch [{epoch}/{config['epochs']}] ---")

        train_loss, train_codebook_losses, train_accuracy, train_codebook_accuracy = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            accumulation_steps=config["gradient_accumulation_steps"],
            cfg_drop_prob=config["cfg_drop_prob"],
            gradient_clip_norm=config["gradient_clip_norm"],
            log_every_batches=config["log_every_batches"],
        )
        validation_loss, validation_codebook_losses, validation_accuracy, validation_codebook_accuracy = validate(
            model, validation_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch} hoàn tất trong {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy * 100:.2f}% | "
            f"Validation Loss: {validation_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy * 100:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        print(
            "   Train loss/codebook: "
            + ", ".join(
                f"CB{k}={value:.4f}"
                for k, value in enumerate(train_codebook_losses.tolist())
            )
        )
        print(
            "   Validation loss/codebook: "
            + ", ".join(
                f"CB{k}={value:.4f}"
                for k, value in enumerate(validation_codebook_losses.tolist())
            )
        )
        print(
            "   Train accuracy/codebook: "
            + ", ".join(
                f"CB{k}={value * 100:.2f}%"
                for k, value in enumerate(train_codebook_accuracy.tolist())
            )
        )
        print(
            "   Validation accuracy/codebook: "
            + ", ".join(
                f"CB{k}={value * 100:.2f}%"
                for k, value in enumerate(validation_codebook_accuracy.tolist())
            )
        )

        last_loss = train_loss
        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "training_loss": last_loss,
            "validation_loss": validation_loss,
            "training_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
            "training_codebook_accuracy": train_codebook_accuracy.tolist(),
            "validation_codebook_accuracy": validation_codebook_accuracy.tolist(),
            "best_validation_loss": best_validation_loss,
            "config": config,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"[CHECKPOINT] Đã lưu: {checkpoint_path}")
        if improved:
            torch.save(checkpoint, best_checkpoint_path)
            print(f"[BEST] Validation loss mới: {best_checkpoint_path}")
        print()

        if epochs_without_improvement >= config["early_stopping_patience"]:
            print(
                "Early stopping: validation loss không cải thiện trong "
                f"{config['early_stopping_patience']} epoch."
            )
            break

    print("\n[4/4] Đang đánh giá một lần trên test set...")
    if os.path.isfile(best_checkpoint_path):
        best_checkpoint = torch.load(
            best_checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(best_checkpoint["model_state_dict"])
        print(f"Đã nạp best checkpoint từ epoch {best_checkpoint['epoch']}.")
    test_loss, test_codebook_losses, test_accuracy, test_codebook_accuracy = validate(
        model, test_loader, criterion, device
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(
        "Test loss/codebook: "
        + ", ".join(
            f"CB{k}={value:.4f}"
            for k, value in enumerate(test_codebook_losses.tolist())
        )
    )
    print(
        "Test accuracy/codebook: "
        + ", ".join(
            f"CB{k}={value * 100:.2f}%"
            for k, value in enumerate(test_codebook_accuracy.tolist())
        )
    )

    print("=" * 60)
    print("  QUÁ TRÌNH HUẤN LUYỆN HOÀN TẤT!")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
