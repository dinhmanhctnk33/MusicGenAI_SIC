import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from data import create_dataloader
from model import MusicGenTransformer


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# Cấu hình nhỏ, phù hợp RTX 2050 4 GB.
TRAIN_CONFIG = {
    "batch_size": 1,
    "epochs": 4,
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "audio_duration": 5.0,
    "gradient_accumulation_steps": 8,
    "test_ratio": 0.4,
    "cfg_drop_prob": 0.1,
    "codebook_size": 2048,
    "num_codebooks": 4,
    "text_dim": 768,
    "d_model": 256,
    "num_layers": 4,
    "nhead": 8,
    "dim_feedforward": 1024,
    "max_seq_len": 512,
}


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
    text_embeddings = batch["text_embeddings"].to(device, non_blocking=True)
    text_attention_mask = batch["text_attention_mask"].to(device, non_blocking=True)

    # Classifier-free guidance dropout. Giữ attention mask gốc để tránh một hàng
    # memory bị mask hoàn toàn và sinh NaN trong cross-attention.
    if cfg_drop_prob > 0:
        drop_mask = (
            torch.rand(text_embeddings.shape[0], 1, 1, device=device)
            < cfg_drop_prob
        )
        text_embeddings = text_embeddings.masked_fill(drop_mask, 0.0)

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

    return loss


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    scaler,
    device,
    accumulation_steps=8,
    cfg_drop_prob=0.1,
):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    use_amp = device.type == "cuda"

    for batch_idx, batch in enumerate(dataloader):
        loss = _compute_loss(
            model,
            batch,
            criterion,
            device,
            use_amp,
            cfg_drop_prob=cfg_drop_prob,
        )
        scaler.scale(loss / accumulation_steps).backward()

        is_update_step = (batch_idx + 1) % accumulation_steps == 0
        is_last_batch = (batch_idx + 1) == len(dataloader)
        if is_update_step or is_last_batch:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.detach().item()
        if (batch_idx + 1) % 10 == 0 or is_last_batch:
            print(
                f"   Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"- Loss: {loss.item():.4f}"
            )

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    use_amp = device.type == "cuda"

    for batch in dataloader:
        loss = _compute_loss(
            model,
            batch,
            criterion,
            device,
            use_amp,
            cfg_drop_prob=0.0,
        )
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def _create_train_test_loaders(config):
    full_loader = create_dataloader(
        batch_size=config["batch_size"],
        audio_duration=config["audio_duration"],
        shuffle=False,
        auto_download=True,
        num_workers=0,
    )
    dataset = full_loader.dataset

    test_size = int(len(dataset) * config["test_ratio"])
    train_size = len(dataset) - test_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=generator
    )

    common = {
        "batch_size": config["batch_size"],
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, test_loader


def _load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    if not path or not os.path.isfile(path):
        return 1, float("inf")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    last_loss = checkpoint.get("training_loss", float("inf"))
    print(f"Đã tiếp tục từ checkpoint: {path} (epoch {checkpoint['epoch']})")
    return start_epoch, last_loss


def train_model():
    config = TRAIN_CONFIG.copy()
    print("=" * 60)
    print("      MUSIC GENAI - HUẤN LUYỆN MÔ HÌNH (TRAIN.PY)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị huấn luyện: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Cảnh báo: CUDA không khả dụng; training sẽ rất chậm trên CPU.")

    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "checkpoints"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "latest_musicgen_model.pt")

    print("\n[1/4] Đang nạp dữ liệu và chia train/test theo tỷ lệ 60/40...")
    train_loader, test_loader = _create_train_test_loaders(config)
    print(f"Train: {len(train_loader.dataset)} | Test: {len(test_loader.dataset)}")

    print("\n[2/4] Đang khởi tạo MusicGen Transformer...")
    model = MusicGenTransformer(
        codebook_size=config["codebook_size"],
        num_codebooks=config["num_codebooks"],
        text_dim=config["text_dim"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        nhead=config["nhead"],
        dim_feedforward=config["dim_feedforward"],
        max_seq_len=config["max_seq_len"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Đặt MUSICGEN_RESUME thành đường dẫn checkpoint nếu muốn tiếp tục training.
    resume_path = os.environ.get("MUSICGEN_RESUME")
    start_epoch, last_loss = _load_checkpoint(
        resume_path, model, optimizer, scheduler, scaler, device
    )

    print(f"\n[3/4] Bắt đầu huấn luyện {config['epochs']} epoch...\n")
    for epoch in range(start_epoch, config["epochs"] + 1):
        start_time = time.time()
        print(f"--- Epoch [{epoch}/{config['epochs']}] ---")

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            accumulation_steps=config["gradient_accumulation_steps"],
            cfg_drop_prob=config["cfg_drop_prob"],
        )
        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch} hoàn tất trong {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f}"
        )

        last_loss = train_loss
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "training_loss": last_loss,
                "config": config,
            },
            checkpoint_path,
        )
        print(f"[CHECKPOINT] Đã lưu: {checkpoint_path}")
        print()

    print("\n[4/4] Đang đánh giá một lần trên test set...")
    test_loss = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

    print("=" * 60)
    print("  QUÁ TRÌNH HUẤN LUYỆN HOÀN TẤT!")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
