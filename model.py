import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GENERATION_CONFIG, MODEL_CONFIG
from scratch_components import ByteTokenizer, ScratchTextEncoder


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class PositionalEncoding(nn.Module):
    """Mã hóa vị trí (Positional Encoding) dạng Sinusoidal cho chuỗi audio tokens."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model phải là số chẵn để dùng sinusoidal positional encoding.")
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


class MusicGenTransformer(nn.Module):
    """
    Kiến trúc Transformer Decoder tự hồi quy sinh nhạc từ Text Embeddings.
    Sử dụng Multi-Codebook Projection & Cross-Attention Conditioning.
    """

    def __init__(
            self,
            codebook_size: int = MODEL_CONFIG["codebook_size"],
            num_codebooks: int = MODEL_CONFIG["num_codebooks"],
            text_dim: int = MODEL_CONFIG["text_dim"],
            text_num_layers: int = MODEL_CONFIG["text_num_layers"],
            text_nhead: int = MODEL_CONFIG["text_nhead"],
            text_dim_feedforward: int = MODEL_CONFIG["text_dim_feedforward"],
            d_model: int = MODEL_CONFIG["d_model"],
            nhead: int = MODEL_CONFIG["nhead"],
            num_layers: int = MODEL_CONFIG["num_layers"],
            dim_feedforward: int = MODEL_CONFIG["dim_feedforward"],
            dropout: float = MODEL_CONFIG["dropout"],
            max_seq_len: int = MODEL_CONFIG["max_seq_len"],
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.text_encoder = ScratchTextEncoder(
            vocab_size=ByteTokenizer.VOCAB_SIZE,
            d_model=text_dim,
            max_length=64,
            nhead=text_nhead,
            num_layers=text_num_layers,
            dim_feedforward=text_dim_feedforward,
            dropout=dropout,
        )

        if d_model % nhead != 0:
            raise ValueError("d_model phải chia hết cho nhead.")

        # 1. Embedding layer cho từng Codebook + Token đặc biệt <BOS>/<PAD>
        # Token id = codebook_size dùng làm token bắt đầu chuỗi <BOS>
        self.codebook_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size + 1, d_model) for _ in range(num_codebooks)
        ])

        # Chieu dac trung tu text encoder tu huan luyen sang decoder.
        self.text_projection = nn.Linear(text_dim, d_model, bias=False)

        # 3. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

        # 4. Transformer Decoder Stack
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 5. Prediction Heads: Dự đoán logits riêng cho từng codebook
        self.heads = nn.ModuleList([
            nn.Linear(d_model, codebook_size) for _ in range(num_codebooks)
        ])

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Tạo causal mask đảm bảo mô hình chỉ nhìn về các token quá khứ."""
        mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(
            self,
            target_tokens: torch.Tensor,
            text_embeddings: torch.Tensor,
            text_attention_mask: torch.Tensor = None,
            target_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass cho quá trình Training.

        Args:
            target_tokens  : [batch_size, num_codebooks, seq_len] - Chuỗi token âm thanh
            text_embeddings: token ids [batch, text_seq] hoac hidden states [batch, text_seq, text_dim]

        Returns:
            logits: [batch_size, num_codebooks, seq_len, codebook_size]
        """
        if target_tokens.ndim != 3:
            raise ValueError("target_tokens phải có shape [batch, codebooks, sequence].")

        bs, K, seq_len = target_tokens.shape
        device = target_tokens.device
        if K != self.num_codebooks:
            raise ValueError(
                f"Model cần {self.num_codebooks} codebook, nhưng đầu vào có {K}."
            )
        delayed_len = seq_len + self.num_codebooks - 1
        if delayed_len > self.max_seq_len:
            raise ValueError(
                f"Sequence sau delay pattern dài {delayed_len}, "
                f"vượt max_seq_len={self.max_seq_len}."
            )
        if target_tokens.numel() > 0:
            token_min = int(target_tokens.min())
            token_max = int(target_tokens.max())
            if token_min < 0 or token_max >= self.codebook_size:
                raise ValueError(
                    f"Token phải nằm trong [0, {self.codebook_size - 1}], "
                    f"nhưng nhận min={token_min}, max={token_max}."
                )

        # Delay pattern cho phép codebook sau thấy codebook trước của cùng frame.
        delayed_tokens = torch.full(
            (bs, K, delayed_len), self.codebook_size,
            dtype=torch.long, device=device
        )
        for k in range(K):
            delayed_tokens[:, k, k:k + seq_len] = target_tokens[:, k, :]

        # Teacher forcing: dịch toàn bộ delay pattern sang phải một bước.
        bos = torch.full(
            (bs, K, 1), self.codebook_size, dtype=torch.long, device=device
        )
        decoder_tokens = torch.cat([bos, delayed_tokens[:, :, :-1]], dim=-1)

        if text_embeddings.ndim == 2:
            if text_attention_mask is None:
                text_attention_mask = text_embeddings.ne(ByteTokenizer.PAD_ID)
            text_embeddings = self.text_encoder(text_embeddings, text_attention_mask)
        delayed_logits = self._decode(
            decoder_tokens,
            text_embeddings,
            text_attention_mask=text_attention_mask,
            target_mask=target_mask,
        )
        return torch.stack(
            [delayed_logits[:, k, k:k + seq_len, :] for k in range(K)],
            dim=1,
        )

    def _decode(
            self,
            decoder_tokens: torch.Tensor,
            text_embeddings: torch.Tensor,
            text_attention_mask: torch.Tensor = None,
            target_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Decode chuỗi đã có BOS; dùng chung cho training và generation."""
        if decoder_tokens.ndim != 3:
            raise ValueError("decoder_tokens phải có shape [batch, codebooks, sequence].")
        if text_embeddings.ndim != 3:
            raise ValueError("text_embeddings phải có shape [batch, sequence, text_dim].")
        bs, K, seq_len = decoder_tokens.shape
        device = decoder_tokens.device
        if K != self.num_codebooks:
            raise ValueError(f"Cần {self.num_codebooks} codebook, nhưng nhận {K}.")
        if bs != text_embeddings.size(0):
            raise ValueError("Batch size của audio và text không khớp.")
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence dài {seq_len}, vượt max_seq_len={self.max_seq_len}."
            )

        # Sum Embedding từ tất cả K codebooks
        audio_emb = torch.zeros(
            bs, seq_len, self.d_model, device=device,
            dtype=self.codebook_embeddings[0].weight.dtype
        )
        for k in range(K):
            audio_emb += self.codebook_embeddings[k](decoder_tokens[:, k, :])
        audio_emb = audio_emb / math.sqrt(K)

        # Thêm Positional Encoding
        audio_emb = self.pos_encoder(audio_emb)
        audio_emb = self.dropout(audio_emb)

        # Biến đổi Text Condition
        text_cond = self.text_projection(text_embeddings)

        # Tạo Causal Mask tự hồi quy
        if target_mask is None:
            target_mask = self._generate_causal_mask(seq_len, device)

        memory_key_padding_mask = None
        if text_attention_mask is not None:
            if text_attention_mask.shape != text_embeddings.shape[:2]:
                raise ValueError(
                    "text_attention_mask phải có shape [batch, text_sequence]."
                )
            memory_key_padding_mask = ~text_attention_mask.to(device=device, dtype=torch.bool)

        # Đi qua Transformer Decoder với Cross-Attention
        decoder_output = self.transformer_decoder(
            tgt=audio_emb,
            memory=text_cond,
            tgt_mask=target_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        # Dự đoán logits cho từng codebook
        logits = torch.stack([head(decoder_output) for head in self.heads], dim=1)
        return logits  # Shape: [batch_size, num_codebooks, seq_len, codebook_size]

    @torch.no_grad()
    def generate(
            self,
            text_embeddings: torch.Tensor,
            text_attention_mask: torch.Tensor = None,
            max_steps: int = 500,
            temperature: float = 1.0,
            top_k: int = 250,
            guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Inference / Sampling tự hồi quy để sinh ra chuỗi Acoustic Tokens từ Text Embedding.
        """
        if temperature <= 0:
            raise ValueError("temperature phải lớn hơn 0.")
        if guidance_scale < 1.0:
            raise ValueError("guidance_scale phải >= 1.0.")
        max_audio_steps = self.max_seq_len - self.num_codebooks + 1
        if max_steps < 1 or max_steps > max_audio_steps:
            raise ValueError(
                f"max_steps phải nằm trong [1, {max_audio_steps}]."
            )
        if text_embeddings.ndim == 2:
            if text_attention_mask is None:
                text_attention_mask = text_embeddings.ne(ByteTokenizer.PAD_ID)
            text_embeddings = self.text_encoder(text_embeddings, text_attention_mask)
        device = text_embeddings.device
        bs = text_embeddings.size(0)

        # Khởi tạo token <BOS> (id = codebook_size)
        delayed_tokens = torch.full(
            (bs, self.num_codebooks, 1),
            fill_value=self.codebook_size,
            dtype=torch.long,
            device=device
        )

        was_training = self.training
        self.eval()
        try:
            pattern_steps = max_steps + self.num_codebooks - 1
            for pattern_idx in range(pattern_steps):
                if guidance_scale > 1.0:
                    # Chạy conditional/unconditional trong cùng một batch để CFG
                    # chỉ tốn một forward pass cho mỗi bước autoregressive.
                    cfg_tokens = torch.cat([delayed_tokens, delayed_tokens], dim=0)
                    cfg_embeddings = torch.cat(
                        [torch.zeros_like(text_embeddings), text_embeddings], dim=0
                    )
                    cfg_mask = None
                    if text_attention_mask is not None:
                        cfg_mask = torch.cat(
                            [text_attention_mask, text_attention_mask], dim=0
                        )
                    uncond_logits, cond_logits = self._decode(
                        cfg_tokens,
                        cfg_embeddings,
                        text_attention_mask=cfg_mask,
                    ).chunk(2, dim=0)
                    logits = uncond_logits + guidance_scale * (
                        cond_logits - uncond_logits
                    )
                else:
                    logits = self._decode(
                        delayed_tokens, text_embeddings,
                        text_attention_mask=text_attention_mask
                    )
                next_logits = logits[:, :, -1, :] / temperature
                next_frame = torch.full(
                    (bs, self.num_codebooks), self.codebook_size,
                    dtype=torch.long, device=device
                )
                for k in range(self.num_codebooks):
                    token_idx = pattern_idx - k
                    if token_idx < 0 or token_idx >= max_steps:
                        continue
                    k_logits = next_logits[:, k, :]
                    if top_k > 0:
                        threshold = torch.topk(
                            k_logits, min(top_k, k_logits.size(-1))
                        ).values[:, -1:]
                        k_logits = k_logits.masked_fill(
                            k_logits < threshold, -float("inf")
                        )
                    probs = F.softmax(k_logits, dim=-1)
                    next_frame[:, k] = torch.multinomial(
                        probs, num_samples=1
                    ).squeeze(1)
                delayed_tokens = torch.cat(
                    [delayed_tokens, next_frame.unsqueeze(-1)], dim=-1
                )

            pattern = delayed_tokens[:, :, 1:]
            return torch.stack(
                [pattern[:, k, k:k + max_steps] for k in range(self.num_codebooks)],
                dim=1,
            )
        finally:
            self.train(was_training)


# --- Kiểm thử chạy mô hình ---
if __name__ == "__main__":
    print("=== KIỂM THỬ KHỞI TẠO MÔ HÌNH TRANSFORMER MODEL.PY ===")

    # Cấu hình thử nghiệm
    batch_size = 1
    num_codebooks = MODEL_CONFIG["num_codebooks"]
    audio_seq_len = 250  # 5 giây, an toàn hơn với GPU 4 GB
    text_seq_len = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    # Tạo giả lập dữ liệu đầu vào từ data.py
    dummy_audio_tokens = torch.randint(
        0,
        MODEL_CONFIG["codebook_size"],
        (batch_size, num_codebooks, audio_seq_len),
        device=device,
    )
    dummy_text_embeddings = torch.randn(
        batch_size, text_seq_len, MODEL_CONFIG["text_dim"], device=device
    )

    # Khởi tạo mô hình MusicGen Transformer
    model = MusicGenTransformer(**MODEL_CONFIG).to(device)

    # 1. Test Forward Pass (Dùng trong Training)
    logits = model(dummy_audio_tokens, dummy_text_embeddings)
    print(f"-> Forward Logits Output Shape: {list(logits.shape)}")
    print(
        "   Kỳ vọng: [Batch, Codebooks, SeqLen, CodebookSize] -> "
        f"[1, {num_codebooks}, {audio_seq_len}, {MODEL_CONFIG['codebook_size']}]"
    )

    # 2. Test Generation / Sampling Pass (Dùng trong Inference)
    print("\nĐang chạy thử nghiệm hàm generate (sinh 10 bước token)...")
    gen_tokens = model.generate(
        dummy_text_embeddings,
        text_attention_mask=torch.ones(
            batch_size, text_seq_len, dtype=torch.long, device=device
        ),
        max_steps=10,
        guidance_scale=GENERATION_CONFIG["guidance_scale"],
    )
    print(f"-> Generated Tokens Output Shape: {list(gen_tokens.shape)}")
    print("   (Kỳ vọng: [Batch, Codebooks, Steps] -> [1, 4, 10])")
