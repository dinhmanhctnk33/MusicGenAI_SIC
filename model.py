import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


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
            codebook_size: int = 2048,
            num_codebooks: int = 4,
            text_dim: int = 768,  # Chiều ẩn từ T5-base
            d_model: int = 256,  # Cấu hình phù hợp GPU 4 GB
            nhead: int = 8,
            num_layers: int = 4,
            dim_feedforward: int = 1024,
            dropout: float = 0.1,
            max_seq_len: int = 512
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        if d_model % nhead != 0:
            raise ValueError("d_model phải chia hết cho nhead.")

        # 1. Embedding layer cho từng Codebook + Token đặc biệt <BOS>/<PAD>
        # Token id = codebook_size dùng làm token bắt đầu chuỗi <BOS>
        self.codebook_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size + 1, d_model) for _ in range(num_codebooks)
        ])

        # 2. Chiếu đặc trưng Text Embedding từ T5 (768 -> d_model)
        self.text_projection = nn.Linear(text_dim, d_model)

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
            text_embeddings: [batch_size, text_seq_len, text_dim] - Vector điều kiện từ T5

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
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence dài {seq_len}, vượt max_seq_len={self.max_seq_len}."
            )

        # Teacher forcing: dịch chuỗi sang phải để model không nhìn thấy token cần dự đoán.
        bos = torch.full(
            (bs, K, 1), self.codebook_size, dtype=torch.long, device=device
        )
        decoder_tokens = torch.cat([bos, target_tokens[:, :, :-1]], dim=-1)

        return self._decode(
            decoder_tokens,
            text_embeddings,
            text_attention_mask=text_attention_mask,
            target_mask=target_mask,
        )

    def _decode(
            self,
            decoder_tokens: torch.Tensor,
            text_embeddings: torch.Tensor,
            text_attention_mask: torch.Tensor = None,
            target_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Decode chuỗi đã có BOS; dùng chung cho training và generation."""
        bs, K, seq_len = decoder_tokens.shape
        device = decoder_tokens.device

        # Sum Embedding từ tất cả K codebooks
        audio_emb = torch.zeros(
            bs, seq_len, self.d_model, device=device,
            dtype=self.codebook_embeddings[0].weight.dtype
        )
        for k in range(K):
            audio_emb += self.codebook_embeddings[k](decoder_tokens[:, k, :])

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
            top_k: int = 250
    ) -> torch.Tensor:
        """
        Inference / Sampling tự hồi quy để sinh ra chuỗi Acoustic Tokens từ Text Embedding.
        """
        self.eval()
        if temperature <= 0:
            raise ValueError("temperature phải lớn hơn 0.")
        if max_steps < 1 or max_steps > self.max_seq_len - 1:
            raise ValueError(
                f"max_steps phải nằm trong [1, {self.max_seq_len - 1}]."
            )
        device = text_embeddings.device
        bs = text_embeddings.size(0)

        # Khởi tạo token <BOS> (id = codebook_size)
        generated_tokens = torch.full(
            (bs, self.num_codebooks, 1),
            fill_value=self.codebook_size,
            dtype=torch.long,
            device=device
        )

        for _ in range(max_steps):
            # Lấy logits cho bước hiện tại
            logits = self._decode(
                generated_tokens,
                text_embeddings,
                text_attention_mask=text_attention_mask
            )
            next_logits = logits[:, :, -1, :] / temperature  # [bs, K, codebook_size]

            next_tokens = []
            for k in range(self.num_codebooks):
                k_logits = next_logits[:, k, :]

                # Top-k Sampling
                if top_k > 0:
                    v, _ = torch.topk(k_logits, min(top_k, k_logits.size(-1)))
                    k_logits[k_logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(k_logits, dim=-1)
                sampled_token = torch.multinomial(probs, num_samples=1)  # [bs, 1]
                next_tokens.append(sampled_token)

            # Ghép token vừa sample vào chuỗi đã sinh [bs, K, 1]
            next_tokens_tensor = torch.cat(next_tokens, dim=1).unsqueeze(-1)
            generated_tokens = torch.cat([generated_tokens, next_tokens_tensor], dim=-1)

        # Trả về chuỗi token đã sinh (loại bỏ token <BOS> ở đầu)
        return generated_tokens[:, :, 1:]


# --- Kiểm thử chạy mô hình ---
if __name__ == "__main__":
    print("=== KIỂM THỬ KHỞI TẠO MÔ HÌNH TRANSFORMER MODEL.PY ===")

    # Cấu hình thử nghiệm
    batch_size = 1
    num_codebooks = 4
    audio_seq_len = 250  # 5 giây, an toàn hơn với GPU 4 GB
    text_seq_len = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    # Tạo giả lập dữ liệu đầu vào từ data.py
    dummy_audio_tokens = torch.randint(0, 2048, (batch_size, num_codebooks, audio_seq_len)).to(device)
    dummy_text_embeddings = torch.randn(batch_size, text_seq_len, 768).to(device)

    # Khởi tạo mô hình MusicGen Transformer
    model = MusicGenTransformer(
        codebook_size=2048,
        num_codebooks=num_codebooks,
        text_dim=768,
        d_model=256,
        num_layers=4,
        dim_feedforward=1024,
        nhead=8
    ).to(device)

    # 1. Test Forward Pass (Dùng trong Training)
    logits = model(dummy_audio_tokens, dummy_text_embeddings)
    print(f"-> Forward Logits Output Shape: {list(logits.shape)}")
    print("   (Kỳ vọng: [Batch, Codebooks, SeqLen, CodebookSize] -> [1, 4, 250, 2048])")

    # 2. Test Generation / Sampling Pass (Dùng trong Inference)
    print("\nĐang chạy thử nghiệm hàm generate (sinh 10 bước token)...")
    gen_tokens = model.generate(dummy_text_embeddings, max_steps=10)
    print(f"-> Generated Tokens Output Shape: {list(gen_tokens.shape)}")
    print("   (Kỳ vọng: [Batch, Codebooks, Steps] -> [1, 4, 10])")
