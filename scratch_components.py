"""Cac thanh phan tu xay dung, khong dung model/tokenizer pretrained."""

from __future__ import annotations

import unicodedata

import torch
import torch.nn as nn


class ByteTokenizer:
    """UTF-8 byte tokenizer xac dinh, khong can vocabulary ben ngoai."""

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    BYTE_OFFSET = 3
    VOCAB_SIZE = 259

    def encode(self, text: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        text = unicodedata.normalize("NFKC", text.strip())
        payload = list(text.encode("utf-8"))[: max(0, max_length - 2)]
        ids = [self.BOS_ID, *[value + self.BYTE_OFFSET for value in payload], self.EOS_ID]
        mask = [1] * len(ids)
        padding = max_length - len(ids)
        ids.extend([self.PAD_ID] * padding)
        mask.extend([0] * padding)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    def batch_encode(self, texts: list[str], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [self.encode(text, max_length) for text in texts]
        return torch.stack([item[0] for item in encoded]), torch.stack([item[1] for item in encoded])


class ScratchTextEncoder(nn.Module):
    """Transformer encoder khoi tao ngau nhien va duoc train end-to-end."""

    def __init__(self, vocab_size: int, d_model: int, max_length: int,
                 nhead: int, num_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        valid_mask = attention_mask.bool() & input_ids.ne(ByteTokenizer.PAD_ID)
        safe_mask = valid_mask.clone()
        empty_rows = ~safe_mask.any(dim=1)
        safe_mask[empty_rows, 0] = True
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.dropout(hidden)
        hidden = self.encoder(hidden, src_key_padding_mask=~safe_mask)
        hidden = self.norm(hidden)
        return torch.nan_to_num(hidden) * valid_mask.unsqueeze(-1)
