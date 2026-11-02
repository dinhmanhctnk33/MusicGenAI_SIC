# MusicGenAI SIC

MusicGenAI SIC là dự án thử nghiệm sinh token âm nhạc có điều kiện từ mô tả văn bản. Pipeline sử dụng:

- **T5-base** để mã hóa tiêu đề, nghệ sĩ và thể loại thành text embeddings.
- **EnCodec 32 kHz** để chuyển âm thanh thành acoustic tokens.
- **Transformer Decoder** để học phân phối token âm thanh dựa trên điều kiện văn bản.
- **FMA Small** làm dữ liệu huấn luyện và kiểm thử.

## Cấu trúc dự án

```text
Music-GenAI/
├── data.py          # Tải, chuẩn hóa và mã hóa dữ liệu FMA
├── model.py         # Mô hình MusicGen Transformer
├── train.py         # Huấn luyện, kiểm thử và lưu checkpoint
├── pyproject.toml   # Khai báo thư viện Python
├── uv.lock          # Khóa phiên bản thư viện
├── dataset/         # FMA metadata và audio (không lưu trực tiếp trong Git)
└── checkpoints/     # Checkpoint sinh ra khi huấn luyện
```

## Yêu cầu

- Windows 10/11
- Python 3.12
- GPU NVIDIA hỗ trợ CUDA; cấu hình hiện tại được tối ưu cho **RTX 2050 4 GB**
- Kết nối Internet cho lần đầu tải dataset và pretrained models
- Dung lượng trống đủ cho FMA Small và model cache

## Cài đặt

Dự án sử dụng [uv](https://docs.astral.sh/uv/) để quản lý môi trường và thư viện.

```cmd
git clone https://github.com/dinhmanhctnk33/MusicGenAI_SIC.git
cd MusicGenAI_SIC
uv sync
```

Kiểm tra PyTorch đã nhận GPU:

```cmd
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Kết quả mong đợi trên máy thử nghiệm:

```text
True
NVIDIA GeForce RTX 2050
```

## Chuẩn bị dữ liệu

Chạy:

```cmd
.\.venv\Scripts\python.exe data.py
```

Nếu chưa có dữ liệu, chương trình tự tải và giải nén:

- `fma_metadata`
- `fma_small`

Sau khi lọc, pipeline sử dụng 8.000 bài thuộc FMA Small. Mỗi mẫu gồm:

- `audio_tokens`: `[4, 250]`
- `text_embeddings`: `[64, 768]`
- `text_attention_mask`: `[64]`
- prompt và track ID

Audio được chuẩn hóa mono, 32 kHz và cắt/đệm về 5 giây. EnCodec sử dụng 4 codebook, mỗi codebook có 2.048 token.

## Kiểm tra mô hình

```cmd
.\.venv\Scripts\python.exe model.py
```

Cấu hình mặc định phù hợp GPU 4 GB:

| Tham số | Giá trị |
|---|---:|
| `d_model` | 256 |
| Số decoder layer | 4 |
| Attention heads | 8 |
| Feed-forward dimension | 1024 |
| Maximum sequence length | 512 |
| Codebook size | 2048 |
| Số codebook | 4 |

## Huấn luyện

```cmd
.\.venv\Scripts\python.exe train.py
```

Cấu hình huấn luyện hiện tại:

- 4 epoch
- Batch size 1
- Gradient accumulation 8 bước
- Automatic Mixed Precision FP16 trên CUDA
- Gradient clipping với `max_norm=1.0`
- AdamW và Cosine Annealing learning-rate scheduler
- Classifier-free guidance dropout 10%
- Chia 8.000 bài theo tỷ lệ **60% train / 40% test**:
  - 4.800 bài train
  - 3.200 bài test

Test set chỉ được đánh giá một lần sau khi hoàn tất huấn luyện, không dùng để cập nhật trọng số.

Checkpoint mới nhất được lưu tại:

```text
checkpoints/latest_musicgen_model.pt
```

Checkpoint chứa trạng thái model, optimizer, scheduler, AMP scaler, epoch, training loss và cấu hình.

## Tiếp tục huấn luyện

Trong Command Prompt:

```cmd
set MUSICGEN_RESUME=checkpoints\latest_musicgen_model.pt
.\.venv\Scripts\python.exe train.py
```

Để bắt đầu lại từ đầu, mở cửa sổ CMD mới hoặc xóa biến môi trường:

```cmd
set MUSICGEN_RESUME=
```

## Lưu ý

- Dataset, môi trường `.venv` và checkpoint không nên commit trực tiếp vào Git thông thường vì có kích thước lớn.
- T5 và EnCodec hiện mã hóa dữ liệu trực tuyến nên mỗi epoch có thể mất nhiều thời gian.
- Khi gặp `CUDA device-side assert`, cần đóng tiến trình Python cũ trước khi chạy lại vì lỗi CUDA có thể còn giữ trạng thái trong tiến trình.
- Nếu thiếu VRAM, giữ `batch_size=1`, audio 5 giây và không tăng kích thước model.

## Giấy phép dữ liệu

FMA có giấy phép riêng cho metadata và từng track âm thanh. Người sử dụng dự án cần tuân thủ giấy phép của [Free Music Archive](https://github.com/mdeff/fma) và giấy phép tương ứng của từng bản nhạc.

