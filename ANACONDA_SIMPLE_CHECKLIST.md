# ✅ Checklist Chạy Experiments — Anaconda Prompt Format (ĐÃ SỬA LỖI PẠRSER)

**CẬP NHẬT RẤT QUAN TRỌNG**: Tôi vừa kiểm tra lại file `scripts/train.py` của bạn. Parser của bạn KHÔNG nhận tham số `--seed` (số ít) và `--fold`. Thay vào đó, nó nhận tham số `--seeds 1 2 3 4 5` (số nhiều, dạng list) và tự động lặp qua tất cả 5 folds bên trong code (`train_all_folds`).

Điều này có nghĩa là bạn **không cần** copy-paste tay 25 dòng cho mỗi config! Bạn chỉ cần chạy **1 ĐƯỜNG LỆNH DUY NHẤT** cho mỗi config, script của bạn sẽ tự động chạy đủ 5 seeds x 5 folds = 25 runs.

**Status**: Đã test với parser thật của train.py 
**Date**: March 19, 2026

---

## 🔧 Chuẩn Bị Ban Đầu (Chạy 1 lần)

```bash
# Mở Anaconda Prompt, kích hoạt môi trường của bạn
conda activate <tên_môi_trường_nếu_có>

# Di chuyển vào folder project
cd c:\Users\LENOVO\Documents\PYTHON\BRT\31-10-2025\new_testing\Vital_sign_Dataset

# Kiểm tra configs tồn tại
dir vitaldb_aki\configs\*.yaml | find /c "yaml"

# Kiểm tra GPU
nvidia-smi
```

---

## 📋 TIER 1: 4.4.3 — Ablations 
*Mỗi lệnh dưới đây tương đương với 25 runs. Nó sẽ lo toàn bộ quá trình.*

### 1. No TCN Branch
*Loại bỏ hoàn toàn nhánh TCN để đánh giá RNN+Attention.*
```bash
python scripts\train.py --config vitaldb_aki/configs/no_tcn_branch.yaml --model temporal_synergy --experiment-name exp_no_tcn_branch --seeds 1 2 3 4 5
```

### 2. No RNN Branch
*Loại bỏ hoàn toàn nhánh RNN để đánh giá TCN+Attention.*
```bash
python scripts\train.py --config vitaldb_aki/configs/no_rnn_branch.yaml --model temporal_synergy --experiment-name exp_no_rnn_branch --seeds 1 2 3 4 5
```

### 3. No Attention
*Loại bỏ cơ chế gom nhóm Attention Pooling.*
```bash
python scripts\train.py --config vitaldb_aki/configs/no_attention.yaml --model temporal_synergy --experiment-name exp_no_attention --seeds 1 2 3 4 5
```

---

## 📋 TIER 1: 4.5.1 — Window Sensitivity
*Mỗi lệnh dưới đây tương đương với 25 runs.*

### Cửa sổ thời gian - Mô hình SynerT (Base Model)

```bash
# Cửa sổ 10 phút
python scripts\train.py --config vitaldb_aki/configs/window_600_sec.yaml --model temporal_synergy --experiment-name exp_window_600_sec --seeds 1 2 3 4 5

# Cửa sổ 20 phút
python scripts\train.py --config vitaldb_aki/configs/window_1200_sec.yaml --model temporal_synergy --experiment-name exp_window_1200_sec --seeds 1 2 3 4 5

# Cửa sổ 30 phút
python scripts\train.py --config vitaldb_aki/configs/window_1800_sec.yaml --model temporal_synergy --experiment-name exp_window_1800_sec --seeds 1 2 3 4 5

# Cửa sổ 60 phút (Đây là config mặc định - base case)
python scripts\train.py --config vitaldb_aki/configs/window_3600_sec.yaml --model temporal_synergy --experiment-name exp_window_3600_sec --seeds 1 2 3 4 5
```

### Cửa sổ thời gian - Mô hình Baseline (Dilated RNN)

```bash
# Cửa sổ 10 phút
python scripts\train.py --config vitaldb_aki/configs/dilated_rnn_window_600_sec.yaml --model dilated_rnn --experiment-name exp_dilated_rnn_window_600_sec --seeds 1 2 3 4 5

# Cửa sổ 20 phút
python scripts\train.py --config vitaldb_aki/configs/dilated_rnn_window_1200_sec.yaml --model dilated_rnn --experiment-name exp_dilated_rnn_window_1200_sec --seeds 1 2 3 4 5

# Cửa sổ 30 phút
python scripts\train.py --config vitaldb_aki/configs/dilated_rnn_window_1800_sec.yaml --model dilated_rnn --experiment-name exp_dilated_rnn_window_1800_sec --seeds 1 2 3 4 5

# Cửa sổ 60 phút (Base case)
python scripts\train.py --config vitaldb_aki/configs/dilated_rnn_window_3600_sec.yaml --model dilated_rnn --experiment-name exp_dilated_rnn_window_3600_sec --seeds 1 2 3 4 5
```

---

## 📋 TIER 2: Quick Tests (Không cần training)

*(Hai lệnh này chỉ load test model/oof dự đoán đã được chạy từ trước, thao tác rất nhanh)*

### 4.5.3 Khoảng thiếu dữ liệu / Nhiễu loạn dự báo
*Kiểm tra độ chênh lệch dự báo khi bị nhiễu do mất dữ liệu*
```bash
python scripts\test_missingness_stress_4_5_3.py
```

### 4.6.4 Phân tích kết quả - Giải thích kết quả Model bằng Weights (Attention)
*Triển khai giải thích Attention Map cho các ca dương tính/âm tính (Model Tính toán trên Checkpoint đã train)*
```bash
python scripts\export_attention_4_6_4.py
```
