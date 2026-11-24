# Nhận diện Mống mắt (Iris Recognition): Truyền thống vs Deep Learning

Dự án nghiên cứu và so sánh phương pháp **Truyền thống** (Log-Gabor IrisCode + Hamming Distance, LBP/HOG + SVM) và **Deep Learning** (CNN Classification, ArcFace Metric Learning) trên bộ dữ liệu **CASIA-Iris-Thousand** cho cả hai bài toán:
- **Identification (`1:N` — Closed-set)**: Nhận diện danh tính từ ảnh mống mắt.
- **Verification (`1:1` — Open-set)**: Xác thực 2 ảnh có thuộc cùng một mắt hay không (trên nhóm người hoàn toàn mới chưa xuất hiện khi train).

---

## Cấu trúc thư mục dự án

```text
c:/prj3/
├── configs/
│   └── dataset_config.yaml      # Cấu hình tham số chia tập & chất lượng ảnh
├── data/
│   ├── raw/                     # Dữ liệu gốc (không commit lên Git)
│   ├── processed/               # Ảnh Rubber Sheet / ROI sau tiền xử lý (không commit)
│   └── splits/                  # Bộ chia CSV cố định dùng chung (được theo dõi trên Git)
├── notebooks/
│   └── 01_eda_and_splits.ipynb  # Notebook Colab: Tải data, EDA, sinh file splits
├── results/                     # Biểu đồ EDA, trọng số model, bảng đánh giá
├── scripts/
│   └── prepare_splits.py        # Script CLI sinh file splits từ dòng lệnh
├── src/
│   ├── data/                    # Quét dataset, kiểm tra chất lượng, chia tập & check leakage
│   ├── preprocessing/           # Phân đoạn & chuẩn hóa Rubber Sheet (Giai đoạn 2)
│   ├── traditional/             # Log-Gabor IrisCode, LBP/HOG + ML (Giai đoạn 3)
│   ├── deep_learning/           # CNN & ArcFace Metric Learning (Giai đoạn 4)
│   └── evaluation/              # Chỉ số Rank-k, EER, FAR/FRR, ROC/DET (Giai đoạn 5)
├── tests/
│   └── test_dataset_utils.py    # Unit test cho pipeline xử lý & chia tập dữ liệu
├── .gitignore
└── requirements.txt
```

---

## Hướng dẫn tự thực hiện (Bước 1: Repo, Dữ liệu & EDA)

### 1. Tạo Git repository và đẩy lên GitHub
Tại thư mục `c:\prj3`, chạy lần lượt:
```powershell
git init
git add .
git commit -m "Scaffold step 1: dataset utils, split generator, and Colab EDA notebook"
```
Sau đó tạo một repository trống trên GitHub (không tích chọn README hay `.gitignore` vì local đã có sẵn), rồi liên kết và đẩy lên:
```powershell
git branch -M main
git remote add origin https://github.com/<USERNAME>/<TEN_REPO>.git
git push -u origin main
```
> **Lưu ý:** File `.gitignore` đã được cấu hình để **chặn không đẩy** ảnh dữ liệu nặng (`data/raw/`), file nén (`*.zip`), trọng số mô hình (`*.pth`) và khóa bí mật (`kaggle.json`), nhưng **vẫn giữ lại** các file chia tập cố định trong `data/splits/*.csv`.

---

### 2. Tự tải dữ liệu CASIA-Iris-Thousand từ Kaggle
Truy cập Kaggle và tìm từ khóa `"CASIA-Iris-Thousand"` (bộ gồm 20.000 ảnh từ 1.000 người, dung lượng ~515 MB). Bạn có 2 lựa chọn:
- **Cách A (Khuyên dùng — Tải thủ công lên Google Drive):**
  1. Tải file `.zip` từ trang dataset Kaggle về máy tính.
  2. Upload file `.zip` đó lên Google Drive vào thư mục `MyDrive/prj3/CASIA-Iris-Thousand.zip`.
  3. Trong `notebooks/01_eda_and_splits.ipynb`, đặt `DOWNLOAD_MODE = "drive_zip"`.
- **Cách B (Dùng Kaggle API trực tiếp trên Colab):**
  1. Vào Kaggle -> **Account / Settings** -> **API** -> **Create New Token** để tải file `kaggle.json` (tuyệt đối giữ bí mật, không commit lên GitHub).
  2. Upload `kaggle.json` lên Colab (`/root/.config/kaggle/kaggle.json`) hoặc đặt `KAGGLE_USERNAME` / `KAGGLE_KEY` và giữ `DOWNLOAD_MODE = "kagglehub"`.

---

### 3. (Tùy chọn) Tạo môi trường ảo `.venv` trên Windows để chạy kiểm thử cục bộ
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

---

### 4. Chạy Notebook 01 trên Google Colab (Runtime CPU)
1. Mở [Google Colab](https://colab.research.google.com/), chọn tab **GitHub**, dán link repo của bạn và mở `notebooks/01_eda_and_splits.ipynb`.
2. Chọn **Runtime -> Change runtime type -> CPU** (bước EDA không cần GPU).
3. Cập nhật biến `REPO_URL` thành link GitHub repo của bạn ở cell đầu tiên rồi chạy toàn bộ notebook (**Runtime -> Run all**).
4. Kiểm tra các biểu đồ lưu tại `results/eda/` (trên Drive: `MyDrive/prj3/eda/`) và các file chia tập tại `data/splits/` (trên Drive: `MyDrive/prj3/splits/`).
5. Tải các file trong `MyDrive/prj3/splits/` về đặt vào `c:\prj3\data\splits\` và commit lên GitHub để đồng bộ cho Giai đoạn 2 (Tiền xử lý & Daugman Rubber Sheet).
