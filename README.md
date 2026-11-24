# Nhận diện Mống mắt (Iris Recognition): Phương pháp Truyền thống và Học sâu

## 1. Tổng quan dự án

Dự án nghiên cứu, triển khai và so sánh hiệu năng giữa **phương pháp truyền thống** và **học sâu (Deep Learning)** cho bài toán nhận diện sinh trắc học mống mắt trên bộ dữ liệu **CASIA-Iris-Thousand**. Hệ thống giải quyết đồng thời hai bài toán chuẩn trong sinh trắc học:
- **Nhận dạng (`1:N` — Closed-set Identification):** Phân loại ảnh mống mắt vào đúng lớp danh tính đã đăng ký.
- **Xác thực (`1:1` — Open-set Verification):** Đối chiếu cặp hai ảnh mống mắt để kết luận cùng một mắt (*Genuine*) hay khác mắt (*Impostor*) trên tập đối tượng hoàn toàn mới chưa xuất hiện trong quá trình huấn luyện.

---

## 2. Bộ dữ liệu và Thiết kế thực nghiệm

### Thông số bộ dữ liệu (CASIA-Iris-Thousand)
- **Quy mô:** 1.000 đối tượng, mỗi đối tượng có ảnh mắt trái (`L`) và mắt phải (`R`) được xem là 2 lớp độc lập về mặt sinh học, tạo thành **2.000 lớp mắt** với **20.000 ảnh** (10 ảnh/mắt).
- **Đặc tính ảnh:** Ảnh thang xám chụp dưới ánh sáng hồng ngoại gần (NIR), độ phân giải `640 × 480`, thu thập bằng thiết bị IKEMB-100 dual-eye iris camera.
- **Thách thức nhiễu thực tế:** Đeo kính mắt, đốm phản xạ ánh sáng (specular reflection), che khuất do mí mắt/lông mi và ảnh mất nét.

### Giao thức chia tập (dùng chung cho mọi mô hình)
Để đảm bảo đánh giá công bằng và tái lập kết quả, dự án sinh một bộ chia cố định kèm kiểm tra tự động chống rò rỉ dữ liệu (`verify_no_leakage`):
- **Kịch bản Closed-set (`1:N`):** Phân tầng theo từng lớp mắt theo tỷ lệ `6 Train / 2 Val / 2 Test`; đồng thời gắn cờ tập con `in_subset_200_eyes` (100 đối tượng đầu, 200 lớp mắt) phục vụ đối chiếu với các mô hình ML truyền thống có chi phí tính toán cao (như RBF SVM).
- **Kịch bản Open-set (`1:1`):** Tách biệt hoàn toàn theo đối tượng (`700 Train / 150 Val / 150 Test`). Từ các tập `Val` và `Test` sinh cố định tệp cặp đối sánh `val_pairs.csv` và `test_pairs.csv` (mỗi tập gồm 3.000 cặp *Genuine* và 3.000 cặp *Impostor*, trong đó **25%** cặp mạo danh là **mắt trái và mắt phải của cùng một đối tượng** nhằm kiểm tra tính chặt chẽ của mô hình).

---

## 3. Phương pháp và Chỉ số đánh giá

- **Tiền xử lý:** Định vị đồng tử và vòng mống mắt, tạo mặt nạ nhiễu (noise mask), chuẩn hóa vùng vành khăn theo mô hình **Daugman Rubber Sheet**, tăng cường tương phản **CLAHE** và trích xuất vùng ảnh ROI.
- **Phương pháp truyền thống:**
  - Pipeline sinh trắc học cổ điển: Bộ lọc **1D Log-Gabor** tạo mã nhị phân **IrisCode** và đối sánh bằng **Hamming Distance** có dịch vòng.
  - Đặc trưng thủ công (**LBP**, **HOG**) kết hợp giảm chiều **PCA** và bộ phân loại học máy (**SVM**, **k-NN**).
- **Phương pháp Deep Learning:**
  - Học chuyển giao phân loại (`1:N`) với các mạng gọn nhẹ (**ResNet**, **MobileNetV3**, **EfficientNet**).
  - Học biểu diễn sâu (`1:1` Open-set & `1:N`) với **ArcFace Loss** và đối sánh **Cosine Similarity**.
- **Chỉ số đánh giá:**
  - Bài toán `1:N`: Rank-1 Accuracy, Rank-5 Accuracy, Macro F1-score, Confusion Matrix.
  - Bài toán `1:1`: **EER** (Equal Error Rate), **FAR**, **FRR**, **ROC-AUC**, **TAR @ FAR = 0.1% / 1%**, phân bố điểm *Genuine* vs *Impostor*.

---

## 4. Cấu trúc thư mục

```text
.
├── configs/
│   └── dataset_config.yaml      # Cấu hình tham số chia tập và ngưỡng chất lượng ảnh
├── data/
│   ├── raw/                     # Lưu trữ dữ liệu ảnh gốc (không theo dõi trên Git)
│   ├── processed/               # Dữ liệu sau tiền xử lý Rubber Sheet / ROI (không theo dõi trên Git)
│   └── splits/                  # Các tệp CSV phân chia tập và dataset_summary.json dùng chung
├── notebooks/
│   └── 01_eda_and_splits.ipynb  # Khảo sát EDA, trực quan hóa chất lượng ảnh và sinh bộ chia tập
├── results/                     # Biểu đồ phân tích, trọng số mô hình và bảng kết quả
├── scripts/
│   └── prepare_splits.py        # Script CLI quét dữ liệu, kiểm tra hợp lệ và tạo bộ chia tập
├── src/
│   ├── data/                    # Module quét dữ liệu, đo chỉ số chất lượng, chia tập & kiểm tra leakage
│   ├── preprocessing/           # Phân đoạn mống mắt và chuẩn hóa Rubber Sheet
│   ├── traditional/             # Log-Gabor IrisCode, đặc trưng thủ công và bộ phân loại ML
│   ├── deep_learning/           # Pipeline PyTorch, CNN Classification và ArcFace Metric Learning
│   └── evaluation/              # Tính toán các chỉ số sinh trắc học và vẽ đường cong ROC/DET
├── tests/
│   └── test_dataset_utils.py    # Bộ kiểm thử đơn vị cho module xử lý và chia tập dữ liệu
├── .gitignore
└── requirements.txt
```

---

## 5. Cài đặt và Thực thi

### Cài đặt phụ thuộc
```bash
pip install -r requirements.txt
```
*(PyTorch được mặc định sử dụng bản cài đặt sẵn có hỗ trợ CUDA trên Google Colab; khi chạy cục bộ có thể cài bổ sung `torch` và `torchvision` phù hợp với cấu hình phần cứng.)*

### Chuẩn bị dữ liệu và sinh bộ chia tập
- **Trên Google Colab:** Đặt tệp nén `CASIA-Iris-Thousand.zip` tại `MyDrive/prj3/CASIA-Iris-Thousand.zip` trên Google Drive, sau đó chạy toàn bộ [notebooks/01_eda_and_splits.ipynb](notebooks/01_eda_and_splits.ipynb). Notebook sẽ giải nén vào ổ cục bộ (`/content/data`), tạo biểu đồ EDA tại `results/eda/` và xuất các tệp phân chia tại `data/splits/`.
- **Qua dòng lệnh (CLI):** Giải nén bộ dữ liệu và thực thi:
  ```bash
  python scripts/prepare_splits.py --raw-dir <path_to_CASIA-Iris-Thousand>
  ```

### Kiểm thử đơn vị
```bash
python -m unittest discover -s tests -v
```
