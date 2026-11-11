# 🏠 Dự đoán giá nhà AI bằng SVM (SVR) và phân cụm K-means

Ứng dụng Streamlit cho phép:

- **SVM (SVR):** huấn luyện mô hình hồi quy để ước lượng giá nhà.
- **K-means:** phân nhóm các căn nhà có thuộc tính tương đồng; xem biểu đồ Elbow (WCSS), biểu đồ cụm và thử gán cụm cho một căn nhà mới.
- **Khám phá dữ liệu:** xem trước, thống kê mô tả và tải dữ liệu CSV.

> Lưu ý học thuật: SVR là thuật toán dự đoán giá (bài toán hồi quy). K-means là thuật toán học không giám sát dùng để phân cụm, không trực tiếp dự đoán giá.

## 1. Cấu trúc thư mục

```text
house-price-streamlit/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

## 2. Dữ liệu đầu vào

- Chưa tải CSV: ứng dụng tự dùng **650 dòng dữ liệu demo giả lập** để bạn kiểm tra các chức năng.
- Dùng dữ liệu Kaggle House Prices: tải tệp `train.csv` lên ứng dụng. Cột mục tiêu mặc định là `SalePrice` nếu có.
- Với dữ liệu khác: tệp CSV cần có ít nhất 10 dòng và một cột giá dạng số. Chọn cột giá ở thanh bên.
- Không cần đưa dữ liệu cá nhân hoặc tệp CSV lớn lên GitHub. Có thể tải CSV trực tiếp trong giao diện ứng dụng.

Các thuộc tính gợi ý cho bộ Kaggle gồm `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `YearBuilt`, `LotArea` nếu các cột này tồn tại.

## 3. Chạy thử trên máy tính

Cài Python 3.10 trở lên, mở Terminal/Command Prompt tại thư mục dự án và chạy:

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo:

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

Cài thư viện và chạy ứng dụng:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Trình duyệt sẽ mở ứng dụng Streamlit. Nếu chưa tự mở, truy cập địa chỉ `http://localhost:8501`.

## 4. Đưa project lên GitHub

1. Đăng nhập GitHub và chọn **New repository**.
2. Đặt tên, ví dụ `house-price-streamlit`, chọn Public hoặc Private rồi tạo repository.
3. Trong repository, chọn **Add file → Upload files**.
4. Tải lên `app.py`, `requirements.txt`, `README.md`, `.gitignore` và thư mục `.streamlit` (bên trong có `config.toml`).
5. Chọn **Commit changes**.

Có thể dùng Git trên máy tính thay vì Upload files:

```bash
git init
git add .
git commit -m "Create house price Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/house-price-streamlit.git
git push -u origin main
```

Thay `YOUR_USERNAME` bằng tên tài khoản GitHub của bạn.

## 5. Triển khai bằng Streamlit Community Cloud

1. Mở [Streamlit Community Cloud](https://share.streamlit.io/) và đăng nhập bằng GitHub.
2. Chọn **Create app**.
3. Chọn repository `house-price-streamlit`, branch `main` và file chính `app.py`.
4. Chọn **Deploy**.
5. Sau khi triển khai thành công, Streamlit cung cấp một đường dẫn dạng `https://...streamlit.app` để mở ứng dụng.

Khi cập nhật mã nguồn và commit lên GitHub, Community Cloud sẽ nhận thay đổi từ repository và cập nhật ứng dụng.

## 6. Cách đọc kết quả

### SVM / SVR

Dữ liệu được chia thành tập huấn luyện và kiểm tra. Pipeline xử lý giá trị thiếu, chuẩn hóa biến số, mã hóa biến phân loại rồi huấn luyện SVR.

- **MAE:** sai số tuyệt đối trung bình.
- **MSE:** bình phương sai số trung bình.
- **RMSE:** căn bậc hai của MSE, cùng đơn vị với giá nhà.
- **R²:** mức độ giải thích biến thiên của giá trên tập kiểm tra. R² có thể âm nếu mô hình dự đoán kém hơn mốc dự đoán trung bình.

### K-means

Dữ liệu đầu vào được điền giá trị thiếu và chuẩn hóa trước khi phân cụm. Biểu đồ Elbow hiển thị WCSS (inertia) theo K. Dùng điểm gãy làm gợi ý, sau đó xem quy mô và giá trung bình của từng cụm để diễn giải.

**Không so sánh K-means với SVR bằng MAE/RMSE/R² như hai mô hình dự đoán cùng loại.** Hai thuật toán trả lời hai câu hỏi khác nhau.
