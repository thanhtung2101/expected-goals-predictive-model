# expected-goals-predictive-model
# ⚽ Expected Goals (xG) Predictive Model: A Spatial & Probability Analysis

## 📌 Tổng quan dự án (Project Overview)
Dự án này ứng dụng Machine Learning để xây dựng mô hình dự báo **Bàn thắng kỳ vọng (Expected Goals - xG)**. Bản chất của bài toán là phân loại nhị phân (Binary Classification) nhằm đánh giá xác suất một cú sút chuyển hóa thành bàn thắng dựa trên các biến số về không gian, hình học và ngữ cảnh tình huống.

Tư duy thiết kế và phương pháp luận của dự án này hoàn toàn tương đồng với các bài toán đánh giá rủi ro/dự báo xác suất trong doanh nghiệp (như Credit Scoring, Probability of Default, hoặc Behavioral Scoring) – nơi việc chuyển đổi dữ liệu thô thành các đặc trưng có khả năng diễn giải cao là yếu tố quyết định.

## 🛠 Công nghệ & Thư viện (Tech Stack)
* **Ngôn ngữ:** R
* **Thu thập dữ liệu:** `StatsBombR` (Trích xuất dữ liệu Event-level qua API)
* **Xử lý dữ liệu:** `tidyverse`, `dplyr`
* **Machine Learning:** `tidymodels`, `parsnip`, `recipes`
* **Trực quan hóa:** `ggplot2`

---

## 🔬 Phương pháp triển khai (Methodology)

### 1. Thu thập và Tiền xử lý dữ liệu (Data Acquisition & Preprocessing)
* Kéo dữ liệu sự kiện (Event-level data) của giải đấu FIFA World Cup 2022 thông qua API miễn phí của StatsBomb.
* Làm phẳng dữ liệu JSON, lọc các bản ghi thuộc nhóm "Dứt điểm" (Shots) trong thời gian thi đấu chính thức.
* Định dạng biến mục tiêu `is_goal` thành Factor để chuẩn bị cho mô hình phân loại.

### 2. Trích xuất đặc trưng hình học (Feature Engineering)
Thay vì sử dụng trực tiếp tọa độ thô, không gian sân bóng được lượng hóa bằng toán học để tạo ra các biến số mang ý nghĩa thực tiễn cao:
* **Khoảng cách ($Distance$):** Sử dụng định lý Pytago để tính khoảng cách tuyệt đối từ điểm sút đến tâm khung thành.
* **Góc sút ($\theta$):** Áp dụng hàm lượng giác nghịch đảo để tính độ mở của góc sút dựa trên chiều rộng tiêu chuẩn của khung thành (8 yards). Góc càng rộng thể hiện vị trí sút càng trực diện.

### 3. Mô hình hóa (Probability Modeling)
* **Thuật toán:** Logistic Regression. Lựa chọn tối ưu để dự báo xác suất và đảm bảo tính minh bạch (White-box model), cho phép diễn giải rõ ràng tác động của từng biến số.
* **Tiền xử lý:** Sử dụng package `recipes` để One-Hot Encoding cho các biến phân loại (bộ phận dứt điểm) và thiết lập vai trò định danh (ID role) cho biến tọa độ.
* **Xử lý mất cân bằng:** Áp dụng phân tầng (Stratified Split) 80/20 để đảm bảo tỷ lệ bàn thắng/sút hỏng được giữ nguyên giữa tập Train và Test.

---

## 📊 Kết quả & Đánh giá (Results & Evaluation)

### 1. Đánh giá sức mạnh phân tách (ROC-AUC)
Do tính chất mất cân bằng (Imbalanced Data) của dữ liệu bóng đá, chỉ số ROC-AUC được sử dụng làm thước đo chính.
* **Chỉ số ROC-AUC:** Đạt mức **~0.8558** (sau khi cấu hình `event_level = "second"` để định vị đúng Positive Class).
* **Diễn giải:** Mô hình có khả năng phân tách xuất sắc, phân biệt chính xác giữa một cơ hội nguy hiểm thực sự và một pha dứt điểm cầu may trong hơn 85% trường hợp.

### 2. Giải mã Biến số (Feature Interpretation)
Kết quả trích xuất hệ số mô hình xác nhận thuật toán đã học đúng quy luật không gian:
<img width="971" height="281" alt="image" src="https://github.com/user-attachments/assets/6f417ed2-cdd0-4a41-a599-29bbd93357d1" />

* **Distance (Estimate < 0, p-value < 0.05):** Khoảng cách càng xa, xác suất ghi bàn (log-odds) càng giảm mạnh.
* **Angle (Estimate > 0, p-value < 0.05):** Góc sút càng rộng (càng trực diện), tỷ lệ chuyển hóa thành bàn thắng càng cao.

---

## 🗺 Trực quan hóa dữ liệu (Spatial Visualization)

<img width="3600" height="2400" alt="WorldCup_xG_ShotMap" src="https://github.com/user-attachments/assets/b148cd9e-e281-43e5-aff7-621cfc08ef98" />


Bản đồ phân tích không gian (Shot Map) được xây dựng thể hiện phân phối của các dự đoán xG:
* **Cụm điểm nóng:** Các điểm sút có xG cao (màu đỏ, kích thước lớn) tập trung chủ yếu trong khu vực cấm địa (18-yard box).
* **Kiểm định thực tế:** Hầu hết các pha ghi bàn thực tế đều trùng khớp với các tọa độ có xG dự đoán cao, chứng minh độ tin cậy của mô hình khi ứng dụng vào đánh giá hiệu suất.

---

## 💡 Kết luận (Conclusion)
Dự án thể hiện khả năng làm chủ End-to-End Data Pipeline: từ trích xuất API, Feature Engineering bằng tư duy toán học, xây dựng và tinh chỉnh mô hình Machine Learning, cho đến trực quan hóa dữ liệu không gian. Những kỹ năng này là nền tảng để giải quyết các bài toán phân loại dữ liệu, đánh giá rủi ro và tối ưu hóa quyết định kinh doanh.

---
