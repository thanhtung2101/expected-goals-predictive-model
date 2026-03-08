# =========================================================================================
# DỰ ÁN MÔ HÌNH HÓA XÁC SUẤT: DỰ ĐOÁN BÀN THẮNG KỲ VỌNG (EXPECTED GOALS - xG)
# =========================================================================================
# Mục tiêu: Xây dựng mô hình Machine Learning (Logistic Regression) để dự báo xác suất 
#           một cú sút chuyển hóa thành bàn thắng dựa trên dữ liệu không gian và tình huống.
# Tư duy phân tích: Bài toán phân loại nhị phân (Binary Classification) này sử dụng logic 
#                   tương đồng với các mô hình chấm điểm rủi ro (Risk Scoring) hoặc 
#                   dự báo xác suất vỡ nợ (Probability of Default).
# Công cụ sử dụng: R (tidyverse, tidymodels, StatsBombR, ggplot2)
# =========================================================================================

# -----------------------------------------------------------------------------------------
# PHẦN 1: THIẾT LẬP MÔI TRƯỜNG VÀ THU THẬP DỮ LIỆU (DATA ACQUISITION)
# -----------------------------------------------------------------------------------------
# Cài đặt thư viện nếu chưa có:
# install.packages(c("devtools", "tidyverse", "tidymodels", "ggplot2"))
# devtools::install_github("statsbomb/SDMTools")
# devtools::install_github("statsbomb/StatsBombR")

library(StatsBombR)
library(tidyverse)
library(tidymodels)
library(ggplot2)

# 1.1 Gọi API từ StatsBomb để lấy danh sách giải đấu miễn phí
comps <- FreeCompetitions()

# 1.2 Lọc lấy dữ liệu của FIFA World Cup 2022 (competition_id = 43, season_id = 106)
wc_comp <- comps %>% filter(competition_id == 43 & season_id == 106)

# 1.3 Lấy danh sách toàn bộ các trận đấu thuộc giải đấu này
wc_matches <- FreeMatches(wc_comp)

# 1.4 Tải dữ liệu chi tiết cấp độ sự kiện (Event-level data)
# Quá trình này xử lý lượng dữ liệu lớn (JSON format) và chuyển về dạng bảng
wc_events <- free_allevents(MatchesDF = wc_matches, Parallel = TRUE)

# 1.5 Làm sạch và chuẩn hóa dữ liệu sơ bộ bằng hàm nội bộ của StatsBombR
wc_events_clean <- allclean(wc_events)


# -----------------------------------------------------------------------------------------
# PHẦN 2: TIỀN XỬ LÝ DỮ LIỆU & TRÍCH XUẤT ĐẶC TRƯNG (FEATURE ENGINEERING)
# -----------------------------------------------------------------------------------------

# 2.1 Lọc dữ liệu mục tiêu: Chỉ lấy các tình huống Dứt điểm (Shots) trong thời gian thi đấu
shots_data <- wc_events_clean %>%
  filter(type.name == "Shot" & period != 5) %>% # period != 5 để loại bỏ loạt sút luân lưu
  select(
    id, match_id, player.name, team.name, 
    location.x, location.y, # Tọa độ x, y của điểm sút trên sân
    shot.type.name,         # Loại tình huống (Bóng sống, phạt trực tiếp...)
    shot.body_part.name,    # Bộ phận dứt điểm (Đầu, chân trái, chân phải...)
    shot.outcome.name       # Kết quả cú sút (Bàn thắng, Bị cản phá, Ra ngoài...)
  ) %>%
  # Tạo biến mục tiêu (Target Variable) cho mô hình phân loại: 1 (Bàn thắng), 0 (Không vào)
  mutate(is_goal = ifelse(shot.outcome.name == "Goal", 1, 0))

# 2.2 Feature Engineering: Ứng dụng hình học để lượng hóa không gian
# Tâm khung thành đối phương được quy ước tại tọa độ (x = 120, y = 40)
shots_features <- shots_data %>%
  filter(!is.na(location.x) & !is.na(location.y)) %>% # Xóa các record thiếu tọa độ
  mutate(
    # Tính Khoảng cách (Distance) áp dụng định lý Pytago
    distance = sqrt((120 - location.x)^2 + (40 - location.y)^2),
    
    # Tính Góc sút (Shot Angle) áp dụng lượng giác (khung thành rộng 8 yards)
    angle_rad = atan((8 * (120 - location.x)) / ((120 - location.x)^2 + (location.y - 40)^2 - 16)),
    
    # Xử lý các trường hợp ngoại lệ (sút từ vạch biên ngang) để tránh lỗi góc âm
    angle_rad = ifelse(angle_rad < 0, angle_rad + pi, angle_rad),
    
    # Chuyển đổi từ Radian sang Độ (Degrees) để dễ diễn giải kết quả kinh doanh
    angle_degree = angle_rad * (180 / pi)
  )


# -----------------------------------------------------------------------------------------
# PHẦN 3: XÂY DỰNG MÔ HÌNH HỌC MÁY (PREDICTIVE MODELING)
# -----------------------------------------------------------------------------------------

# 3.1 Chuẩn bị dữ liệu cuối cùng cho mô hình
model_data <- shots_features %>%
  select(is_goal, distance, angle_degree, shot.body_part.name, location.x, location.y) %>%
  mutate(
    is_goal = as.factor(is_goal), # Ép kiểu factor cho bài toán Classification
    shot.body_part.name = as.factor(shot.body_part.name) 
  ) %>%
  drop_na()

# 3.2 Phân chia dữ liệu Huấn luyện & Kiểm thử (Train/Test Split)
set.seed(2026) # Cố định seed đảm bảo tính lặp lại (reproducibility)
# Phân tầng (strata) theo is_goal để đảm bảo tỷ lệ bàn thắng cân bằng giữa 2 tập
data_split <- initial_split(model_data, prop = 0.80, strata = is_goal)

train_data <- training(data_split)
test_data  <- testing(data_split)

# 3.3 Khởi tạo Pipeline Tiền xử lý (Recipe)
xg_rec <- recipe(is_goal ~ ., data = train_data) %>%
  # Gắn cờ tọa độ x, y là "ID" để giữ lại vẽ biểu đồ, nhưng KHÔNG đưa vào thuật toán dự báo
  update_role(location.x, location.y, new_role = "ID") %>% 
  # Biến đổi One-Hot Encoding cho các biến phân loại (Categorical variables)
  step_dummy(all_nominal_predictors()) 

# 3.4 Thiết lập thuật toán Logistic Regression
log_spec <- logistic_reg() %>% 
  set_engine("glm") %>% 
  set_mode("classification")

# 3.5 Đóng gói quy trình (Workflow) và Huấn luyện mô hình
xg_workflow <- workflow() %>%
  add_recipe(xg_rec) %>%
  add_model(log_spec)

xg_fit <- xg_workflow %>% fit(data = train_data)

# 3.6 Trích xuất và in ra các hệ số của mô hình (Coefficients)
model_coeffs <- tidy(xg_fit)
print(model_coeffs)


# -----------------------------------------------------------------------------------------
# PHẦN 4: ĐÁNH GIÁ MÔ HÌNH (MODEL EVALUATION)
# -----------------------------------------------------------------------------------------

# 4.1 Thực hiện dự đoán xác suất trên tập Test
predictions <- predict(xg_fit, new_data = test_data, type = "prob") %>%
  bind_cols(test_data)

# 4.2 Đánh giá hiệu suất phân loại bằng chỉ số ROC-AUC
# Phù hợp cho dữ liệu mất cân bằng (Imbalanced data) như bóng đá hay rủi ro tín dụng
# 4.2 Đánh giá hiệu suất phân loại bằng chỉ số ROC-AUC
roc_auc_score <- roc_auc(predictions, truth = is_goal, .pred_1, event_level = "second")

cat("Chỉ số ROC-AUC của mô hình:", round(roc_auc_score$.estimate, 4), "\n")

# -----------------------------------------------------------------------------------------
# PHẦN 5: TRỰC QUAN HÓA DỮ LIỆU (DATA VISUALIZATION) - BẢN ĐỒ SÚT BÓNG (SHOT MAP)
# -----------------------------------------------------------------------------------------

# 5.1 Xây dựng nền sơ đồ sân bóng (Pitch Layout) theo chuẩn kích thước StatsBomb
xmin <- 0; xmax <- 120; ymin <- 0; ymax <- 80

pitch_base <- ggplot() +
  # Vẽ mặt cỏ, vạch biên và vạch giữa sân
  geom_rect(aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), fill="darkgreen", color="white", linewidth=1) +
  geom_segment(aes(x=xmax/2, xend=xmax/2, y=ymin, yend=ymax), color="white", linewidth=1) +
  geom_point(aes(x=xmax/2, y=ymax/2), color="white", size=3) +
  geom_curve(aes(x=xmax/2, xend=xmax/2, y=ymax/2-10, yend=ymax/2+10), color="white", linewidth=1, curvature = 0.5) +
  # Vẽ khu vực cấm địa (18-yard box)
  geom_rect(aes(xmin=xmin, xmax=18, ymin=18, ymax=62), fill="transparent", color="white", linewidth=1) +
  geom_rect(aes(xmin=xmax-18, xmax=xmax, ymin=18, ymax=62), fill="transparent", color="white", linewidth=1) +
  # Vẽ khu vực 5m50 (6-yard box)
  geom_rect(aes(xmin=xmin, xmax=6, ymin=30, ymax=50), fill="transparent", color="white", linewidth=1) +
  geom_rect(aes(xmin=xmax-6, xmax=xmax, ymin=30, ymax=50), fill="transparent", color="white", linewidth=1) +
  # Vẽ khung thành
  geom_segment(aes(x=xmin, xend=xmin, y=36, yend=44), linewidth=2, color="white") +
  geom_segment(aes(x=xmax, xend=xmax, y=36, yend=44), linewidth=2, color="white") +
  # Vẽ chấm phạt đền (Penalty spots)
  geom_point(aes(x=12, y=40), color="white", size=2) +
  geom_point(aes(x=xmax-12, y=40), color="white", size=2) +
  # Định dạng hệ tọa độ không gian
  scale_x_continuous(limits=c(xmin-5, xmax+5), expand=c(0,0)) +
  scale_y_continuous(limits=c(ymin-5, ymax+5), expand=c(0,0)) +
  coord_fixed(ratio = (ymax-ymin)/(xmax-xmin)) + 
  theme_void()

# 5.2 Lớp phủ (Overlay) kết quả dự báo lên sơ đồ sân bóng
final_shot_map <- pitch_base +
  # Ánh xạ tọa độ, xác suất (xG) thành kích thước/màu sắc, và kết quả thực tế thành hình dạng
  geom_point(data = predictions, aes(x = location.x, y = location.y, color = .pred_1, size = .pred_1, shape = is_goal), alpha = 0.7) +
  scale_color_gradientn(colors = c("blue", "yellow", "red"), name = "xG Dự đoán") +
  scale_size_continuous(range = c(2, 12), name = "xG (Kích thước)") +
  scale_shape_manual(values = c("0" = 1, "1" = 19), labels = c("0" = "Không bàn thắng", "1" = "Bàn thắng"), name = "Kết quả") +
  labs(
    title = "Bản đồ phân tích không gian: Cú sút & Bàn thắng kỳ vọng (xG)",
    subtitle = "World Cup 2022 - Logistic Regression Probability Model"
  ) +
  theme(
    plot.title = element_text(face="bold", size=18, hjust=0.5, margin = margin(b=10)),
    plot.subtitle = element_text(size=14, hjust=0.5, color="gray30"),
    legend.position = "right",
    plot.background = element_rect(fill = "white", color = NA) # Đảm bảo nền trắng khi xuất file
  )

# Hiển thị bản đồ
print(final_shot_map)

