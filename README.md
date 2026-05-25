# Dự án dự báo doanh số chuỗi siêu thị Favorita
*(Time Series Forecasting: SARIMA vs XGBoost vs LightGBM vs CatBoost)*

## 1. Mô tả dự án
Dự án dự báo doanh số hằng ngày tại hệ thống siêu thị Favorita dựa trên dữ liệu lịch sử (2013–2017).
Mục tiêu là so sánh mô hình truyền thống (SARIMA) với các mô hình học máy hiện đại để đánh giá hiệu quả dự báo doanh số theo cấp độ **store × product family × day**.

## 2. Dữ liệu sử dụng
*Do dung lượng file dữ liệu lớn nên không được tải trực tiếp lên GitHub. Bạn có thể tiếp cận dữ liệu theo cách sau:*
* **Nguồn dữ liệu gốc:** https://www.kaggle.com/code/ryanholbrook/exercise-forecasting-with-machine-learning/input
*Tuy nhiên nhóm đã không sử dụng bộ dữ liệu test.csv (vì không có giá trị cho biến mục tiêu) và sử dụng bộ train.csv (được nhóm đổi tên thành train_test.csv) cho toàn bộ dự án. 
* **train_test.csv:** lịch sử doanh số theo ngày, từng cửa hàng và từng nhóm sản phẩm
* **transactions.csv:** số lượng giao dịch theo cửa hàng
* **oil.csv:** giá dầu thô WTI
* **holidays_events.csv:** lịch nghỉ lễ quốc gia/địa phương
* **stores.csv:** thông tin cửa hàng

## 3. Phương pháp & kỹ thuật

* Tiền xử lý dữ liệu, mã hóa nhãn, xử lý giá trị thiếu
* Tạo đặc trưng thời gian, độ trễ, trung bình trượt, biến giả ngày lễ, đặc trưng giao dịch và giá dầu
* Chia tập dữ liệu hợp lý theo mốc thời gian
* Huấn luyện và so sánh mô hình:
  * **SARIMA**
  * **XGBoost**
  * **LightGBM**
  * **CatBoost**
## 4. Đánh giá mô hình
Sử dụng các chỉ số:
* **RMSE**: Root Mean Squared Error
* **MAE**: Mean Absolute Error

| Model                | RMSE       | MAE       |
| -------------------- | ---------- | --------- |
| **XGBoost**          | **233.55** | **54.21** |
| LightGBM             | 236.00     | 54.51     |
| CatBoost             | 249.47     | 55.96     |
| SARIMA (daily total) | >100k      | >90k      |

**XGBoost cho kết quả tốt nhất**, đặc biệt ở cấp độ chi tiết store × family

## 5. Cách chạy dự án
### Cài đặt môi trường
```
pip install -r requirements.txt
```
### Chạy notebook
```
jupyter notebook
```
Mở file notebook và chạy theo từng bước: xử lý dữ liệu -> tạo đặc trưng -> huấn luyện -> đánh giá

## 6. File quan trọng

* `requirements.txt` - danh sách thư viện
* `models/` - chứa file mô hình đã lưu `.pkl`
* `notebook.ipynb` - toàn bộ quy trình và mã lệnh

## 7. Kết luận

* **SARIMA** phù hợp xu hướng + mùa vụ, nhưng giới hạn khi phải dự báo nhiều chuỗi con
* **Các mô hình Machine Learning** khai thác tốt biến động thực tế, tương tác đặc trưng, cho độ chính xác cao hơn
* **XGBoost** là lựa chọn tối ưu cho bài toán này


