# GUI Project 1 · Nhà Insight

Ứng dụng Streamlit riêng cho **Dự đoán giá nhà & Phát hiện giá bất thường**, dựa trên yêu cầu trong `GUI/` và mẫu `GUI_ITViec/demo_streamlit_v3.py`. Không sửa hoặc phụ thuộc mã nguồn còn thiếu trong `Project 1/run_pipeline.py`.

## Chạy trên máy

Mở terminal trong thư mục `GUI_Project1`:

### Trên macOS / Linux:
```bash
bash setup.sh
# Hoặc chạy trực tiếp: python3 -m streamlit run app.py
```

### Trên Windows (1 lệnh duy nhất):
```cmd
python -m streamlit run app.py
# Hoặc click đúp file: setup.bat
```

Ứng dụng sẽ tự động mở tại địa chỉ: `http://localhost:8501`. File `setup.sh` (macOS/Linux) và `setup.bat` (Windows) sẽ tự kiểm tra môi trường: nếu máy đã có sẵn các thư viện thì chạy ngay, nếu thiếu sẽ tự động cài đặt từ `requirements.txt`.

## Các màn hình

1. **Tổng quan**: mục tiêu, phạm vi, chỉ số mô hình và điều hướng.
2. **Dự đoán một căn nhà**: quận, loại nhà, diện tích, phòng, tầng, pháp lý, nội thất, hướng; thông tin nâng cao và tải kết quả.
3. **Dự đoán từ CSV**: tải tệp không cần giá, xem trước, kiểm tra dữ liệu, dự đoán, biểu đồ, tải CSV UTF-8 BOM.
4. **Phát hiện bất thường**: chọn cột giá và đơn vị, 5 phương pháp, chỉnh ngưỡng/trọng số, lọc tin gắn cờ, tải toàn bộ kết quả.
5. **Mô hình & hướng dẫn**: chỉ số kiểm thử thực, schema và giải thích phương pháp/giới hạn.

Các mẫu trên giao diện là dữ liệu **giả định**, không phải tin rao. Mẫu có giá: `samples/nha_co_gia.csv`; mẫu không giá tải trực tiếp từ trang dự đoán CSV. Không nhập giá giả vào kết quả: dự đoán luôn chạy mô hình đã huấn luyện.

## Mô hình và dữ liệu thật

- Nguồn: `Project 1/.../data/model_ready/nha_o/spark/`.
- RandomForestRegressionModel: 50 cây, 7.692 đặc trưng, 24.930 dòng train / 6.241 dòng test.
- Mục tiêu: `ln(price_billion)`; giá xuất ra = `exp(prediction)`, đơn vị tỷ đồng.
- `assets/forest.json.gz`: cấu trúc cây, median/mode imputer, StringIndexer, OneHotEncoder và StandardScaler đã fit. Không pickle, không cần Spark/Java khi chạy GUI.
- `assets/reference.parquet`: đặc trưng tổng quát giải mã từ vector train để đối chiếu bất thường; không kèm mã tin, số điện thoại hay địa chỉ cụ thể.
- `assets/metrics.json`: chỉ số tính lại và SHA256 artifact nguồn. R² log ≈ 0.836554, RMSE log ≈ 0.308229, khớp số làm tròn của notebook. Không dùng R² 0.8850 của GBT sklearn vì model đó không có trong gói bàn giao.
- Kiểm thử round-trip xác nhận tái mã hóa 100 vector kiểm thử từ dữ liệu đã giải mã; bộ xuất xác nhận lại R²/RMSE trên toàn bộ 6.241 vector test. Chưa so sánh trực tiếp với một JVM Spark đang chạy.

Đóng gói lại khi có đầy đủ Project 1 cạnh thư mục GUI:

```bash
python export_assets.py
# Hoặc: python export_assets.py --source /duong/dan/toi/spark
```

Script ghi đè **chỉ các asset sinh ra trong GUI_Project1/assets**, không huấn luyện lại, không thay đổi nguồn. Chỉ hỗ trợ cấu trúc artifact Random Forest hiện tại, không phải bộ chuyển đổi Spark tổng quát.

## CSV và dữ liệu thiếu

Bắt buộc `district_name` (đúng tên quận cũ trong tập train) và `size` (8–10.000 m²). Tên cột tiếng Việt tương thích mẫu như `quan_huyen`, `dien_tich_dat`, `so_phong_ngu` cũng được nhận. Xem bảng trên giao diện. Chỉ hỗ trợ `property_type=nha_o`.

- UTF-8/UTF-8 BOM; dấu phẩy phân cách; dấu chấm thập phân; không dùng dấu phân cách hàng nghìn.
- Tối đa 5.000 dòng / 10 MB; dòng sai được báo và từ chối cả lần chạy, không âm thầm bỏ dòng.
- Giá `price` mặc định VNĐ, `price_billion`/`gia_ban` mặc định tỷ đồng, `price_per_m2`/`gia_m2`/`don_gia` mặc định triệu/m². **Luôn xác nhận đơn vị trên GUI.** Có thể chọn cột giá tên khác.
- Biến thiếu điền bằng thống kê train. Categorical chưa gặp dùng mã unknown đúng cơ chế Spark; GUI cảnh báo. Các cột giá thực không đi vào mô hình giá.
- `address` và `body` chỉ là thông tin kèm theo; không tự suy ra phường, tọa độ, pháp lý hay đặc trưng NLP từ văn bản. `ward_id`/`street_name` cần đúng mã/tên nguồn nếu có.
- Các biến nâng cao tiền xử lý như `area_chart_min/max`, `area_yoy_change_pct`, `char_*`, `txt_*` có thể truyền qua CSV. Các biến chưa cung cấp không được tự suy diễn từ mô tả.

## Bất thường: phạm vi và giới hạn

Đây là **bản tích hợp bổ sung cho GUI**, không tái lập chính xác mô hình bất thường sklearn trong notebook vì không có các artifact đó.

- Residual-z dùng mean/std residual **in-sample train** của Random Forest; cần calibration độc lập trước triển khai thực tế. S1 = min(|z|/5, 1).
- Min/Max dùng P1–P99 đơn giá train, thay thế nguồn biểu đồ giá khu vực 13 tháng chưa đi kèm. S2 là cờ 0/1.
- P10–P90 theo quận × loại nhà (ít nhất 30 dòng), rồi lùi lên quận/toàn train. S3 = khoảng cách ngoài khoảng / trung vị, chặn ở 1.
- Isolation Forest fit trên train: log diện tích, số phòng, tầng, log đơn giá. S4 = phân vị điểm so với train; thanh ngưỡng top % áp dụng phân phối train, không ép tỷ lệ bất thường của tệp tải lên.
- Composite = 100 × tổng S1–S4 theo trọng số chuẩn hóa. Mặc định 25% mỗi tín hiệu, ngưỡng 60; chưa có đánh giá Precision/Recall cho cấu hình này.

Không có xác suất/độ tin cậy dự đoán đã hiệu chỉnh. Giá là ước lượng từ giá rao, không phải giá giao dịch. Gắn cờ không chứng minh tin sai hoặc lừa đảo. Phạm vi TP.HCM theo địa giới cũ, không gồm Bình Dương/Bà Rịa–Vũng Tàu cũ. Dữ liệu nhập tay ít biến có thể cho độ chính xác thấp hơn kiểm thử đủ đặc trưng.

## Triển khai

Có thể copy **riêng GUI_Project1** thành repository rồi chọn `app.py` làm entrypoint trên Streamlit Community Cloud, Python 3.11. Giữ đủ `assets/`, `samples/`, `requirements.txt`, `style.css`, `.streamlit/config.toml`; không đưa `.venv` lên Git. Nếu đưa toàn workspace, chọn `GUI_Project1/app.py` và kiểm tra dependency/config theo working directory của host.

Đã cung cấp `Procfile`, `setup.sh` theo cấu trúc bài mẫu; Streamlit Cloud không cần chạy hai file này. Với host kiểu Procfile, đặt working directory là GUI_Project1. Chưa publish website hoặc tạo link công khai.

Ứng dụng không chủ động lưu CSV tải lên; kết quả ở bộ nhớ phiên. Tệp mô hình chỉ nạp từ asset cục bộ, không cho upload pickle. Các chuỗi dễ thành công thức Excel được escape khi tải CSV. Khi publish, đơn vị triển khai cần xác định chính sách riêng tư, quyền truy cập và quy định của host.

## Tài liệu đối chiếu kỹ thuật

- [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
- [Spark RandomForestRegressor (nguồn Apache)](https://github.com/apache/spark/blob/v4.0.1/mllib/src/main/scala/org/apache/spark/ml/regression/RandomForestRegressor.scala)

Thiết kế theo skill frontend-design: bố cục Streamlit bám bài mẫu, màu xanh dịu, giao diện responsive, kiểm tra luồng thao tác và báo rõ giới hạn mô hình.
