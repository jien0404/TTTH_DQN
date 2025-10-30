# Báo Cáo Học Sâu Gia Cường: Dueling DQN với DDQN cho Điều Hướng Robot

## 1. Giới Thiệu
Báo cáo này trình bày chi tiết việc triển khai một tác nhân Học Sâu Gia Cường (Deep Reinforcement Learning - DRL) sử dụng **Dueling Deep Q-Network (Dueling DQN)** kết hợp với **Double DQN (DDQN)** cho nhiệm vụ điều hướng robot. Tác nhân được thiết kế để di chuyển trong một môi trường lưới hướng tới mục tiêu xác định trong khi tránh các chướng ngại vật. Triển khai sử dụng PyTorch để xây dựng mô hình mạng nơ-ron và tích hợp các kỹ thuật tiên tiến như Experience Replay, hàm phần thưởng tùy chỉnh, và các chiến lược khám phá để nâng cao hiệu quả học tập.

## 2. Tổng Quan Thuật Toán
Thuật toán được triển khai tích hợp **Dueling DQN**, **DDQN**, và một hàm phần thưởng tùy chỉnh để giải quyết các thách thức trong điều hướng robot với không gian hành động rời rạc. Dưới đây là phân tích các thành phần chính:

### 2.1 Kiến Trúc Dueling DQN
Kiến trúc Dueling DQN chia việc ước lượng giá trị Q thành hai nhánh:
- **Nhánh Giá Trị (Value Stream)**: Ước lượng giá trị trạng thái \( V(s) \), biểu thị tổng phần thưởng kỳ vọng khi ở trạng thái \( s \).
- **Nhánh Lợi Thế (Advantage Stream)**: Ước lượng lợi thế \( A(s, a) \) cho mỗi hành động \( a \), thể hiện lợi ích tương đối khi thực hiện hành động \( a \) trong trạng thái \( s \).

Giá trị Q được tính theo công thức:
\[
Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a') \right)
\]
Công thức này cho phép mạng học riêng giá trị trạng thái và lợi thế hành động, cải thiện độ ổn định và hiệu quả học tập. Mạng bao gồm:
- Một lớp trích xuất đặc trưng chung (hai lớp kết nối đầy đủ với 64 đơn vị, kích hoạt ReLU).
- Nhánh giá trị (một lớp kết nối đầy đủ trả về một giá trị duy nhất).
- Nhánh lợi thế (một lớp kết nối đầy đủ trả về giá trị cho mỗi hành động).

### 2.2 Double DQN (DDQN)
Để giảm thiểu thiên kiến ước lượng quá mức trong DQN truyền thống, **DDQN** được sử dụng. DDQN tách biệt việc chọn và đánh giá hành động:
- Mạng Q chính chọn hành động tốt nhất cho trạng thái tiếp theo.
- Mạng Q mục tiêu đánh giá giá trị Q của hành động đã chọn.

Giá trị Q mục tiêu được tính như sau:
\[
Q_{\text{target}} = r + \gamma \cdot Q_{\text{target}}(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)
\]
Trong đó:
- \( r \): Phần thưởng nhận được.
- \( \gamma \): Hệ số chiết khấu (đặt là 0.99).
- \( s' \): Trạng thái tiếp theo.
- \( \theta \): Tham số của mạng Q chính.
- \( \theta^- \): Tham số của mạng Q mục tiêu, được cập nhật định kỳ.

### 2.3 Biểu Diễn Trạng Thái
Trạng thái là một vector 26 chiều bao gồm:
- Một lưới 5x5 được làm phẳng (25 chiều) tập trung vào robot, mã hóa sự hiện diện của chướng ngại vật và mục tiêu.
- Khoảng cách Euclidean đến mục tiêu (1 chiều).

Biểu diễn này ghi nhận thông tin môi trường cục bộ và tiến độ của tác nhân hướng tới mục tiêu.

### 2.4 Không Gian Hành Động
Không gian hành động bao gồm 8 hướng rời rạc:
- Đông (1, 0)
- Đông Bắc (1, 1)
- Bắc (0, 1)
- Tây Bắc (-1, 1)
- Tây (-1, 0)
- Tây Nam (-1, -1)
- Nam (0, -1)
- Đông Nam (1, -1)

### 2.5 Hàm Phần Thưởng
Hàm phần thưởng được thiết kế để hướng dẫn tác nhân tới mục tiêu, tránh chướng ngại vật và hạn chế hành vi lặp lại:
- **Đạt Mục Tiêu**: +100 phần thưởng cộng với phần thưởng tò mò (curiosity).
- **Va Chạm Chướng Ngại Vật**: -50 phạt.
- **Tiến Gần Mục Tiêu**: Tỷ lệ với sự giảm khoảng cách tới mục tiêu (\( 10 \cdot (\text{khoảng_cách_trước} - \text{khoảng_cách}) \)).
- **Phạt Bước Đi**: -0.1 để khuyến khích đường đi ngắn nhất.
- **Phạt Lặp Lại**: Lên đến -2 cho mỗi lần lặp lại vị trí, dựa trên tần suất thăm.
- **Phần Thưởng Tò Mò**: \( \frac{0.1}{\sqrt{\text{số_lần_thăm}}} \) để khuyến khích khám phá các vị trí ít được thăm.
- **Phạt Khoảng Cách**: \( -0.05 \cdot \text{khoảng_cách_tới_mục_tiêu} \) để phạt khi ở xa mục tiêu.

Hàm phần thưởng cân bằng giữa hành vi hướng tới mục tiêu, khám phá, và hiệu quả, đồng thời ngăn chặn va chạm và chuyển động lặp lại.

### 2.6 Chiến Lược Khám Phá
Một chiến lược **epsilon-greedy** được sử dụng để khám phá:
- **Huấn Luyện**: \( \epsilon \) bắt đầu từ 1.0 và giảm theo hệ số 0.998 mỗi bước, với giá trị tối thiểu là 0.02.
- **Kiểm Tra**: \( \epsilon = 0.0 \), chọn hành động tham lam dựa trên giá trị Q.

### 2.7 Experience Replay
Một bộ đệm phát lại (dung lượng: 20.000) lưu trữ các chuyển tiếp (\( s, a, r, s', \text{done} \)). Các lô nhỏ kích thước 64 được lấy mẫu ngẫu nhiên để huấn luyện mạng Q, cải thiện hiệu quả mẫu và giảm tương quan giữa các cập nhật.

### 2.8 Quá Trình Huấn Luyện
- **Bộ Tối Ưu Hóa**: Adam với tốc độ học 0.0005.
- **Hàm Mất Mát**: Sai số bình phương trung bình (MSE) giữa giá trị Q dự đoán và mục tiêu.
- **Cập Nhật Mạng Mục Tiêu**: Cứ sau 200 bước, mạng mục tiêu được cập nhật với tham số của mạng Q chính.
- **Số Liệu**: Phần thưởng tập, độ dài tập, và giá trị Q trung bình được theo dõi để giám sát hiệu suất.

## 3. Chi Tiết Triển Khai
Triển khai được đóng gói trong lớp `Controller`, quản lý:
- **Khởi Tạo**: Thiết lập Dueling DQN, mạng mục tiêu, bộ tối ưu hóa, và bộ đệm phát lại.
- **Ra Quyết Định**: Sử dụng epsilon-greedy để chọn hành động dựa trên trạng thái hiện tại.
- **Huấn Luyện**: Lấy mẫu lô nhỏ, tính toán mục tiêu DDQN, và cập nhật mạng Q.
- **Tính Toán Phần Thưởng**: Tính phần thưởng dựa trên độ gần mục tiêu, va chạm, và khám phá.
- **Lưu Trữ Mô Hình**: Lưu và tải trọng số mô hình cho huấn luyện và kiểm tra.

Mã sử dụng PyTorch cho các hoạt động mạng nơ-ron và chạy trên CPU (CUDA là tùy chọn). Môi trường là một lưới 32x32 với kích thước ô và đệm do người dùng xác định.

## 4. Các Tính Năng Chính
- **Dueling DQN**: Cải thiện ước lượng giá trị Q bằng cách tách biệt việc học giá trị trạng thái và lợi thế hành động.
- **DDQN**: Giảm thiên kiến ước lượng quá mức, tăng cường độ ổn định huấn luyện.
- **Hàm Phần Thưởng Tùy Chỉnh**: Khuyến khích hành vi hướng tới mục tiêu, phạt va chạm, và thúc đẩy khám phá thông qua phần thưởng tò mò và phạt lặp lại.
- **Lịch Sử Vị Trí**: Theo dõi các vị trí đã thăm để phạt hành vi lặp lại, ngăn tác nhân bị kẹt trong vòng lặp.
- **Khám Phá Dựa trên Tò Mò**: Thưởng cho các lượt thăm đến các vị trí ít được khám phá, tăng cường độ phủ của không gian trạng thái.

## 5. Các Cải Tiến Tiềm Năng
- **Tinh Chỉnh Siêu Tham Số**: Tối ưu hóa tốc độ học, kích thước lô, hoặc tốc độ giảm epsilon để hội tụ nhanh hơn.
- **Experience Replay Ưu Tiên**: Ưu tiên các chuyển tiếp quan trọng để học hiệu quả hơn.
- **Không Gian Hành Động Liên Tục**: Mở rộng mô hình để xử lý hành động liên tục cho điều hướng mượt mà hơn.
- **Mở Rộng Môi Trường**: Kiểm tra thuật toán trên các môi trường lớn hơn hoặc phức tạp hơn.
- **Tinh Chỉnh Phần Thưởng**: Tiếp tục tinh chỉnh hàm phần thưởng để cân bằng tốt hơn giữa khám phá và khai thác.

## 6. Kết Luận
Dueling DQN kết hợp với DDQN được triển khai cung cấp một khung hiệu quả cho điều hướng robot trong môi trường dựa trên lưới. Sự kết hợp của kiến trúc mạng tinh vi, hàm phần thưởng mạnh mẽ, và các chiến lược khám phá cho phép tác nhân học các đường đi hiệu quả tới mục tiêu trong khi tránh chướng ngại vật. Các công việc tương lai có thể tập trung vào việc mở rộng phương pháp cho các môi trường phức tạp hơn hoặc tích hợp thêm các kỹ thuật DRL để nâng cao hiệu suất.

---

*Lưu ý*: Báo cáo này được viết dựa trên mã được cung cấp trong file `Controller.py`. Nếu cần bổ sung thông tin thực nghiệm hoặc kết quả hiệu suất, vui lòng cung cấp thêm dữ liệu từ các thử nghiệm hoặc yêu cầu cụ thể.