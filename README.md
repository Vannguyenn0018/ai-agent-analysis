# Phân Tích Hiện Trạng và Khuyến Nghị Triển Khai AI Agent Trong Ngành Khoa Học Máy Tính

## 📌 Giới Thiệu Dự Án

Dự án này tập trung vào việc nghiên cứu, làm sạch dữ liệu và phân tích hành vi, tâm lý cũng như đặc thù công việc của nhân sự trong ngành **Khoa học máy tính (Computer Science - CS)** đối với công nghệ AI/LLM. Từ các kết quả phân tích định lượng, dự án đưa ra 5 nhóm khuyến nghị chiến lược phục vụ cho việc thiết kế và phát triển các hệ thống **AI Agent** thực tế nhằm tối ưu hóa hiệu suất lao động.

* **Mục tiêu:** Đánh giá mức độ ứng dụng AI, xác định các pain-points trong công việc và đề xuất giải pháp AI Agent phù hợp.
* **Công cụ sử dụng:** `Python`, `Pandas`, `Matplotlib`, `Seaborn`, `NumPy`, `SciPy` (mstats).
* **Môi trường phát triển:** Workbank, Google Colab, Streamlit, GitHub.

---

## 📊 Cấu Trúc Dữ Liệu Đầu Vào

Nghiên cứu tiến hành hợp nhất và xử lý thông tin từ 4 tập dữ liệu nền tảng:

1. `domain_worker_desires.csv`: Nguyện vọng và mức độ mong muốn tự động hóa của người lao động.
2. `domain_worker_metadata.csv`: Thông tin nhân khẩu học, mức lương, kinh nghiệm và tần suất sử dụng LLM.
3. `task_statement_with_metadata.csv`: Danh mục các tác vụ cốt lõi (Core tasks) và tần suất, tầm quan trọng theo chuẩn O*NET.
4. `expert_rated_technological_capability.csv`: Đánh giá của chuyên gia công nghệ về khả năng tự động hóa của AI đối với từng tác vụ.

---

## 🛠️ Quy Trình Xử Lý Dữ Liệu (Data Pipeline)

### 1. Data Preprocessing

* **Hợp nhất dữ liệu:** Kết hợp các file thông qua khóa `User ID` và `Task ID` thành một bộ cơ sở dữ liệu duy nhất.
* **Xử lý giá trị thiếu:** Điền khuyết các trường `Wage` và `Employment` bằng phương pháp **Trung vị (Median)** để tránh hiện tượng lệch phân phối thu nhập.
* **Lọc chuyên ngành:** Sử dụng từ khóa định danh (`Software`, `Computer`, `Data`, `Programmer`, `Developer`) để lọc riêng tệp dữ liệu thuộc khối ngành Khoa học máy tính.

### 2. Outlier Treatment

* **Biến Thu nhập (Income):** Chuyển đổi dữ liệu dạng khoảng (Text) sang biến liên tục (Midpoint transformation). Áp dụng kỹ thuật **Winsorization** tại phân vị 1% và 99% để triệt tiêu ảnh hưởng của các giá trị cực đoan.
* **Biến LLM Familiarity:** Gộp các nhóm phản hồi có tần suất xuất hiện cực thấp (< 5%) vào nhóm liền kề để đảm bảo tính vững và tránh sai số chuẩn trong các mô hình hồi quy.

### 3. Feature Engineering

* Phân loại thế hệ lao động (`Generation`) dựa trên độ tuổi (`Age`):
* **Gen Z:** < 28 tuổi (Nhóm lao động trẻ, tìm kiếm sự cân bằng).
* **Millennials:** 28 - 43 tuổi (Nhóm chuyển giao công nghệ).
* **Gen X+:** > 43 tuổi (Nhóm mang đậm văn hóa lao động truyền thống).

---

## 📈 Key Insights

### 📊 Hành vi và Xu hướng sử dụng AI

* **Đa dạng hóa tác vụ:** AI không chỉ được dùng để **Coding** mà còn chiếm tỷ trọng rất lớn trong việc **Truy cập thông tin (Information Access)**, **Chỉnh sửa nội dung (Edit)**, **Tạo ý tưởng (Idea Generation)** và **Phân tích dữ liệu (Analysis)**.
* **Tâm lý đón nhận:** Người sử dụng AI hàng ngày (`Daily User`) có xu hướng ít lo ngại về việc "AI gây ra đau khổ/đe dọa thay thế" hơn đáng kể so với nhóm chưa từng sử dụng (`Never User`).
* **Động lực tự động hóa:** **48.56%** nhân sự mong muốn tự động hóa để tối ưu hóa **Thời gian rảnh (Free Time)**, trong khi chỉ có **20.22%** hướng tới việc **Giảm căng thẳng (Stress)**.

### 🎯 Baseline Task Analysis

Qua biểu đồ phân tán giữa Tần suất (Frequency) và Mức độ quan trọng (Importance), dự án định vị được các tác vụ tốn thời gian nhất thuộc nhóm vị trí hỗ trợ (`Computer User Support Specialists`) và kiểm thử (`Software Quality Assurance Analysts and Testers`):

* Đọc tài liệu kỹ thuật, chẩn đoán sự cố máy tính thủ công.
* Ghi chép log lỗi (bug tracking) và lập tài liệu quy trình kiểm thử lặp đi lặp lại.

### 🧠 Đánh giá của chuyên gia về năng lực của AI

Các chuyên gia công nghệ xếp hạng khả năng AI tự động hóa cao nhất ở các vị trí:

1. **Web Developers** (4.09/5)
2. **Computer User Support Specialists** (3.89/5)
3. **Computer Programmers** (3.84/5)

---

## 💡 Đề Xuất 5 Mô Hình AI Agent Chiến Lược

> 🚀 **1. AI Agent Tự Động Hóa Tài Liệu & Tri Thức**
> * *Giải pháp:* Phát triển Agent tự động chuyển đổi mã nguồn thành tài liệu dự án (Documentation) hoặc tóm tắt tài liệu kỹ thuật nhanh.
> * *Cơ sở:* Động lực tiết kiệm thời gian (48.56%) và gánh nặng từ các tác vụ đọc hiểu/ghi chép quy trình.
> 
> 

> 🌐 **2. AI Agent Chuyên Biệt Cho Web & Hệ Thống**
> * *Giải pháp:* Tập trung sinh mã boilerplate tự động, tối ưu hóa hiệu suất giao diện (frontend/backend) và tự động hóa email phản hồi khách hàng kỹ thuật.
> * *Cơ sở:* Chuyên gia đánh giá nhóm Web Developers có tiềm năng ứng dụng AI cao nhất (4.09).
> 
> 

> 🛠️ **3. AI Agent Đa Nhiệm (Multi-task Copilot trong IDE)**
> * *Giải pháp:* Tích hợp sâu vào công cụ làm việc của lập trình viên, hỗ trợ đồng thời: Truy vấn nhanh tài liệu, gợi ý cấu trúc thiết kế hệ thống và phân tích dữ liệu log lỗi.
> * *Cơ sở:* Biểu đồ Radar chứng minh hành vi sử dụng LLM phân bổ đều trên nhiều loại tác vụ chứ không độc tôn ở Coding.
> 
> 

> 🤝 **4. AI Agent Cộng Tác & Minh Bạch (Explainable AI)**
> * *Giải pháp:* Thiết kế Agent hoạt động dưới vai trò "Người đồng hành", giải thích rõ lý do đưa ra các đoạn mã/gợi ý để nhân sự kiểm soát hoàn toàn kết quả.
> * *Cơ sở:* Giảm thiểu tâm lý lo ngại mang tính thế hệ (đặc biệt ở nhóm Gen X+) và xây dựng lòng tin dựa trên dữ liệu thái độ tích cực của người dùng thường xuyên.
> 
> 

> 🔍 **5. AI Agent Chẩn Đoán & Quản Lý Lỗi Tự Động**
> * *Giải pháp:* Tự động quét log hệ thống, phân loại lỗi phần mềm và đẩy thẳng lên hệ thống bug tracking; hỗ trợ chẩn đoán bước đầu cho kỹ sư hỗ trợ người dùng.
> * *Cơ sở:* Giải quyết trực diện tác vụ có "điểm đau composite" cao nhất được tìm thấy trong phân tích phân tán.
> 
> 

---
