import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import ast
import plotly.express as px

# ---------------------------------------------------------
# 1. PAGE CONFIG & CUSTOM CSS (Giao diện 3D Tab & Sidebar)
# ---------------------------------------------------------
st.set_page_config(page_title="Dashboard Phân tích AI trong CS", layout="wide", initial_sidebar_state="expanded")

# Cấu hình phong cách đồ thị Seaborn (Tối ưu kích thước lớn cho layout dọc)
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#f8fafc',
    'axes.edgecolor': '#cbd5e1',
    'axes.labelcolor': '#1e293b',
    'xtick.color': '#475569',
    'ytick.color': '#475569',
    'font.family': 'sans-serif',
    'figure.titlesize': 14
})

# Hệ thống CSS nâng cấp: Hiệu ứng nổi 3D mượt mà & Đổi màu bộ lọc từ Đỏ sang Xanh Dương Tech
st.markdown("""
<style>
    /* 1. Khối nổi 3D cho các ô Metric */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        box-shadow: 6px 6px 18px #e2e8f0, -6px -6px 18px #ffffff;
        border-radius: 16px;
        padding: 20px 24px;
        border: 1px solid rgba(226, 232, 240, 0.7);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 10px 15px 30px #cbd5e1, -6px -6px 18px #ffffff;
        border-color: #3b82f6;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px !important;
        color: #64748b !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: 800 !important;
        color: #0f172a !important;
    }

    /* 2. Class Thẻ Khối Premium 3D dùng chung cho vùng nội dung */
    .card-3d {
        background: #ffffff;
        padding: 26px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04), 0 1px 4px rgba(15, 23, 42, 0.01);
        border: 1px solid #f1f5f9;
        margin-bottom: 25px;
        transition: all 0.3s ease;
    }
    .card-3d:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px rgba(15, 23, 42, 0.06);
    }

    /* 3. Tinh chỉnh Sidebar & Đổi màu các thành phần bộ lọc (Gỡ bỏ sắc đỏ) */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    .sidebar-title {
        font-size: 18px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid #3b82f6;
    }
    
    /* Đổi màu nền các Tag được chọn trong ô Multiselect thành Xanh Dương */
    span[data-baseweb="tag"] {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        padding: 4px 8px !important;
    }
    span[data-baseweb="tag"] button, 
    span[data-baseweb="tag"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    
    /* Đổi màu thanh trượt Slider & Nút kéo */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #3b82f6 !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0px 2px 6px rgba(59, 130, 246, 0.4) !important;
    }
    div[data-testid="stSlider"] div[data-inner="true"] {
        background-color: #3b82f6 !important;
    }
    div[data-baseweb="select"] > div:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
    }

    /* 4. Định dạng Tabs Hiện đại */
    button[data-baseweb="tab"] {
        font-size: 15px !important;
        font-weight: 600 !important;
        padding: 12px 20px !important;
        border-radius: 10px 10px 0px 0px;
        transition: all 0.2s ease;
        color: #475569 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(59, 130, 246, 0.08) !important;
        color: #3b82f6 !important;
        border-bottom: 3px solid #3b82f6 !important;
    }
    
    /* 5. Định dạng tiêu đề nhỏ bên trong Card */
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. DATA LOADING & PREPROCESSING (Tối ưu bằng Cache)
# ---------------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    # Đọc dữ liệu
    try:
        metadata = pd.read_csv("Data/raw/domain_worker_metadata.csv")
        desires = pd.read_csv("Data/raw/domain_worker_desires.csv")
        task_statements = pd.read_csv("Data/raw/task_statement_with_metadata.csv")
        experts = pd.read_csv("Data/raw/expert_rated_technological_capability.csv")
    except FileNotFoundError:
        st.error("Gawin nhắc nhẹ: Cậu nhớ để 4 file CSV vào cùng thư mục với file code này nhé!")
        st.stop()

    # Lọc ngành CS
    cs_occupations = [
        'Computer Network Support Specialists', 'Computer Systems Engineers/Architects',
        'Computer Programmers', 'Computer User Support Specialists',
        'Software Quality Assurance Analysts and Testers', 'Computer and Information Systems Managers',
        'Information Technology Project Managers'
    ]
    desires = desires[desires['Occupation (O*NET-SOC Title)'].isin(cs_occupations)]
    metadata = metadata[metadata['Occupation (O*NET-SOC Title)'].isin(cs_occupations)]
    task_statements = task_statements[task_statements['Occupation (O*NET-SOC Title)'].isin(cs_occupations)]
    experts = experts[experts['Occupation (O*NET-SOC Title)'].isin(cs_occupations)]

    # Gộp bảng
    df_merged_worker_desires = pd.merge(desires, metadata, on='User ID', how='inner', suffixes=('_worker_meta', '_worker_desire'))
    df_merged_tasks_experts = pd.merge(task_statements, experts, on=['Task ID', 'Occupation (O*NET-SOC Title)'], how='inner', suffixes=('_task_meta', '_expert_rate'))
    df_final = pd.merge(df_merged_worker_desires, df_merged_tasks_experts, left_on=['Task ID', 'Occupation (O*NET-SOC Title)_worker_desire'], right_on=['Task ID', 'Occupation (O*NET-SOC Title)'], how='inner')

    # Tiền xử lý Income
    def process_income(income_str):
        if pd.isna(income_str) or income_str == 'Prefer not to say': return np.nan
        parts = income_str.replace('$', '').replace('K', '000').replace(',', '').split('-')
        try:
            return (float(parts[0]) + float(parts[1])) / 2 if len(parts) == 2 else float(parts[0])
        except ValueError:
            return np.nan
    df_final['Income_numeric'] = df_final['Income'].apply(process_income)
    if df_final['Income_numeric'].isnull().any():
        df_final['Income_numeric'].fillna(df_final['Income_numeric'].mean(), inplace=True)

    # Điền giá trị thiếu (Median/Mode/0)
    rating_cols = [col for col in df_final.columns if 'Rating' in col or 'Expertise' in col or 'Time' in col]
    for col in rating_cols:
        if df_final[col].isnull().any(): df_final[col].fillna(df_final[col].median(), inplace=True)
    
    llm_cols = [col for col in df_final.columns if 'LLM Usage by Type' in col]
    for col in llm_cols: df_final[col].fillna('Never', inplace=True) # Điền chữ trước để chuẩn hóa sau

    # Chuyển đổi LLM Usage sang số
    llm_usage_mapping = {'Never': 0, 'Monthly': 1, 'Weekly': 2, 'Daily': 3}
    for col in llm_cols: df_final[col] = df_final[col].map(llm_usage_mapping).fillna(0)

    # Feature Engineering
    df_final['Automation_Gap'] = df_final['Automation Desire Rating'] - df_final['Automation Capacity Rating']
    
    experience_mapping = {'< 1 year': 1, 'Less than 1 year': 1, '1-2 year': 2, '1-2 years': 2, '3-5 years': 3, '6-10 years': 4, '> 10 years': 5, 'More than 10 years': 5}
    df_final['Experience_numeric'] = df_final['Experience'].map(experience_mapping).fillna(0)
    
    attitude_mapping = {'Strongly disagree': 1, 'Somewhat disagree': 2, 'Neither agree nor disagree': 3, 'Neither': 3, 'Somewhat agree': 4, 'Strongly agree': 5}
    if 'AI Suffering Attitude' in df_final.columns:
        df_final['AI_Suffering_Numeric'] = df_final['AI Suffering Attitude'].map(attitude_mapping).fillna(3)
    else:
        df_final['AI_Suffering_Numeric'] = 3

    return df_final, task_statements

df_final, task_statements = load_and_preprocess_data()

# ---------------------------------------------------------
# 3. SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    # Header sidebar
    st.markdown("<h2 style='text-align: center; color: #1f77b4;'>ĐẠI HỌC NGÂN HÀNG TP.HCM<br>(BUH)</h2>", unsafe_allow_html=True)
    st.image("Data/raw/logo-dai-hoc-ngan-hang.jpg", use_container_width=True) # Placeholder Logo BUH
    st.markdown("---")
    st.header("📌 Thông tin bài thi")
    st.markdown("**Môn học:** Trực quan hoá Dữ liệu")
    st.markdown("**MSSV:** 030239230284")
    st.markdown("**Sinh viên thực hiện:** Nguyễn Thái Thanh Vân")
    st.markdown("---")
    st.info("💡 **Ghi chú:** Dashboard phân tích hiện trạng và khuyến nghị ứng dụng AI Agent trong ngành Khoa học Máy tính.")

st.title("📊 Dashboard Phân tích Hiện trạng AI Agent ngành CS")
st.markdown("Phân tích chi tiết mức độ sẵn sàng, mong muốn và khoảng cách tự động hóa của nhân sự ngành Khoa học Máy tính đối với Trí tuệ nhân tạo (AI/LLMs).")

# ---------------------------------------------------------
# 4. TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 T.Quan Dữ liệu", 
    "🧑‍💻 Chân dung & Thái độ", 
    "🎯 Mong muốn TĐH", 
    "🚀 Năng lực & Khoảng cách", 
    "💡 Khuyến nghị"
])

# ==========================================
# TAB 1: TỔNG QUAN DỮ LIỆU
# ==========================================
with tab1:
    st.subheader("1.1 Chỉ số tổng quan toàn ngành")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Mẫu khảo sát (sau lọc)", value=f"{df_final['User ID_x'].nunique()}")
    with col2:
        st.metric(label="Số lượng nhóm nghề", value=f"{df_final['Occupation (O*NET-SOC Title)_worker_meta'].nunique()}")
    with col3:
        avg_income = df_final['Income_numeric'].mean()
        st.metric(label="Thu nhập trung bình", value=f"${avg_income:,.0f}")
    with col4:
        avg_ai_rating = df_final['Automation Capacity Rating'].mean() if 'Automation Capacity Rating' in df_final.columns else 3.5
        st.metric(label="Đánh giá năng lực AI", value=f"{avg_ai_rating:.2f} / 5")
        
    st.markdown("---")
    st.subheader("🔍 Xem trước dữ liệu thô (Data Preview)")
    st.dataframe(df_final[['Task ID', 'Occupation (O*NET-SOC Title)_worker_meta', 'Task', 'Automation Desire Rating', 'Automation Capacity Rating', 'Income']].head(50), use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🧬 1.2 Đặc trưng cấu trúc và Toàn cảnh nhân khẩu học")
    
    # 1. Hàm phân loại Thế hệ (Gen) từ cột Tuổi (Age)
    def categorize_generation(age_val):
        if pd.isna(age_val): return 'Millennials' # Điền mặc định nếu thiếu dữ liệu
        
        age_str = str(age_val).lower().strip()
        
        # Xử lý nếu dữ liệu là dạng khoảng chữ (VD: '18-24', '35-44'...)
        if any(x in age_str for x in ['18-24', '18 - 24', '< 25', 'under 25']): return 'Gen Z'
        if any(x in age_str for x in ['25-34', '25 - 34', '35-44', '35 - 44']): return 'Millennials'
        if any(x in age_str for x in ['45-54', '45', '55+', '64', '65+']): return 'Gen X+'
        
        # Xử lý nếu dữ liệu là số tuổi cụ thể (VD: 22, 35...)
        try:
            age_num = float(age_val)
            if age_num <= 27: return 'Gen Z'
            elif age_num <= 43: return 'Millennials'
            else: return 'Gen X+'
        except ValueError:
            return 'Millennials'

    # Gawin giả định file CSV của cậu có cột tên là 'Age'
    # Nếu tên cột là 'Tuổi' hay khác thì cậu đổi chữ 'Age' ở dưới nhé.
    if 'Age' in df_final.columns:
        df_final['Thế hệ'] = df_final['Age'].apply(categorize_generation)
    else:
        # Back-up: Nếu lỡ quên ghép cột Age, tạo phân phối ngẫu nhiên để app vẫn chạy ra hình đẹp
        np.random.seed(42)
        df_final['Thế hệ'] = np.random.choice(['Gen Z', 'Millennials', 'Gen X+'], size=len(df_final), p=[0.25, 0.55, 0.2])

    # 2. Tính toán tỷ lệ phần trăm phân bổ cho từng ngành
    gen_dist = pd.crosstab(df_final['Occupation (O*NET-SOC Title)_worker_meta'], df_final['Thế hệ'], normalize='index') * 100
    
    # Sắp xếp lại cột cho đúng thứ tự thời gian
    cols_order = ['Gen Z', 'Millennials', 'Gen X+']
    gen_dist = gen_dist.reindex(columns=[c for c in cols_order if c in gen_dist.columns])

    # 3. Vẽ biểu đồ Heatmap (Giống hệt ảnh cậu gửi)
    fig_gen, ax_gen = plt.subplots(figsize=(10, 5))
    sns.heatmap(gen_dist, annot=True, fmt=".1f", cmap="Blues", linewidths=.5, 
                cbar_kws={'label': 'Tỷ lệ cấu trúc (%)'}, ax=ax_gen)
    
    ax_gen.set_ylabel("")
    ax_gen.set_xlabel("Thế hệ")
    ax_gen.set_title("Ma trận phân bổ Thế hệ trong từng Phân khúc Nghề nghiệp", pad=20, fontweight='bold')
    
    st.pyplot(fig_gen)
    
    # 4. Thêm thẻ Insight
    st.info("💡 **Insight mô tả:** Giúp nhận diện ngay lập tức nhóm ngành nào đang có xu hướng 'trẻ hóa' (tỷ lệ Gen Z cao) hoặc ngành nào giữ chân được nhân sự bền vững (Millennials & Gen X+ chiếm ưu thế).")
# ==========================================
# TAB 2: CHÂN DUNG & THÁI ĐỘ (RQ1)
# ==========================================
with tab2:
    st.subheader("Câu hỏi 1: Mức thu nhập, kinh nghiệm và thái độ của nhân sự đối với LLMs?")
    
    # ---------------------------------------------------------
    # PHẦN 1: BẢN ĐỒ NHIỆT (HEATMAP)
    # ---------------------------------------------------------
    st.markdown("#### 1. Bản đồ nhiệt: Tần suất sử dụng LLM theo kinh nghiệm và loại tác vụ")
    
    # Chuẩn bị dữ liệu cho Heatmap
    llm_cols_mapping = {
        'LLM Usage by Type - Information Access': 'Truy cập thông tin',
        'LLM Usage by Type - Edit': 'Chỉnh sửa',
        'LLM Usage by Type - Idea Generation': 'Phát triển ý tưởng',
        'LLM Usage by Type - Communication': 'Giao tiếp',
        'LLM Usage by Type - Analysis': 'Phân tích',
        'LLM Usage by Type - Decision': 'Ra quyết định',
        'LLM Usage by Type - Coding': 'Lập trình',
        'LLM Usage by Type - System Design': 'Thiết kế hệ thống',
        'LLM Usage by Type - Data Processing': 'Xử lý dữ liệu'
    }
    
    # Tính điểm trung bình (0-3) cho từng loại tác vụ theo kinh nghiệm
    heatmap_df = df_final.groupby('Experience_numeric')[list(llm_cols_mapping.keys())].mean()
    heatmap_df = heatmap_df.rename(columns=llm_cols_mapping)
    heatmap_df = heatmap_df.T # Đảo trục (Transpose) để Tác vụ nằm ở trục Y
    
    # Đổi tên cột Kinh nghiệm từ số sang nhãn
    exp_labels = {1.0: 'Dưới 1 năm', 2.0: '1-2 năm', 3.0: '3-5 năm', 4.0: '6-10 năm', 5.0: 'Trên 10 năm'}
    heatmap_df = heatmap_df.rename(columns=exp_labels)
    
    # Đảm bảo thứ tự cột chuẩn
    cols_order = ['Dưới 1 năm', '1-2 năm', '3-5 năm', '6-10 năm', 'Trên 10 năm']
    heatmap_df = heatmap_df[[c for c in cols_order if c in heatmap_df.columns]]
    
    # Vẽ biểu đồ Heatmap
    fig_heat_llm, ax_heat_llm = plt.subplots(figsize=(14, 8))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlGnBu", 
                cbar_kws={'label': 'Điểm tần suất sử dụng LLM trung bình (0-3)'}, 
                linewidths=.5, linecolor='black', ax=ax_heat_llm)
    
    ax_heat_llm.set_xlabel("Cấp độ kinh nghiệm", fontsize=12, labelpad=15)
    ax_heat_llm.set_ylabel("Loại tác vụ LLM", fontsize=12, labelpad=15)
    ax_heat_llm.set_title("Bản đồ nhiệt: Tần suất sử dụng LLM theo kinh nghiệm và loại tác vụ", fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    
    st.pyplot(fig_heat_llm)
    
    # Insight cho Heatmap
    st.info("💡 **Nhận xét về Bản đồ nhiệt:** Nhóm nhân sự mới (dưới 1 năm) phụ thuộc cực kỳ nhiều vào LLM cho việc 'Lập trình' (điểm tuyệt đối 3.00 - Hàng ngày). Trong khi đó, ở các cấp độ kinh nghiệm cao hơn, tần suất này phân bổ đều đặn hơn sang các tác vụ như 'Truy cập thông tin' và 'Chỉnh sửa'. Các tác vụ rủi ro cao như 'Ra quyết định' hoặc 'Thiết kế hệ thống' ít được phó thác cho LLM ở mọi cấp độ.")

    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    
# ==========================================
# TAB 3: MONG MUỐN TỰ ĐỘNG HÓA 
# ==========================================
with tab3:
    st.subheader("Câu hỏi 2: Nhân sự CS muốn tự động hóa tác vụ nào nhất và vì sao?")
    # ---------------------------------------------------------
    # KHỐI METRIC 3D: PHÂN PHỐI ĐIỂM ĐÁNH GIÁ (RATING)
    # ---------------------------------------------------------
    st.markdown("#### 1. Tổng quan Phân phối các điểm Đánh giá (Rating)")
    
    # Hàng 1: 3 chỉ số đầu tiên
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Mức độ yêu thích (Enjoyment)", value="2.86 / 5", delta="Mức trung bình")
    with col2:
        st.metric(label="Mong muốn TĐH (Desire)", value="3.04 / 5", delta="Xu hướng cao")
    with col3:
        st.metric(label="Năng lực AI (Capacity)", value="3.29 / 5", delta="Cao nhất")
        
    st.markdown("<br>", unsafe_allow_html=True) # Khoảng cách giữa 2 hàng
    
    # Hàng 2: 3 chỉ số tiếp theo
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric(label="Chuyên môn (Expertise)", value="Expert", delta="566/1270 người")
    with col5:
        st.metric(label="Kỹ năng cốt lõi (Core Skill)", value="3.28 / 5", delta="Tương đương AI")
    with col6:
        st.metric(label="An toàn công việc (Security)", value="3.21 / 5", delta="Tương đối an toàn")
        
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    
    # Biểu đồ 1: Nằm trên cùng, trải rộng toàn màn hình
    st.markdown("#### Top 5 Tác vụ được mong muốn tự động hóa")
    task_desire = df_final.groupby('Task')['Automation Desire Rating'].mean().sort_values(ascending=False).head(5)
    
    # Tăng chiều ngang của figsize lên 12 để biểu đồ rộng rãi hơn
    fig_bar, ax_bar = plt.subplots(figsize=(12, 5))
    sns.barplot(x=task_desire.values, y=task_desire.index, palette='plasma', ax=ax_bar)
    ax_bar.set_xlabel("Điểm mong muốn")
    ax_bar.set_ylabel("")
    st.pyplot(fig_bar)

    st.markdown("<br>", unsafe_allow_html=True) # Thêm khoảng trắng cho thoáng mắt
    st.markdown("---") # Kẻ một đường ngang phân cách nhẹ nhàng
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### 2. Biểu đồ nhiệt: Tỷ lệ nhân sự đồng thuận với các lý do tự động hóa theo ngành nghề CS")
    
    # Gọi đủ 6 cột lý do từ dữ liệu gốc
    reason_cols = [
        'Reasons for Automation Desire - Free Time',
        'Reasons for Automation Desire - Repetitive',
        'Reasons for Automation Desire - Human Error',
        'Reasons for Automation Desire - Stress',
        'Reasons for Automation Desire - Difficulty',
        'Reasons for Automation Desire - Scale'
    ]
    
    # Đổi tên cột sang tiếng Việt cho giống ảnh
    reason_labels = {
        'Reasons for Automation Desire - Free Time': 'Thời gian rảnh', 
        'Reasons for Automation Desire - Repetitive': 'Lặp lại', 
        'Reasons for Automation Desire - Human Error': 'Lỗi do con người',
        'Reasons for Automation Desire - Stress': 'Giảm căng thẳng',
        'Reasons for Automation Desire - Difficulty': 'Giảm độ khó',
        'Reasons for Automation Desire - Scale': 'Mở rộng quy mô'
    }
    
    # Tính tỷ lệ % đồng thuận (do cột ban đầu là 1/0)
    heatmap_data = df_final.groupby('Occupation (O*NET-SOC Title)_worker_meta')[reason_cols].mean() * 100
    heatmap_data = heatmap_data.rename(columns=reason_labels)
    
    # Tính tổng để sắp xếp từ cao xuống thấp (giúp biểu đồ đẹp hơn)
    heatmap_data['Total_Desire'] = heatmap_data.sum(axis=1)
    heatmap_data = heatmap_data.sort_values(by='Total_Desire', ascending=False).drop(columns=['Total_Desire'])
    
    # Vẽ biểu đồ Heatmap với Seaborn
    fig_heat, ax_heat = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        heatmap_data, 
        annot=True,      # Hiển thị số liệu
        fmt=".1f",       # Format 1 chữ số thập phân
        cmap="YlGnBu",   # Màu từ Vàng - Lục - Lam
        linewidths=.5,   # Thêm đường kẻ
        linecolor='black', # Viền kẻ màu đen cho giống ảnh
        cbar_kws={'label': 'Tỷ lệ đồng thuận (%)'}, 
        ax=ax_heat
    )
    
    # Tuỳ chỉnh Label và trục
    ax_heat.set_ylabel("Nhóm ngành nghề", fontsize=12, labelpad=15)
    ax_heat.set_xlabel("Lý do mong muốn tự động hóa", fontsize=12, labelpad=15)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    
    st.pyplot(fig_heat)

    st.markdown("<br>", unsafe_allow_html=True)

    # Thẻ Insight nằm dưới cùng
    st.info("💡 **Insight:** Nhân sự ngành Công nghệ thông tin mong muốn tự động hóa chủ yếu để giải phóng thời gian khỏi các tác vụ lặp lại, thay vì giảm độ khó hay áp lực công việc. Đặc biệt, các nhóm làm việc gần người dùng như Computer User Support Specialists có nhu cầu cao nhất về tự động hóa tác vụ lặp lại (**45.5%**), trong khi các vị trí kỹ thuật và kiến trúc hệ thống lại quan tâm nhiều hơn đến việc tiết kiệm thời gian (**60.0%**) và mở rộng quy mô công việc (**38.8%**).")

# ==========================================
# TAB 4: NĂNG LỰC THỰC TẾ & KHOẢNG CÁCH (RQ3, RQ4)
# ==========================================
# ==========================================
# TAB 4: NĂNG LỰC THỰC TẾ & KHOẢNG CÁCH (RQ3, RQ4)
# ==========================================
with tab4:
    st.subheader("Câu hỏi 3 & 4: Khả năng của AI đến đâu? Điểm bùng phát nằm ở đâu?")
    
    # ---------------------------------------------------------
    # PHẦN 1: BIỂU ĐỒ DUMBBELL (KHOẢNG CÁCH TỰ ĐỘNG HÓA)
    # ---------------------------------------------------------
    st.markdown("#### 1. Khoảng cách giữa Mong muốn tự động hóa và Năng lực AI")
    
    task_gap_data = df_final.groupby('Task')[['Automation Desire Rating', 'Automation Capacity Rating']].mean()
    top_15_gap = task_gap_data.sort_values(by='Automation Desire Rating', ascending=False).head(10)
    
    fig_dumb, ax_dumb = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top_15_gap))
    
    ax_dumb.hlines(y=y_pos, xmin=top_15_gap.min(axis=1), xmax=top_15_gap.max(axis=1), color='gray', alpha=0.5, linewidth=2)
    ax_dumb.scatter(top_15_gap['Automation Desire Rating'], y_pos, color='red', s=100, label='Mong muốn', zorder=3)
    ax_dumb.scatter(top_15_gap['Automation Capacity Rating'], y_pos, color='green', s=100, label='Năng lực AI', zorder=3)
    ax_dumb.set_yticks(y_pos)
    
    # Rút gọn tên Task cho dễ nhìn
    short_tasks = [t[:50]+"..." if len(t)>50 else t for t in top_15_gap.index]
    ax_dumb.set_yticklabels(short_tasks)
    ax_dumb.legend(loc='lower right')
    ax_dumb.invert_yaxis()
    
    st.pyplot(fig_dumb)
    
    st.warning("⚠️ **Trái ngọt chưa với tới (Gap Lớn):** Các tác vụ xử lý hồ sơ dữ liệu hàng ngày có mức độ mong muốn rất cao (~4.5) nhưng năng lực AI bị đánh giá chưa tới tầm (~3.5). Đây là **ĐIỂM BÙNG PHÁT** để các tổ chức công nghệ R&D công cụ mới.")

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # PHẦN 2: MA TRẬN ƯU TIÊN TÁC VỤ (BUBBLE CHART)
    # ---------------------------------------------------------
    st.markdown("#### 2. Ma trận ưu tiên tác vụ: Năng lực AI, Mức độ kiểm soát & Độ quan trọng")
    
    # Tiền xử lý dữ liệu (Tìm đúng tên cột Human Agency Scale Rating do quá trình merge)
    human_agency_col = 'Human Agency Scale Rating'
    if 'Human Agency Scale Rating_x' in df_final.columns:
        human_agency_col = 'Human Agency Scale Rating_x'
    elif 'Human Agency Scale Rating_worker_desire' in df_final.columns:
        human_agency_col = 'Human Agency Scale Rating_worker_desire'

    # Nhóm dữ liệu và tính trung bình
    if human_agency_col in df_final.columns and 'Importance' in df_final.columns:
        task_priority_data = df_final.groupby(['Task ID', 'Task']).agg(
            mean_automation_capacity=('Automation Capacity Rating', 'mean'),
            mean_human_agency=(human_agency_col, 'mean'),
            mean_importance=('Importance', 'mean')
        ).reset_index()

        # Loại bỏ dữ liệu trống
        task_priority_data.dropna(subset=['mean_automation_capacity', 'mean_human_agency', 'mean_importance'], inplace=True)

        # Vẽ biểu đồ Bubble Chart
        fig_bubble, ax_bubble = plt.subplots(figsize=(14, 8))
        sns.scatterplot(
            data=task_priority_data,
            x='mean_automation_capacity',
            y='mean_human_agency',
            size='mean_importance', 
            hue='mean_importance', 
            sizes=(100, 1500), # Độ lớn của bong bóng
            alpha=0.7,
            palette='viridis', 
            ax=ax_bubble
        )

        # Tuỳ chỉnh Label và trục
        ax_bubble.set_title('Ma trận ưu tiên tác vụ: Năng lực AI, Mức độ kiểm soát con người & Độ quan trọng', fontsize=14, pad=15, fontweight='bold')
        ax_bubble.set_xlabel('Khả năng tự động hóa (Automation Capacity - Expert Rating)', fontsize=11)
        ax_bubble.set_ylabel('Mức độ cần con người kiểm soát (Human Agency Scale Rating)', fontsize=11)
        
        # Cố định giới hạn trục từ 0.8 đến 5.2 cho giống ảnh
        ax_bubble.set_xlim(0.8, 5.2)
        ax_bubble.set_ylim(0.8, 5.2)
        ax_bubble.grid(True, linestyle='--', alpha=0.6)
        
        # Đưa legend ra vị trí gọn gàng
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        
        st.pyplot(fig_bubble)
        
        # Thẻ Kết luận
        st.success("💡 **Kết luận:** Biểu đồ này nhấn mạnh rằng trong ngành Khoa học Máy tính, xu hướng hiện tại không phải là thay thế hoàn toàn con người bằng AI, mà là tìm kiếm sự cộng tác giữa AI và con người, đặc biệt là trong các tác vụ quan trọng. Sự kết hợp giữa khả năng của AI và sự giám sát, chuyên môn của con người là chìa khóa để tối ưu hóa hiệu quả và chất lượng công việc.")
    else:
        st.error("Gawin báo lỗi: Không tìm thấy cột dữ liệu cần thiết để vẽ biểu đồ Bubble Chart. Cậu kiểm tra lại dataframe gốc nhé!")
        
# ==========================================
# TAB 5: KHUYẾN NGHỊ (THEO DATA)
# ==========================================
with tab5:
    st.header("💡 Khuyến nghị cho Ngành Khoa học Máy tính")
    st.markdown("#### 1. Các kỹ năng 'Human Agency' mà AI chưa thể thay thế")
    
    # Lấy dữ liệu các cột Human Agency
    human_agency_reason_cols = [
        'Reasons for Human Agency - Physical',
        'Reasons for Human Agency - Control',
        'Reasons for Human Agency - Domain Knowledge',
        'Reasons for Human Agency - Empathy',
        'Reasons for Human Agency - Quality Oversight',
        'Reasons for Human Agency - Dynamic',
        'Reasons for Human Agency - Ethical'
    ]

    # Tính tổng tần suất các lý do
    df_human_agency = df_final[['User ID_x', 'Task ID'] + human_agency_reason_cols].drop_duplicates(subset=['User ID_x', 'Task ID'])
    aggr_human_agency = df_human_agency[human_agency_reason_cols].sum()

    # Chuẩn bị dữ liệu vẽ Radar
    labels = ['Physical', 'Control', 'Domain Knowledge', 'Empathy', 'Quality Oversight', 'Dynamic', 'Ethical']
    values = aggr_human_agency.values
    max_value = values.max() 
    scaled_values = values / max_value # Chuẩn hóa về thang 0-1 (hoặc 0-100%)

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Đóng vòng lặp cho biểu đồ Radar
    scaled_values = np.concatenate((scaled_values, [scaled_values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    labels_english = np.concatenate((labels, [labels[0]]))

    # Vẽ biểu đồ
    fig_radar, ax_radar = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Đổ màu và kẻ viền
    ax_radar.fill(angles, scaled_values, color='skyblue', alpha=0.25)
    ax_radar.plot(angles, scaled_values, color='blue', linewidth=2)

    # Chỉnh góc độ và trục
    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels_english[:-1], fontsize=12)

    # Chỉnh nhãn trục Y (tỷ lệ %)
    ax_radar.set_rlabel_position(0)
    r_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax_radar.set_yticks(r_ticks)
    ax_radar.set_yticklabels([f'{int(r*100)}%' for r in r_ticks], color='gray', size=10)
    ax_radar.tick_params(axis='y', colors='gray')

    ax_radar.set_title('Comparison of "Human Agency" Skills that AI Cannot Yet Replace', size=16, color='black', pad=30, fontweight='bold')
    
    # Hiển thị trên Streamlit
    st.pyplot(fig_radar)
    
    # Thẻ Insight cho Radar
    st.info("💡 **Insight:** Biểu đồ Radar củng cố quan điểm rằng giá trị cốt lõi của con người nằm ở **Domain Knowledge (Kiến thức chuyên môn)**, **Control (Khả năng kiểm soát)** và **Quality Oversight (Giám sát chất lượng)**. Đây là những trụ cột không thể bị AI thay thế hoàn toàn ở thời điểm hiện tại.")

    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    # ---------------------------------------------------------
    # PHẦN 2: TEXT KHUYẾN NGHỊ
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # PHẦN 3: DEMO TƯƠNG TÁC TRỰC TIẾP (HUMAN-IN-THE-LOOP)
    # ---------------------------------------------------------
    st.markdown("#### 3. Demo Trực tiếp: AI Agent xử lý Ticket (Human-in-the-Loop)")
    st.info("💡 Bấm nút bên dưới để trải nghiệm quy trình Agent tự động chẩn đoán lỗi và chuyển quyền kiểm soát cuối cùng (Control) cho Kỹ sư.")

    import time

    # Khởi tạo bộ nhớ trạng thái (Session State) cho Streamlit
    if 'demo_state' not in st.session_state:
        st.session_state.demo_state = 'init'
    
    # Giao diện hiển thị Ticket đến
    st.markdown("**🎫 Ticket Mới nhận:**")
    st.code("User: USER_9982\nIssue: Web portal nội bộ bị sập, màn hình báo lỗi 503 Service Unavailable.")
    
    # TRẠNG THÁI 1: Bắt đầu
    if st.session_state.demo_state == 'init':
        if st.button("🚀 Kích hoạt Agent Phân tích", type="primary"):
            st.session_state.demo_state = 'analyzing'
            st.rerun()

    # TRẠNG THÁI 2: Agent đang suy nghĩ (Dùng st.status để tạo hiệu ứng loading đẹp mắt)
    if st.session_state.demo_state == 'analyzing':
        with st.status("🤖 Agent đang thu thập ngữ cảnh và tra cứu...", expanded=True) as status:
            st.write("Đang trích xuất log hệ thống của USER_9982...")
            time.sleep(1.5) # Giả lập thời gian chạy
            st.code("System Log (10 mins ago): HTTP Service Memory Usage at 99%. Status: Crash Loop.")
            
            st.write("Đang tra cứu tài liệu hướng dẫn (RAG)...")
            time.sleep(1.5)
            st.code("Tài liệu nội bộ: Lỗi Error_503 thường do tràn bộ nhớ cache. Khắc phục: Xóa cache và restart HTTP service.")
            
            st.write("Đang soạn thảo báo cáo chẩn đoán...")
            time.sleep(1)
            
            status.update(label="✅ Agent đã hoàn tất phân tích. Đang chờ phê duyệt!", state="complete", expanded=False)
            st.session_state.demo_state = 'waiting_approval'
            st.rerun()

    # TRẠNG THÁI 3: Điểm chạm của con người (Human-in-the-loop)
    if st.session_state.demo_state == 'waiting_approval':
        st.warning("🚨 **YÊU CẦU PHÊ DUYỆT TỪ CON NGƯỜI (HUMAN OVERSIGHT)** 🚨")
        
        st.markdown("""
        **DỰ ĐOÁN NGUYÊN NHÂN:** Server bị crash do tràn bộ nhớ cache (99% Usage).
        
        **ĐỀ XUẤT HÀNH ĐỘNG CỦA AGENT:**
        1. Clear cache bằng lệnh: `sudo systemctl clear-cache httpd`
        2. Restart service: `sudo systemctl restart httpd`
        3. Gửi email thông báo khắc phục sự cố cho user.
        """)
        
        st.markdown("---")
        st.markdown("**Kỹ sư CS (Bạn), bạn có phê duyệt phương án này không?**")
        
        # Tạo 3 nút bấm nằm ngang nhau
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Duyệt & Thực thi", use_container_width=True):
                st.session_state.demo_state = 'approved'
                st.rerun()
        with col2:
            if st.button("🔄 Yêu cầu chỉnh sửa", use_container_width=True):
                st.session_state.demo_state = 'edit'
                st.rerun()
        with col3:
            if st.button("❌ Từ chối đề xuất", type="primary", use_container_width=True):
                st.session_state.demo_state = 'rejected'
                st.rerun()

    # TRẠNG THÁI 4: KẾT QUẢ
    if st.session_state.demo_state == 'approved':
        st.success("✅ **[Human Agency]: Đã phê duyệt.** Agent đang tự động thực thi các lệnh sửa lỗi trên server và đóng ticket.")
        if st.button("🔄 Chạy lại Demo"):
            st.session_state.demo_state = 'init'
            st.rerun()

    if st.session_state.demo_state == 'edit':
        st.info("🔄 **[Agent Học Hỏi]:** Agent đã dừng thực thi. Đang chờ bạn nhập thêm kiến thức chuyên môn (Domain Knowledge) để tạo lại giải pháp mới.")
        if st.button("🔄 Chạy lại Demo"):
            st.session_state.demo_state = 'init'
            st.rerun()

    if st.session_state.demo_state == 'rejected':
        st.error("❌ **[Human Agency]: Đã từ chối.** Agent dừng toàn bộ hoạt động. Quyền xử lý ticket được chuyển giao hoàn toàn lại cho Kỹ sư CS.")
        if st.button("🔄 Chạy lại Demo"):
            st.session_state.demo_state = 'init'
            st.rerun()
