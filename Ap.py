import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. CẤU HÌNH TRANG STREAMLIT
st.set_page_config(
    page_title="Phân tích AI Agent trong Khoa học máy tính", 
    page_icon="💻",
    layout="wide"
)
# 2. HỆ THỐNG CSS TẠO HIỆU ỨNG NỔI 3D & CARD PREMIUM
st.markdown("""
<style>
    /* 1. Tạo khối nổi 3D cho các ô Metric mặc định của Streamlit */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff, #f6f8fb);
        box-shadow: 5px 5px 15px #e2e8f0, -5px -5px 15px #ffffff;
        border-radius: 16px;
        padding: 22px 18px;
        border: 1px solid rgba(226, 232, 240, 0.8);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    /* Hiệu ứng tương tác 3D nâng lên rõ rệt khi di chuột vào khối chỉ số */
    div[data-testid="stMetric"]:hover {
        transform: translateY(-6px) scale(1.01);
        box-shadow: 12px 12px 28px #cbd5e1, -8px -8px 20px #ffffff;
    }
    
    /* 2. Định dạng chữ trong khối Metric */
    div[data-testid="stMetricLabel"] {
        font-size: 14px !important;
        color: #64748B !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 800 !important;
        color: #0F172A !important;
    }

    /* 3. Tạo class Thẻ Khối 3D tùy chỉnh cho vùng nội dung/đồ thị */
    .card-3d {
        background: #ffffff;
        padding: 24px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06), 0 1px 3px rgba(15, 23, 42, 0.02);
        border: 1px solid rgba(241, 245, 249, 0.9);
        margin-bottom: 25px;
        transition: all 0.3s ease;
    }
    .card-3d:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 35px rgba(15, 23, 42, 0.12);
    }

    /* 4. Làm nổi khối vùng Sidebar bộ lọc bên trái */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
        box-shadow: 4px 0px 20px rgba(15, 23, 42, 0.04);
    }

    /* 5. Định dạng lại khối Tabs thanh lịch */
    button[data-baseweb="tab"] {
        font-size: 15px !important;
        font-weight: 600 !important;
        border-radius: 8px 8px 0px 0px;
        transition: all 0.2s ease;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(59, 130, 246, 0.08) !important;
        color: #3b82f6 !important;
    }
</style>
""", unsafe_allow_html=True)


# 2. HÀM TẢI VÀ XỬ LÝ DỮ LIỆU
@st.cache_data
def load_data():
    # Đọc dữ liệu từ file bạn đã cung cấp
    df = pd.read_csv(r"C:\Users\HP\Downloads\cs_final_df_processed.csv")
    # Làm sạch các giá trị NA trong các cột dùng để lọc
    df = df.dropna(subset=['Generation', 'Occupation', 'Income_Cleaned', 'LLM_Familiarity_Cleaned'])
    return df

df = load_data()

# ----------------------------------------
# 3. SIDEBAR - BỘ LỌC DỮ LIỆU (INTERACTIVE FILTER PANEL)
# ----------------------------------------
st.sidebar.header("🎛️ Bộ lọc dữ liệu")
st.sidebar.markdown("Tùy chỉnh phạm vi phân tích theo thời gian thực.")

# Bộ lọc 1: Thế hệ (Generation)
generations = df['Generation'].unique().tolist()
selected_gens = st.sidebar.multiselect(
    "Thế hệ (Generation)", 
    options=generations, 
    default=generations
)

# Bộ lọc 2: Nhóm nghề nghiệp (Occupation)
occupations = df['Occupation'].unique().tolist()
selected_occs = st.sidebar.multiselect(
    "Nhóm nghề nghiệp", 
    options=occupations, 
    default=occupations
)

# Bộ lọc 3: Mức thu nhập (Income_Cleaned)
min_income = float(df['Income_Cleaned'].min())
max_income = float(df['Income_Cleaned'].max())
selected_income = st.sidebar.slider(
    "Mức thu nhập (USD)", 
    min_value=min_income, 
    max_value=max_income, 
    value=(min_income, max_income),
    step=1000.0
)

# Bộ lọc 4: Mức độ am hiểu AI (LLM Familiarity)
fam_levels = sorted(df['LLM_Familiarity_Cleaned'].unique().tolist())
selected_fam = st.sidebar.multiselect(
    "Mức độ am hiểu AI (1: Thấp -> 5: Rất cao)", 
    options=fam_levels, 
    default=fam_levels
)

# --- ÁP DỤNG BỘ LỌC VÀO DATAFRAME ---
filtered_df = df[
    (df['Generation'].isin(selected_gens)) &
    (df['Occupation'].isin(selected_occs)) &
    (df['Income_Cleaned'] >= selected_income[0]) &
    (df['Income_Cleaned'] <= selected_income[1]) &
    (df['LLM_Familiarity_Cleaned'].isin(selected_fam))
]

# ----------------------------------------
# 4. GIAO DIỆN CHÍNH & TABS
# ----------------------------------------
st.title("📊 Phân tích hiện trạng và đề xuất khuyến nghị AI Agent cho ngành Khoa học máy tính")
st.markdown("---")

# Khởi tạo Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 1. Tổng quan & Mô tả dữ liệu", 
    "📈 2. Tác động của AI đến công việc", 
    "🧠 3. Tâm lý & Thái độ", 
    "💡 4. Khuyến nghị AI Agent"
])

# ==========================================
# TAB 1: TỔNG QUAN VÀ MÔ TẢ DỮ LIỆU
# ==========================================
with tab1:
    st.subheader("1.1 Các chỉ số tổng quan")
    
    # Chỉ số dạng Metric cập nhật theo thời gian thực
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng số khảo sát (sau lọc)", f"{len(filtered_df):,}")
    col2.metric("Số lượng nghề nghiệp", filtered_df['Occupation'].nunique())
    
    avg_income = filtered_df['Income_Cleaned'].mean()
    col3.metric("Thu nhập trung bình", f"${avg_income:,.0f}" if pd.notnull(avg_income) else "$0")
    
    avg_llm = filtered_df['LLM_Familiarity_Cleaned'].mean()
    col4.metric("Mức am hiểu AI trung bình", f"{avg_llm:.2f} / 5" if pd.notnull(avg_llm) else "0")

    st.subheader("1.2 Data Preview")
    st.dataframe(filtered_df.head(), use_container_width=True)
    
    st.subheader("1.3 Đặc trưng cấu trúc và Toàn cảnh nhân khẩu học")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='card-3d'>", unsafe_allow_html=True)
        if not filtered_df.empty:
            # Ý tưởng 1: Tính toán ma trận chéo tỷ lệ phần trăm thế hệ trong từng nhóm nghề
            crosstab_matrix = pd.crosstab(
                filtered_df['Occupation'], 
                filtered_df['Generation'], 
                normalize='index'
            ) * 100
            
            # Sắp xếp lại thứ tự cột cho logic theo thời gian nếu có đủ dữ liệu
            desired_order = [g for g in ['Gen Z', 'Millennials', 'Gen X+'] if g in crosstab_matrix.columns]
            crosstab_matrix = crosstab_matrix[desired_order]
            
            fig1, ax1 = plt.subplots(figsize=(7, 4.8))
            # Vẽ bản đồ nhiệt sang trọng với bảng màu Blues
            sns.heatmap(
                crosstab_matrix, 
                annot=True, 
                fmt=".1f", 
                cmap="Blues", 
                linewidths=0.5,
                cbar_kws={'label': 'Tỷ lệ cấu trúc (%)'}, 
                annot_kws={"weight": "bold", "size": 10},
                ax=ax1
            )
            ax1.set_title("Ma trận phân bổ Thế hệ trong từng Phân khúc Nghề nghiệp", fontsize=11, fontweight='bold', color='#1E293B', pad=12)
            ax1.set_ylabel("")
            ax1.set_xlabel("Thế hệ")
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig1)
            st.caption("💡 **Insight mô tả:** Giúp nhận diện ngay lập tức nhóm ngành nào đang có xu hướng 'trẻ hóa' (tỷ lệ Gen Z cao) hoặc ngành nào giữ chân được nhân sự bền vững (Millennials & Gen X+ chiếm ưu thế).")
        else:
            st.warning("Không có dữ liệu phù hợp với bộ lọc.")
        st.markdown("</div>", unsafe_allow_html=True)
        

# ==========================================
# TAB 2: TÁC ĐỘNG CỦA AI ĐẾN CÔNG VIỆC
# ==========================================
with tab2:
    st.subheader("Đánh giá nhu cầu tự động hóa và kỹ năng chuyên môn")
    if not filtered_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig3, ax3 = plt.subplots(figsize=(6, 5))
            sns.barplot(data=filtered_df, x='Generation', y='Automation Desire Rating', palette='magma', errorbar=None, ax=ax3)
            ax3.set_title("Mức độ mong muốn tự động hóa theo Thế hệ")
            ax3.set_xlabel("Thế hệ")
            ax3.set_ylabel("Automation Desire (1-5)")
            ax3.set_ylim(0, 5)
            st.pyplot(fig3)
            st.info("Biểu đồ phản ánh mức độ khao khát được AI hỗ trợ giải quyết các tác vụ trong công việc.")
            
        with c2:
            fig4, ax4 = plt.subplots(figsize=(6, 5))
            sns.boxplot(data=filtered_df, y='Occupation', x='Core Skill Rating', palette='Set2', ax=ax4)
            ax4.set_title("Đánh giá Kỹ năng cốt lõi (Core Skill) theo Nhóm nghề")
            ax4.set_xlabel("Core Skill Rating (1-5)")
            ax4.set_ylabel("")
            st.pyplot(fig4)
            st.info("Nhóm nghề nào đánh giá kỹ năng của họ quan trọng/khó thay thế nhất?")
    else:
        st.warning("Không có dữ liệu để hiển thị biểu đồ.")

# ==========================================
# TAB 3: TÂM LÝ & THÁI ĐỘ
# ==========================================
with tab3:
    st.subheader("Mức độ an toàn và sự hài lòng trong công việc khi có AI")
    if not filtered_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig5, ax5 = plt.subplots(figsize=(6, 5))
            # So sánh Job Security dựa trên mức độ dùng AI
            sns.pointplot(data=filtered_df, x='LLM_Familiarity_Cleaned', y='Job Security Rating', hue='Generation', dodge=True, ax=ax5)
            ax5.set_title("Am hiểu AI vs. Mức độ an toàn công việc (Job Security)")
            ax5.set_xlabel("Mức độ am hiểu AI")
            ax5.set_ylabel("Job Security Rating (1-5)")
            st.pyplot(fig5)
            
        with c2:
            fig6, ax6 = plt.subplots(figsize=(6, 5))
            # Human Agency Scale Rating
            sns.barplot(data=filtered_df, x='LLM_Familiarity_Dummy', y='Human Agency Scale Rating', palette='pastel', errorbar=None, ax=ax6)
            ax6.set_title("Sự tự chủ của con người (Human Agency) theo tần suất dùng AI")
            ax6.set_xlabel("Sử dụng AI thường xuyên (0 = Cơ bản, 1 = Thường xuyên)")
            ax6.set_ylabel("Human Agency Rating")
            ax6.set_ylim(0, 5)
            st.pyplot(fig6)
            
        st.markdown("""
        **Nhận xét dữ liệu:** Dữ liệu cho thấy sự khác biệt về cảm giác an toàn công việc (Job Security) và mong muốn tự chủ (Human Agency) giữa các thế hệ. Việc ứng dụng AI thường xuyên (Dummy = 1) có thể tác động đến mức độ hài lòng và khao khát duy trì quyền quyết định của chuyên gia.
        """)
    else:
        st.warning("Không có dữ liệu để hiển thị.")

# ==========================================
# TAB 4: ĐỀ XUẤT VÀ KHUYẾN NGHỊ (Lấy từ file Python)
# ==========================================
with tab4:
    st.header("Đề xuất & Khuyến nghị Phát triển AI Agent")
    st.markdown("""
    Căn cứ vào kết quả phân tích dữ liệu ngành Khoa học máy tính (O*NET) và khảo sát tâm lý, nghiên cứu đề xuất hướng phát triển AI Agent như sau:
    
    ### 1. Giải quyết các tác vụ 'Điểm đau' (Pain-Point Tasks)
    * **Khuyến nghị:** Phát triển AI Agent nhằm trực tiếp giải quyết các tác vụ có tần suất cao, tầm quan trọng lớn và mang tính chất lặp lại (ví dụ: ghi chép, xử lý thông tin).
    * **Cơ sở:** Các tác vụ như đọc tài liệu kỹ thuật, chẩn đoán sự cố máy tính, ghi lại lỗi phần mềm và duy trì môi trường kiểm thử tạo ra gánh nặng nhận thức cực kỳ lớn cho người lao động.
    
    ### 2. Phát triển các AI Agent chuyên biệt (Domain-Specific Agents)
    * **Agent Hỗ trợ Chẩn đoán & Xử lý sự cố:** Có khả năng phân tích nhật ký (logs), tài liệu kỹ thuật để đề xuất giải pháp ban đầu. Đặc biệt hữu ích cho *Computer User Support Specialists* và *Systems Analysts*.
    * **Agent Quản lý lỗi & Tài liệu:** Tự động hóa việc ghi nhận, phân loại lỗi và tạo tài liệu kiểm thử. Phù hợp cho *Software Quality Assurance Analysts*.
    * **Agent Cập nhật Kiến thức Tự động:** Giám sát và tóm tắt công nghệ, frameworks mới giúp nhân sự liên tục duy trì chuyên môn.
    
    ### 3. Nguyên tắc thiết kế AI "Hợp tác" (Collaborative Copilot)
    * Nghiên cứu chỉ ra nhu cầu kiểm soát (Human Agency) của các chuyên gia Khoa học máy tính là rất cao. AI Agent không nên được thiết kế để "thay thế tự động hoàn toàn" mà phải hoạt động như một **"người đồng hành"**.
    * Bắt buộc phải tích hợp cơ chế HITL (Human-in-the-loop), cho phép người dùng kiểm duyệt, hiệu chỉnh kết quả đầu ra của AI trước khi ra quyết định cuối cùng.
    
    ### 4. Chiến lược tiếp cận theo Nhân khẩu học (Demographics)
    * **Gen Z & Millennials:** Sẵn sàng ứng dụng AI vào lõi công việc (Coding, Analysis). Có thể triển khai trực tiếp các Agent sinh mã nguồn hoặc tự động hóa luồng dữ liệu.
    * **Gen X+:** Cần tính minh bạch cao và hướng dẫn trực quan. Nên bắt đầu số hóa với các Agent hỗ trợ truy xuất thông tin (Information Access) hoặc nhắc việc quản lý dự án trước khi đi sâu vào chuyên môn kỹ thuật.
    """)
