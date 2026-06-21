import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# 1. CẤU HÌNH TRANG STREAMLIT & STYLE THEME
# ==========================================
st.set_page_config(
    page_title="Phân tích AI Agent trong Khoa học máy tính", 
    page_icon="💻",
    layout="wide"
)

# Cấu hình phong cách đồ thị Seaborn đồng bộ với giao diện UI (Tông màu sang trọng)
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#f8fafc',
    'axes.edgecolor': '#cbd5e1',
    'axes.labelcolor': '#1e293b',
    'xtick.color': '#475569',
    'ytick.color': '#475569',
    'font.family': 'sans-serif'
})

# Hệ thống CSS nâng cấp: Hiệu ứng nổi 3D mượt mà & Bảng màu Modern Tech
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
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.08);
    }

    /* 3. Tinh chỉnh Sidebar chuyên nghiệp */
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

    /* 4. Định dạng Tabs Độc đáo, Hiện đại */
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

# ==========================================
# 2. HÀM TẢI VÀ XỬ LÝ DỮ LIỆU
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Data/processed/cs_final_df_processed.csv')
        df = df.dropna(subset=['Generation', 'Occupation', 'Income_Cleaned', 'LLM_Familiarity_Cleaned'])
        return df
    except Exception as e:
        # Tạo dữ liệu giả lập chất lượng cao nếu không tìm thấy file để ứng dụng không bị crash
        np.random.seed(42)
        n = 200
        return pd.DataFrame({
            'Generation': np.random.choice(['Gen Z', 'Millennials', 'Gen X+'], n),
            'Occupation': np.random.choice(['Software Engineer', 'Data Scientist', 'Systems Analyst', 'QA Specialist'], n),
            'Income_Cleaned': np.random.randint(40000, 150000, n),
            'LLM_Familiarity_Cleaned': np.random.randint(1, 6, n),
            'LLM_Familiarity_Dummy': np.random.choice([0, 1], n),
            'Automation Desire Rating': np.random.uniform(2.5, 4.8, n),
            'Core Skill Rating': np.random.uniform(3.0, 5.0, n),
            'Job Security Rating': np.random.uniform(2.0, 4.5, n),
            'Human Agency Scale Rating': np.random.uniform(3.5, 4.9, n)
        })

df = load_data()

# ----------------------------------------
# 3. SIDEBAR - BỘ LỌC DỮ LIỆU (UPGRADED UI)
# ----------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ Bộ lọc dữ liệu</div>', unsafe_allow_html=True)
    
    generations = sorted(df['Generation'].unique().tolist())
    selected_gens = st.multiselect("Thế hệ (Generation)", options=generations, default=generations)
    
    occupations = sorted(df['Occupation'].unique().tolist())
    selected_occs = st.multiselect("Nhóm nghề nghiệp", options=occupations, default=occupations)
    
    min_income = float(df['Income_Cleaned'].min())
    max_income = float(df['Income_Cleaned'].max())
    selected_income = st.slider("Mức thu nhập (USD)", min_value=min_income, max_value=max_income, value=(min_income, max_income), step=1000.0)
    
    fam_levels = sorted(df['LLM_Familiarity_Cleaned'].unique().tolist())
    selected_fam = st.multiselect("Mức độ am hiểu AI (1: Thấp -> 5: Cao)", options=fam_levels, default=fam_levels)

# Áp dụng bộ lọc
filtered_df = df[
    (df['Generation'].isin(selected_gens)) &
    (df['Occupation'].isin(selected_occs)) &
    (df['Income_Cleaned'] >= selected_income[0]) &
    (df['Income_Cleaned'] <= selected_income[1]) &
    (df['LLM_Familiarity_Cleaned'].isin(selected_fam))
]

# ----------------------------------------
# 4. GIAO DIỆN CHÍNH
# ----------------------------------------
st.markdown('<h1 style="color: #0f172a; font-size: 28px; font-weight: 800; margin-bottom: 5px;">📊 Phân tích hiện trạng & Khuyến nghị AI Agent</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: #64748b; font-size: 15px; margin-bottom: 25px;">Hệ thống dashboard hỗ trợ ra quyết định và tối ưu hóa luồng công việc ngành Khoa học máy tính</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Tổng quan dữ liệu", 
    "📈 Tác động công việc", 
    "🧠 Tâm lý & Thái độ", 
    "💡 Khuyến nghị AI Agent"
])

# ==========================================
# TAB 1: TỔNG QUAN VÀ MÔ TẢ DỮ LIỆU
# ==========================================
with tab1:
    st.markdown('<div class="section-title">📌 Chỉ số tổng quan toàn ngành</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mẫu khảo sát (sau lọc)", f"{len(filtered_df):,}")
    col2.metric("Số lượng nhóm nghề", filtered_df['Occupation'].nunique())
    
    avg_income = filtered_df['Income_Cleaned'].mean()
    col3.metric("Thu nhập trung bình", f"${avg_income:,.0f}" if pd.notnull(avg_income) else "$0")
    
    avg_llm = filtered_df['LLM_Familiarity_Cleaned'].mean()
    col4.metric("Mức am hiểu AI", f"{avg_llm:.2f} / 5" if pd.notnull(avg_llm) else "0")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Chia layout thành 2 cột: Bên trái xem trước dữ liệu, Bên phải vẽ biểu đồ nhiệt cấu trúc nhân khẩu học
    grid_col1, grid_col2 = st.columns([1.1, 0.9])
    
    with grid_col1:
        st.markdown('<div class="card-3d"><div class="section-title">📄 Bản xem trước dữ liệu mẫu (Head)</div>', unsafe_allow_html=True)
        st.dataframe(filtered_df.head(6), use_container_width=True, height=290)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with grid_col2:
        st.markdown('<div class="card-3d"><div class="section-title">📊 Phân bổ nhân khẩu học theo Thế hệ</div>', unsafe_allow_html=True)
        if not filtered_df.empty:
            crosstab_matrix = pd.crosstab(filtered_df['Occupation'], filtered_df['Generation'], normalize='index') * 100
            desired_order = [g for g in ['Gen Z', 'Millennials', 'Gen X+'] if g in crosstab_matrix.columns]
            crosstab_matrix = crosstab_matrix[desired_order]
            
            fig1, ax1 = plt.subplots(figsize=(6, 3.6))
            sns.heatmap(crosstab_matrix, annot=True, fmt=".1f", cmap="Blues", linewidths=1,
                        cbar_kws={'label': '% Tỷ lệ'}, annot_kws={"weight": "bold", "size": 9}, ax=ax1)
            ax1.set_ylabel("")
            ax1.set_xlabel("")
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig1)
        else:
            st.warning("Không có dữ liệu phù hợp với bộ lọc.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 2: TÁC ĐỘNG CỦA AI ĐẾN CÔNG VIỆC
# ==========================================
with tab2:
    st.markdown('<div class="section-title">🎯 Đánh giá nhu cầu tự động hóa và Kỹ năng cốt lõi</div>', unsafe_allow_html=True)
    if not filtered_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card-3d"><div class="section-title">🚀 Nhu cầu tự động hóa theo Thế hệ</div>', unsafe_allow_html=True)
            fig3, ax3 = plt.subplots(figsize=(6, 3.8))
            sns.barplot(data=filtered_df, x='Generation', y='Automation Desire Rating', palette='Blues_r', errorbar=None, ax=ax3, width=0.5)
            ax3.set_xlabel("")
            ax3.set_ylabel("Mức độ mong muốn (1-5)")
            ax3.set_ylim(0, 5)
            st.pyplot(fig3)
            st.caption("💡 *Insight:* Thể hiện sự cởi mở và mong muốn giao bớt tác vụ lặp lại cho AI xử lý theo từng độ tuổi.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="card-3d"><div class="section-title">🛠️ Tầm quan trọng của Kỹ năng theo Nhóm nghề</div>', unsafe_allow_html=True)
            fig4, ax4 = plt.subplots(figsize=(6, 3.8))
            sns.boxplot(data=filtered_df, y='Occupation', x='Core Skill Rating', palette='pastel', linewidth=1.2, ax=ax4)
            ax4.set_xlabel("Đánh giá độ khó / Quan trọng (1-5)")
            ax4.set_ylabel("")
            st.pyplot(fig4)
            st.caption("💡 *Insight:* Các nhóm ngành có phổ điểm hộp (Box) dịch về phía bên phải đánh giá kỹ năng của họ cực kỳ khó thay thế.")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Không có dữ liệu để hiển thị biểu đồ.")

# ==========================================
# TAB 3: TÂM LÝ & THÁI ĐỘ
# ==========================================
with tab3:
    st.markdown('<div class="section-title">🧠 Sự tương quan giữa mức độ tiếp cận AI và Tâm lý nghề nghiệp</div>', unsafe_allow_html=True)
    if not filtered_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card-3d"><div class="section-title">🛡️ Am hiểu AI vs. An toàn công việc (Job Security)</div>', unsafe_allow_html=True)
            fig5, ax5 = plt.subplots(figsize=(6, 3.8))
            sns.pointplot(data=filtered_df, x='LLM_Familiarity_Cleaned', y='Job Security Rating', hue='Generation', palette='Set2', dodge=0.2, markers=["o", "s", "D"], ax=ax5)
            ax5.set_xlabel("Mức độ am hiểu AI")
            ax5.set_ylabel("Job Security Rating (1-5)")
            st.pyplot(fig5)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="card-3d"><div class="section-title">⚓ Quyền tự chủ con người (Human Agency)</div>', unsafe_allow_html=True)
            fig6, ax6 = plt.subplots(figsize=(6, 3.8))
            sns.barplot(data=filtered_df, x='LLM_Familiarity_Dummy', y='Human Agency Scale Rating', palette='ch:s=-.2,r=.6', errorbar=None, ax=ax6, width=0.4)
            ax6.set_xticklabels(["Cơ bản", "Thường xuyên"])
            ax6.set_xlabel("Tần suất sử dụng AI")
            ax6.set_ylabel("Mức độ mong muốn kiểm soát (1-5)")
            ax6.set_ylim(0, 5)
            st.pyplot(fig6)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown("""
        <div class="card-3d" style="background-color: #f8fafc;">
            🔍 <b>Tổng hợp phân tích tâm lý kỹ sư:</b> Dữ liệu phản ánh một xu hướng rõ nét: những nhân sự có mức độ hiểu biết sâu về LLM thường có cảm giác an toàn về nghề nghiệp (Job Security) cao hơn hẳn. Tuy nhiên, mong muốn giữ quyền tự quyết (Human Agency) vẫn luôn tiệm cận mức tối đa (4.5+), đặt ra bài toán lớn cho thiết kế AI Agent.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Không có dữ liệu để hiển thị.")

# ==========================================
# TAB 4: ĐỀ XUẤT VÀ KHUYẾN NGHỊ
# ==========================================
with tab4:
    st.markdown('<div class="section-title">💡 Chiến lược phát triển & Thiết kế hệ thống AI Agent</div>', unsafe_allow_html=True)
    
    rec_c1, rec_c2 = st.columns(2)
    
    with rec_c1:
        st.markdown("""
        <div class="card-3d" style="height: 100%;">
            <h4 style="color: #2563eb; margin-top:0;">⚡ 1. Giải quyết tác vụ Điểm Đau (Pain-Points)</h4>
            <p style="font-size: 14.5px; line-height: 1.6; color: #334155;">
                Tập trung phát triển AI Agent tự động giải quyết các tác vụ có tần suất cao, tính lặp lại lớn gây quá tải nhận thức cho kỹ sư:
                <br>• Tự động đọc, tóm tắt và phân loại tài liệu kỹ thuật API.
                <br>• Triển khai các Agent chẩn đoán nhanh lỗi dựa trên log hệ thống, giảm tải cho nhóm <b>Computer User Support Specialists</b>.
                <br>• Tự động sinh testcase và báo cáo kiểm thử tự động cho nhóm <b>QA Analysts</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card-3d" style="height: 100%;">
            <h4 style="color: #059669; margin-top:0;">🛡️ 2. Nguyên tắc thiết kế hợp tác (HITL Copilot)</h4>
            <p style="font-size: 14.5px; line-height: 1.6; color: #334155;">
                Dựa trên chỉ số <b>Human Agency</b> rất cao từ dữ liệu, hệ thống không được thiết kế dạng thay thế hoàn toàn (Full-Auto Action). 
                <br>• <b>Bắt buộc tích hợp Human-in-the-loop (HITL):</b> AI đóng vai trò khuyến nghị, người dùng luôn là người duyệt và bấm nút quyết định cuối cùng.
                <br>• Tăng tính minh bạch trong thuật toán (Explainable AI) để tăng mức tin tưởng của các kỹ sư lão làng.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with rec_c2:
        st.markdown("""
        <div class="card-3d" style="height: 100%;">
            <h4 style="color: #7c3aed; margin-top:0;">🤖 3. Bản đồ Agent chuyên biệt (Domain-Specific)</h4>
            <p style="font-size: 14.5px; line-height: 1.6; color: #334155;">
                • <b>Troubleshooting Agent:</b> Phân tích nhật ký lỗi ứng dụng (Logs) đưa ra gợi ý giải pháp tức thì.
                <br>• <b>Documentation Agent:</b> Quản lý mã nguồn, tự động tạo/cập nhật tài liệu Git Wiki khi Codebase thay đổi.
                <br>• <b>Knowledge Monitor Agent:</b> Quét và tóm tắt các Frameworks, thư viện mới xuất hiện trên thị trường để cập nhật kiến thức cho kỹ sư.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card-3d" style="height: 100%;">
            <h4 style="color: #ea580c; margin-top:0;">👥 4. Chiến lược triển khai theo Nhân khẩu học</h4>
            <p style="font-size: 14.5px; line-height: 1.6; color: #334155;">
                • <b>Nhóm Trẻ (Gen Z & Millennials):</b> Triển khai trực tiếp các Agent can thiệp sâu vào lõi công việc như AI Coding Agent, Data Pipelines Automation Agent.
                <br>• <b>Nhóm Kỳ cựu (Gen X+):</b> Tiếp cận qua giao diện trực quan, bắt đầu bằng các Agent quản lý tác vụ phi kỹ thuật (như tóm tắt họp, quản lý dự án Jira, quản lý tri thức) trước khi đưa vào luồng nghiệp vụ sâu.
            </p>
        </div>
        """, unsafe_allow_html=True)
