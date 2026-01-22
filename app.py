import streamlit as st
import os
import tempfile
from PIL import Image
from sg_radar_controller import SG_RADAR_Controller

# Page Config
st.set_page_config(
    page_title="SG-R.A.D.A.R",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Premium" feel
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
        color: #1f2937;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        background-color: #2563eb;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    h1 {
        color: #111827;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2, h3 {
        color: #374151;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Controller
@st.cache_resource
def load_controller():
    return SG_RADAR_Controller()

try:
    radar = load_controller()
    st.sidebar.success("시스템 준비 완료 (System Ready)")
except Exception as e:
    st.sidebar.error(f"시스템 초기화 실패: {e}")
    st.stop()

# --- HEADER ---
st.title("📡 SG-R.A.D.A.R")
st.markdown("**세계화학공업(주) 신속 점착 진단 및 분석 추천기 (Rapid Adhesion Diagnosis & Analysis Recommender)**")
st.markdown("---")

# --- SIDEBAR (INPUT) ---
with st.sidebar:
    st.header("1. 현장 데이터 입력")
    
    st.subheader("📷 피착제 표면 사진")
    surface_file = st.file_uploader("피착제 표면을 촬영하여 업로드하세요", type=['jpg', 'jpeg', 'png'], key="surface")
    
    st.subheader("💧 액적 테스트 사진 (DeepDrop)")
    
    st.markdown("**1. 물 (Water)**")
    water_files = st.file_uploader("물방울 사진을 업로드하세요", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, key="water")
    
    st.markdown("**2. 다이아이오도메탄 (Diiodomethane)**")
    diiodo_files = st.file_uploader("다이아이오도메탄 사진을 업로드하세요 (선택)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, key="diiodo")
    
    analyze_btn = st.button("🔍 AI 진단 및 추천 시작")
    
    st.info("💡 Tip: 정확한 표면 에너지 분석을 위해 두 가지 액체를 모두 사용하는 것이 좋습니다.")

# --- MAIN (OUTPUT) ---
if analyze_btn:
    if not surface_file or not water_files:
        st.error("⚠️ 필수 데이터를 입력해주세요. (피착제 사진, 물방울 사진)")
    else:
        # Create temp files for processing
        with st.spinner("이미지 분석 및 시뮬레이션 중... (V-SAMS & DeepDrop Engine)"):
            try:
                # Save Surface Image
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_surf:
                    tmp_surf.write(surface_file.getvalue())
                    surface_path = tmp_surf.name
                
                # Save Water Images
                water_paths = []
                for f in water_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(f.getvalue())
                        water_paths.append(tmp.name)
                        
                # Save Diiodo Images
                diiodo_paths = []
                if diiodo_files:
                    for f in diiodo_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                            tmp.write(f.getvalue())
                            diiodo_paths.append(tmp.name)
                
                # Run Analysis
                result = radar.run_rapid_diagnosis(surface_path, water_paths, diiodo_paths)
                
                # Cleanup Temp Files
                os.remove(surface_path)
                for p in water_paths: os.remove(p)
                for p in diiodo_paths: os.remove(p)
                
                # --- DISPLAY RESULTS ---
                
                # 1. Diagnosis Section
                st.subheader("🔎 1단계: 피착제 진단 결과")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(surface_file, caption="입력된 피착제", use_column_width=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>감지된 재질</h3>
                        <p style="font-size: 24px; font-weight: bold;">{result['diagnosis'].get('material', 'Unknown')}</p>
                        <p style="color: gray;">Surface Material</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    method_tag = result['diagnosis'].get('method', 'Unknown')
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>표면 에너지 (SFE)</h3>
                        <p style="font-size: 24px; font-weight: bold; color: #2563eb;">{result['diagnosis'].get('surface_energy', 0):.1f} <span style="font-size:16px">dyne/cm</span></p>
                        <p style="color: gray; font-size: 12px;">{method_tag}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # 2. Recommendation Section
                st.subheader("🏆 2단계: AI 최적 제품 추천")
                
                best = result['best_product']
                
                st.markdown(f"""
                <div class="recommendation-box">
                    <h2>👑 BEST MATCH: {best['name']} ({best['id']})</h2>
                    <div style="display: flex; justify-content: space-around; margin-top: 20px;">
                        <div>
                            <h4>예상 유지 시간</h4>
                            <h1>{best['pred_time']:.1f} 시간</h1>
                        </div>
                        <div>
                            <h4>잔사 발생 확률</h4>
                            <h1>{(1.0 - best['clean_prob'])*100:.0f}% <span style="font-size:18px">(안전: {best['clean_prob']*100:.0f}%)</span></h1>
                        </div>
                        <div>
                            <h4>종합 점수</h4>
                            <h1>{best['score']:.1f}</h1>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if best['clean_prob'] < 0.8:
                    st.warning("⚠️ 주의: 해당 피착제는 표면 에너지가 매우 낮거나 특수하여, 강한 점착력이 필요하지만 제거 시 잔사가 남을 확률이 일부 존재합니다.")
                else:
                    st.success("✅ 안전: 제거 시 깔끔하게 떨어질 확률이 높습니다. (Clean Removal)")
                
                # 3. Candidates Table
                with st.expander("📊 다른 후보 제품 보기 (Top 5 Candidates)"):
                    candidates = result['top_3_candidates'] # Actually reusing top_3 list but usually controller returns sorted list or we can access full logic
                    # Just show what we have
                    import pandas as pd
                    df_res = pd.DataFrame(candidates)
                    st.dataframe(
                        df_res[['name', 'id', 'pred_time', 'clean_prob', 'score']]
                        .rename(columns={'name': '제품명', 'id': '코드', 'pred_time': '예상시간(h)', 'clean_prob': '깔끔제거확률', 'score': '점수'})
                        .style.format({'예상시간(h)': '{:.1f}', '깔끔제거확률': '{:.2f}', '점수': '{:.1f}'})
                    )
                    
            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {e}")
                st.exception(e)


else:
    # Landing Page State
    st.info("👈 왼쪽 사이드바에서 사진을 업로드하고 'AI 진단' 버튼을 눌러주세요.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ❓ SG-R.A.D.A.R란?")
        st.write("세계화학공업(주)의 **AI Vision & Physics Engine**이 결합된 최첨단 점착 솔루션입니다.")
        st.write("복잡한 물성 계산 없이 사진만으로 최적의 솔루션을 제안할 수 있도록 돕습니다.")
    
    with col2:
        st.markdown("### 🛠 내부 동작 원리")
        st.write("1. **V-SAMS**: 표면의 거칠기와 재질을 시각적으로 분석")
        st.write("2. **DeepDrop**: 물방울의 접촉각을 통해 표면 에너지 계산")
        st.write("3. **XGBoost Brain**: 100만 건의 가상 실험 데이터를 학습한 AI가 최적 매칭 예측")

# --- TECHNICAL DEMO DISCLAIMER (Footer) ---
st.markdown("---")
with st.expander("ℹ️ Technical Demonstration Notes (더미 데이터 및 미구현 기능 명세)", expanded=True):
    st.markdown("""
    **본 시스템은 기술 시연을 위해 일부 구간에 더미 데이터 및 고정값을 사용하고 있습니다.**
    
    | 구분 | 사용 중인 더미 데이터/로직 (Current Status) | 데이터 위치 (Location) | 실제 운영 시 필요 데이터 (Required for Production) |
    |---|---|---|---|
    | **표면 거칠기 (Roughness)** | 고정값 `0.5` 사용 (알고리즘 미적용) | `sg_radar_controller.py` 내 하드코딩 | V-SAMS의 거칠기 측정 모듈 연동 필요 |
    | **제품 데이터베이스** | 샘플 제품 62종 데이터 | `assets/sg_product_db.csv` | 전체 제품 물성 정보가 담긴 ERP/DB 연동 |
    | **소재 분류 (Materials)** | Metal/Plastic 외 0으로 고정 | `sg_radar_controller.py` 내 Feature Vector 생성 로직 | Glass, Wood 등 다양한 소재에 대한 One-Hot Encoding 로직 확장 |
    | **AI 모델 파일** | MobileSAM (`mobile_sam.pt`) | `models/mobile_sam.pt` | (현재 적용됨) 지속적인 파인튜닝 모델 업데이트 |
    
    > **Note**: 업로드하신 사진은 실제 분석에 사용되지만, 위 항목들은 시뮬레이션을 위해 사전 정의된 값을 참조합니다.
    """)



