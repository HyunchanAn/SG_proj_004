# SG-R.A.D.A.R (Rapid Adhesion Diagnosis & Analysis Recommender)

**SG-R.A.D.A.R**는 세계화학공업(주)의 현장 영업 사원들을 위한 **AI 기반 신속 점착 진단 및 제품 추천 시스템**입니다. 
고객사의 피착제 표면 사진과 물방울 테스트 사진만으로 표면 특성을 분석하고, 최적의 산업용 테이프 제품을 추천합니다.

## 🌟 Key Features

1.  **AI Surface Diagnosis (V-SAMS)**: 피착제 표면 이미지를 분석하여 재질(Metal, Plastic 등)과 마감 처리(Mirror, Hairline 등)를 자동으로 식별합니다.
2.  **Contact Angle & SFE Analysis (DeepDrop)**: 물과 다이아이오도메탄 액적 사진을 통해 **OWRK 방법**으로 정밀한 표면 에너지(Surface Free Energy)를 산출합니다.
3.  **Virtual Simulation (AI Brain)**: 1,000건 이상의 가상 데이터를 학습한 XGBoost 모델이 각 제품의 **점착 유지 시간(Holding Time)**과 **잔사 발생 확률(Clean Removal)**을 예측합니다.
4.  **Real Product Database**: 실제 자사 제품 62종의 물성 데이터를 기반으로 추천합니다.
5.  **Interactive Dashboard**: Streamlit 기반의 직관적인 UI로 테블릿/PC에서 손쉽게 사용 가능합니다.

6.  **Secure Authentication**: Supabase 기반의 로그인/회원가입 기능을 제공하여 보안을 강화했습니다.

## 📂 Project Structure

```bash
SG_RADAR/
├── app.py                  # Streamlit 메인 애플리케이션
├── sg_radar_controller.py  # 시스템 제어 컨트롤러 (센서, DB, AI 통합)
├── assets/                 # 자산 폴더
│   └── sg_product_db.csv   # 제품 데이터베이스 (62종)
├── libs/                   # 핵심 라이브러리
│   ├── deepdrop_sfe/       # 표면 에너지 분석 모듈
│   └── vsams_core/         # 표면 재질 분석 모듈
├── models/                 # 학습된 AI 모델 저장소 (.json, .pt)
├── SG_products/            # 제품 카탈로그 원본 (PDF)
└── tools/                  # 유틸리티 스크립트 (데이터 생성, 학습, 파싱 등)
```

## 🛠️ Installation

이 프로젝트는 **Python 3.9+** 환경을 권장합니다.

```bash
# 1. Repository Clone
git clone https://github.com/YourRepo/SG_RADAR.git
cd SG_RADAR

# 2. Update pip
python -m pip install --upgrade pip

# 3. Install Dependencies
pip install streamlit pandas numpy xgboost opencv-python Pillow pypdf torch torchvision ultralytics supabase
# (Optional) Install MobileSAM dependencies if needed specific versions
```

### 4. Setup Secrets
Supabase 인증을 위해 `.streamlit/secrets.toml` 파일을 생성하고 자격 증명을 입력해야 합니다.
```toml
[supabase]
url = "YOUR_SUPABASE_URL"
key = "YOUR_SUPABASE_ANON_KEY"
```

## 🚀 Usage

### 1. Web Application (UI)
대시보드를 실행하여 시각적인 인터페이스로 진단할 수 있습니다.

```bash
python -m streamlit run app.py
```
브라우저에서 `http://localhost:8501`로 접속하세요.

### 2. Test Scenario (Script)
터미널에서 전체 로직을 검증하려면 아래 스크립트를 실행하세요.
```bash
python test_scenario.py
```

## 🧠 Core Logic

- **OWRK Method**: 
  - 물(Polar)과 다이아이오도메탄(Dispersive) 두 가지 액체의 접촉각을 측정하여 표면 에너지를 계산합니다.
- **Scoring System**:
  - `Total Score = (Predicted Holding Time) * (Clean Removal Probability)`
  - 잔사(Residue) 위험이 높은 제품은 강력한 페널티를 부여하여 추천에서 제외됩니다.

## ⚠️ Notes
- `libs/` 폴더 내의 일부 모듈은 데모를 위해 **Mock(가상) 모드**와 호환되도록 설계되었습니다. 실제 센서가 없어도 동작합니다.
- 실제 사용 시 `models/mobile_sam.pt` 파일이 필요할 수 있습니다.

## 🧪 테스트 환경 및 Real AI 모드 (Real AI Mode)
이 레포지토리는 이제 MobileSAM을 활용한 **실제 AI 추론(Real AI Inference)**을 지원합니다.

- **AI 모델**: `models/mobile_sam.pt` (MobileSAM 가중치)가 포함되어 있습니다. 시스템은 이 모델을 로드하여 CPU/GPU에서 실제로 추론을 수행합니다.
- **테스트 데이터**: `test_images/` 폴더에 실제 테스트용 **실사 이미지(Real Photos)**가 포함되어 있습니다.
    - `surface_test_image.jpg`: 피착제 표면
    - `drop_test_image.jpg`: 액적(물방울/다이아이오도메탄)
    - `test_scenario.py` 실행 시 이 이미지들을 사용하여 실제 분석을 수행합니다.

## 🌐 배포 (Deployment)
이 프로젝트는 **Streamlit Community Cloud** 배포에 최적화되어 있습니다.