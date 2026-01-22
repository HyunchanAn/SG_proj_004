# Streamlit Community Cloud 배포 가이드

프로젝트 배포 준비가 완료되었습니다! 아래 절차를 따라 웹에 무료로 배포해보세요.

## 1. GitHub 준비
1.  **코드 푸시(Push)**: 현재 변경된 모든 파일(특히 `packages.txt`)이 GitHub 레포지토리에 업로드되어 있어야 합니다.
    ```bash
    git add .
    git commit -m "배포 준비 완료"
    git push origin main
    ```

## 2. Streamlit Cloud 배포
1.  [share.streamlit.io](https://share.streamlit.io/)에 접속합니다.
2.  GitHub 계정으로 **로그인**합니다.
3.  **"New app"** 버튼을 클릭합니다.
4.  **레포지토리 연결**:
    - Repository: `SG_RADAR` (또는 본인의 레포지토리명) 선택.
    - Branch: `main`.
    - Main file path: `app.py` (**주의**: `streamlit_app.py`가 입력되어 있다면 지우고 `app.py`로 고쳐주세요).
5.  **"Deploy!"** 클릭.
    *   *Private 레포지토리의 경우*: Streamlit이 권한 승인을 요청할 수 있습니다. "Authorize" 또는 "Grant Access"를 눌러주시면 코드는 비공개로 유지된 채 앱만 배포됩니다.

## 3. 중요 참고사항
- **더미 데이터**: 배포된 앱도 로컬과 동일하게 기본 설정된 더미 데이터 및 AI 모델을 사용합니다.
- **AI 모델 파일**: `models/mobile_sam.pt` 파일(약 40MB)이 GitHub에 정상적으로 올라갔는지 확인해주세요. 이 파일이 없으면 배포 시 에러가 나거나 Mock 모드로 동작할 수 있습니다.

## 트러블슈팅
- 만약 `ImportError: libGL.so.1` 같은 에러가 발생한다면, `packages.txt` 파일이 제대로 푸시되지 않았거나 인식되지 않은 것입니다. 레포지토리 루트 경로에 해당 파일이 있는지 확인해주세요.
