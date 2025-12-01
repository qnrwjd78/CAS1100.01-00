# Steam Game Data Analysis Project

이 프로젝트는 Steam 게임 데이터를 수집 및 분석하여 개발자에게 도움이 되는 질문에 대한 결과를 도출하는 프로젝트입니다.

## 1. 환경 설정 (Installation)

Miniconda를 사용하여 가상환경을 생성하고 필요한 패키지를 설치합니다.

### 1.1 Conda 환경 생성 및 활성화
```bash
conda create -n steam_analysis python=3.9
conda activate steam_analysis
```

### 1.2 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

## 2. 데이터셋 준비 (Data Setup)

분석을 위해 Steam Reviews 데이터셋이 필요합니다.

1. **데이터셋 다운로드**: [Steam Reviews Dataset (Kaggle)](https://www.kaggle.com/datasets/forgemaster/steam-reviews-dataset)
2. **압축 해제 및 이동**:
   - 다운로드 받은 파일의 압축을 풉니다.
   - `reviews-*.csv` 파일들이 포함된 폴더의 내용물을 프로젝트 내 `data/reviews` 폴더에 위치시킵니다.
   - 폴더 구조가 다음과 같아야 합니다:
     ```
     Project/
     ├── data/
     │   └── reviews/
     │       ├── reviews-123.csv
     │       ├── reviews-456.csv
     │       └── ...
     ├── load_data.py
     ├── ...
     ```
   - *참고: `data/reviews` 폴더가 없다면 직접 생성해주세요.*

- *만약 데이터 셋이 없다면 미리 전처리 해둔 data/merged_sampled.csv를 사용할 수 있습니다.*

## 3. 실행 방법 (Usage)

데이터 수집부터 분석, 시각화까지 순서대로 실행합니다.

### 3.1 데이터 수집 및 전처리
리뷰 데이터에서 App ID를 추출하고, Steam API 및 웹 크롤링을 통해 메타데이터를 수집합니다.
```bash
python load_data.py
```
- 결과물: `data/merged_sampled.csv`

### 3.2 데이터 분석
수집된 데이터를 바탕으로 6가지 개발자 지향 질문에 대한 분석을 수행합니다.
```bash
python process_data.py
```
- 결과물: `data/analysis_results.json`

#### 분석 질문 목록 (Questions)
1. **최적 출시가 구간은?** (Q1): 가격대별 평점 분포와 통계적 차이를 분석합니다.
2. **출시 직후 리뷰 속도 목표는?** (Q2): 출시 후 90일간의 일일 리뷰 수 분포를 통해 목표치를 설정합니다.
3. **초기 참여도 목표는?** (Q3): 소유자 대비 리뷰 작성 비율(참여도)의 분포를 확인합니다.
4. **가격이 평점에 주는 영향은?** (Q4): 가격과 긍정적 평가 비율 간의 상관관계를 분석합니다.
5. **플레이타임이 평점에 주는 영향은?** (Q5): 평균 플레이타임과 평점 간의 관계를 살펴봅니다.
6. **판매 규모가 평점에 주는 영향은?** (Q6): 소유자 수(판매 규모)와 평점 간의 상관성을 확인합니다.

### 3.3 결과 시각화
분석 결과를 그래프로 시각화하여 저장합니다.
```bash
python viz_result.py
```
- 결과물: `data/plots/*.png`

## 4. 프로젝트 구조
- `loaders/`: 데이터 수집 모듈 (API, Web, Review 등)
- `utils.py`: 공통 유틸리티 함수
- `load_data.py`: 데이터 수집 메인 스크립트
- `process_data.py`: 데이터 분석 메인 스크립트
- `viz_result.py`: 시각화 스크립트
