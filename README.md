# Neurosimulator 

LTP/LTD 실험 데이터(엑셀)를 NeuroSim 모델에 피팅해 실제 소자 특성을 반영한 펄스 단위 업데이트 옵티마이저로 MNIST 분류기를 학습하는 코드입니다.
실험 데이터로 모델 파라미터(A, B)를 추정하고, 그 결과를 반영한 NeuroSimOptimizer로 학습/평가/시각화를 수행합니다.

+ ## 주요 기능
  + 1. 엑셀에서 LTP/LTD 데이터 자동 로드 및 분리 (Conductance, WriteVH, PulseNum)
  + 2. 안정화 구간만 선택해 피팅 품질 향상(데이터 상위 비율 사용)
  + 3. NeuroSim LTP/LTD 폐곡선 모델 파라미터(A, B) 추정
  + 4. 펄스 정수화/포화/양자화가 반영된 커스텀 옵티마이저
  + 5. MNIST 학습, 정확도 로그 저장, 혼동행렬/가중치 분포/정확도 곡선 시각화

+ ## 환경 요구 사항
  + 1. Python 3.10+ (권장 3.12)
  + 2. macOS/Linux/Windows
  + 3. CUDA가 있다면 자동 사용

+ ## 필요 패키지

```bash

pandas  
matplotlib  
numpy  
torch, torchvision  
scipy  
scikit-learn  
seaborn  
openpyxl  

```


+ ## 데이터 형식 (엑셀)
엑셀 파일에 최소 다음 3개 열이 필요합니다. 
  + 1. Conductance
  + 2. WriteVH
  + 3. PulseNum (실측 컨덕턴스, S)>0 이면 LTP, <0 이면 LTD 구분펄스 번호(증가)
    + 3-1. WriteVH 양수(LTP), 음수(LTD)로 자동 분리합니다.
    + 3-2.LTP/LTD 각각을 PulseNum 오름차순으로 사용합니다.
 
+ ## 실행 방법

  + 1. 저장소 루트 기준(예: 파일이 github_code/neurosimulator.py에 있는 경우):
python github_code/neurosimulator.py

  + 2. 실행하면 엑셀 파일 경로 입력을 요청합니다. 예)
LTP/LTD 데이터가 포함된 엑셀 파일 경로를 입력하세요:  /Users/you/data/ltp_ltd.xlsx

+ ## 하이퍼파라미터
 ```bash

  
TARGET_RANGE = (-1.0, 1.0)      # 가중치 스케일 범위
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-4
PULSE_SCALING_FACTOR = 300      # 펄스 민감도(정수화 이전 스케일)
LTP_FIT_RATIO = 1.0             # 피팅에 쓸 상위 비율(안정화 구간)
LTD_FIT_RATIO = 1.0 

```


  + LTP_FIT_RATIO, LTD_FIT_RATIO를 0.7~0.95로 조절하면 초기 불안정 구간을 제외하고 피팅 품질을 올릴 수 있습니다.
  + PULSE_SCALING_FACTOR는 “이상적인 실수 변화량 → 실제 정수 펄스 수” 매핑의 강도를 조정합니다.

+ ## 결과물

실행 후 다음을 확인할 수 있습니다.
  + 학습/테스트 정확도 로그, 최종 정확도 출력
  +  시각화
  + NeuroSim 피팅 곡선(실측 스케일)
  + 학습 전/후 가중치 분포(실측 스케일로 역변환)
  + 정확도 곡선
  + 혼동행렬
  + mnist_accuracy_log.xlsx : 에포크별 테스트 정확도 저장

+ ## 코드 구조
  + read_split_by_write_vh(file_path): 엑셀 로드 & LTP/LTD 분리
  + NeuroSimFitter: 실측 데이터를 스케일링하고 LTP/LTD 각각 A, B 파라미터 피팅
  + NeuroSimOptimizer: 펄스 정수화/포화 반영한 커스텀 옵티마이저
  + SimpleNet: 간단한 2-층 MLP (MNIST)
  + train/test: 학습/평가 루프
  + 시각화 유틸: 정확도/혼동행렬/가중치 분포

+ ## 사용 팁
  + GPU 사용: CUDA 환경이면 자동으로 cuda 사용
  + 데이터 전처리: PulseNum이 반드시 증가 순, LTP/LTD 각각 충분한 샘플 권장(≥ 수십 포인트)
	

