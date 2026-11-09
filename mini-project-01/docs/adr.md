# ADR — 핵심 의사결정 기록

**프로젝트:** 모듈 1 ML/DL 기초 미니프로젝트 — 회귀·분류

> 실험 결과를 반복해서 옮겨 적지 않습니다. **선택 → 근거 → 한계**를 중심으로 각 항목을 1~2문장으로 작성합니다.
>
> 이 파일은 프로젝트 전체의 의사결정을 순서대로 쌓는 기록입니다. 1일차(회귀)는 의사결정 ①·②, 2일차(분류)는 ③·④를 아래에 이어서 작성합니다.

---

## 1일차 — 회귀 (공유자전거 대여 수요 예측)

### 의사결정 ① 입력 특성 선택 (CP-02)

**문제**

예측 시점에 사용할 수 없거나 누수가 발생할 수 있는 특성이 있습니다.

- **검토한 선택지:** (예: `casual`·`registered` 포함 여부, `atemp` 포함 여부, `windspeed` 제거 여부)
  - casual + registered == cnt가 모든 행에서 성립해서, 두 값을 입력으로 쓰면 정답을 미리 알려 주는 것과 같아서 제거
  - temp와 atemp는 0.992로 거의 같은 정보이지만 둘 다 사용함(트리 기반 모델 영향적음)
  - windspeed 도 제거 하지 않음
- **최종 선택(사용·제외한 특성):**
  - 사용: season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed (12개)
  - 직접 만든 특성: is_commute_hour, hr_sin, hr_cos (3개) -> 모델 입력은 총 15개
  - 제외: casual, registered (누수), instant (행 번호), dteday (날짜)
- **선택 근거(누수 확인, 예측 시점, 상관관계 등):**
  - 누수 확인: casual + registered == cnt가 모든 행에서 성립해서, 두 값을 입력으로 쓰면 정답을 미리 알려 주는 것과 같음 (registered와 cnt의 상관은 0.972)
  - 예측 시점: casual, registered는 그 시간의 대여가 끝나야 알 수 있는 값이라 예측 시점에는 사용할 수 없음
  - instant, dteday: 행 번호와 날짜이고, 날짜 정보는 yr, mnth, weekday, hr 등으로 이미 들어 있어 따로 쓰지 않음
  - 상관관계: temp 0.398, hum -0.301, windspeed 0.094로, windspeed는 직선 관계가 거의 없지만 상관계수는 직선 관계만 보여 주므로 낮다는 이유만으로 제거하지 않음
  - temp와 atemp는 0.992로 거의 같은 정보지만, 트리 기반 모델(RandomForest)은 중복 변수에 크게 영향받지 않아 둘 다 사용함
- **감수한 한계 또는 추후 확인할 점:**
  - 감수한 한계: temp와 atemp가 중복되어 두 변수가 중요도를 나눠 가져, 각각의 중요도를 해석하기 어려움
  - 추후 확인할 점: atemp나 windspeed를 제거했을 때 성능이 달라지는지 확인
  - 추후 확인할 점: temp, hum 같은 날씨 특성은 실제 예측 시점에는 예보 값을 써야 하므로, 예보 오차가 성능에 주는 영향 확인



**분석 근거:** `01_regression_data_analysis.md`

### 의사결정 ② 최종 회귀 모델 선택 (CP-07)

**문제**

서로 다른 회귀 모델 및 설정 중 하나를 최종 모델로 선택해야 합니다.

- **비교한 모델·설정:** 
  - DummyRegressor, LinearRegression, DecisionTree(깊이 3, 6, 12), RandomForest A(n=50, depth=10), RandomForest B(n=100, depth=15)
  - Validation MAE가 낮은 상위 2개인 RandomForest B와 A를 최종 후보로 비교함
- **최종 선택한 모델·설정:** RandomForest A (n_estimators=50, max_depth=10)
- **선택 근거(Validation MAE, 과대적합 가능성, 시간대별 오류 등):**
  - Validation MAE는 B 46.091, A 47.494로 차이가 1.403뿐이고, RMSE(72.282 vs 72.799)와 R2(0.895 vs 0.893)도 B와 A가 거의 같음
  - B는 Train MAE가 10.851로 A(20.101)보다 훨씬 낮아 격차(Validation MAE - Train MAE)가 35.240으로 A의 27.393보다 큼. B가 Train 데이터에 더 맞춰져 과대적합 가능성이 더 높음
  - 시간대별 MAE는 두 모델의 패턴이 거의 같음 (17~18시에 가장 큼). B가 24개 시간대 중 18개에서 더 낮지만 차이는 6.4 이하이고, 18시와 19시 등에서는 A가 더 낮음
  - 실제값·예측값 산점도도 두 모델의 모양이 거의 같아, 오류 분석에서 B가 확실히 낫다는 근거가 나오지 않음
  - 그래서 성능 차이가 작은데 과대적합 가능성이 더 높은 B 대신 A를 선택함 (Test 결과는 선택에 사용하지 않음)
- **선택 모델의 장점:**
  - B와 성능이 거의 같으면서 설정이 더 단순함 (트리 50개, 깊이 10)
  - Train과 Validation의 격차가 B보다 작아, 처음 보는 데이터에서도 안정적으로 동작할 가능성이 더 높음
- **남아 있는 한계 또는 개선 방향:**
  - 한계: 실제값이 큰 구간(약 600 이상)에서 실제보다 낮게 예측하는 경향이 있고, 17~18시 퇴근 시간대의 MAE가 약 120으로 가장 큼
  - 한계: A와 B의 차이가 작아 선택이 결정적이라고 말하기 어려움
  - 개선 방향: 트리 수와 깊이를 더 다양하게 비교하고, 대여량이 큰 시간대의 오차를 줄일 수 있는 특성이나 모델을 검토


**분석 근거:** `01_regression_model_analysis.md`

---

## 2일차 — 분류 (온라인 쇼핑 구매 여부 예측 · PyTorch MLP)

### 의사결정 ③ 평가 지표 선택 (CP-04)

**문제**

구매 세션이 전체의 약 15%뿐인 불균형 데이터이므로, Accuracy만으로 구매 클래스를 얼마나 잘 예측하는지 판단하기 어렵습니다.

- **검토한 선택지:** ___ (예: Accuracy, 구매 클래스 F1, Precision, Recall)
    - Accuracy: 전체 예측 중 맞힌 비율
    - Precision: 구매라고 예측한 것 중 실제 구매 비율
    - Recall: 실제 구매 고객 중 찾아낸 비율
    - F1: Precision과 Recall의 조화평균

- **최종 선택한 주 지표와 보조 지표:** 
    - 주 지표는 구매 클래스 F1
    - 보조 지표는 Precision, Recall, Accuracy

- **선택 근거(DummyClassifier와 LogisticRegression 결과, FP·FN 등):** ___
    - DummyClassifier: 모두 미구매로 예측해 Validation Accuracy가 0.845로 높지만, 구매를 하나도 찾지 못해 F1은 0.000임
    - LogisticRegression: Validation Accuracy 0.882로 높지만 Precision 0.734 / Recall 0.367 / F1 0.490임
    - 혼동행렬은 TN 1526 / FP 38 / FN 181 / TP 105로, 구매 고객 286명 중 181명(약 63%)을 놓침
    - Accuracy로는 이 문제가 드러나지 않아, Precision과 Recall을 함께 반영하는 F1을 주 지표로 선택함

- **감수한 한계 또는 추후 확인할 점:** ___
  - 감수한 한계: Recall을 높이면 미구매 세션을 구매로 잘못 예측하는 FP가 늘어 Precision이 떨어질 수 있음
  - 추후 확인할 점: 클래스 가중치 등 불균형 대응으로 Recall을 개선할 수 있는지 확인 (의사결정 ④)


**분석 근거:** `02_classification_data_analysis.md`, `02_classification_model_analysis.md`

### 의사결정 ④ PyTorch MLP 설정 및 분류 임계값 선택 (CP-07)

**문제**

기본 손실 함수로 학습한 MLP-A와 클래스 가중 손실을 적용한 MLP-B를 비교하고, 구매 확률을 0/1로 구분하는 임계값(0.3·0.5·0.7)을 선택해야 합니다.

- **검토한 선택지:** MLP-A(기본 BCE, 최적 에포크 25) / MLP-B(weighted BCE, `pos_weight` 5.46, 최적 에포크 17), 임계값 0.3·0.5·0.7 (총 6개 조합)
- **최종 선택한 MLP 설정·임계값·재학습 에포크:** MLP-B (weighted BCE), 임계값 0.5, 재학습 에포크 17
    - 구조는 입력 -> 64 -> 32 -> 1 (ReLU, Dropout 0.1), Adam(learning rate 0.001)
    - 재학습은 Train+Validation 10,480행으로 진행함
- **선택 근거(Validation F1, Precision·Recall, FP·FN 및 가중 손실의 영향):**
    - MLP-B 0.5가 Validation F1 0.645로 6개 조합 중 가장 높음 (Precision 0.575 / Recall 0.734, FP 155 / FN 76)
    - 2위 MLP-A 0.3(F1 0.641, FP 125 / FN 92)과 차이는 0.004로 작음. MLP-B 0.5는 놓친 구매 고객이 더 적고(FN 76 vs 92) 오탐은 더 많음(FP 155 vs 125)
    - 임계값 0.3: MLP-B는 Recall 0.839(FN 46)지만 FP가 283으로 늘어 F1이 0.593으로 내려감
    - 임계값 0.7: MLP-A는 Precision 0.835(FP 20)지만 Recall 0.353(FN 185)이라 F1이 0.496으로 내려감
    - 가중 손실의 영향(0.5 기준, MLP-A -> MLP-B): Recall 0.566 -> 0.734, Precision 0.717 -> 0.575, F1 0.633 -> 0.645
    - Train·Validation F1 차이는 MLP-B 0.047, MLP-A 0.077임
- **LogisticRegression과 비교했을 때 선택한 MLP의 특징:**
    - LogisticRegression은 Precision 0.734로 높지만 Recall 0.367이라 구매 고객 181명을 놓침 (F1 0.490)
    - MLP-B는 Precision이 0.575로 낮아졌지만 놓친 구매 고객이 76명으로 줄어 Recall 0.734, F1 0.645를 얻음
- **감수한 한계 또는 추후 개선 방향:**
    - 감수한 한계: Precision이 0.575라 구매 예측 중 오탐(FP 155건)이 적지 않음
    - 감수한 한계: 상위 조합 간 F1 차이(0.645 vs 0.641)가 작아 선택이 결정적이라고 말하기 어려움
    - 추후 개선: 임계값을 세 가지만 비교했으므로 더 촘촘한 값이나 PR 곡선으로 확인

**분석 근거:** `02_classification_model_analysis.md`
