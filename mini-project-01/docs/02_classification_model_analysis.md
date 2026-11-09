# 004. 모델 실험 및 오류 분석

**프로젝트:** 온라인 쇼핑 구매 여부 예측 (PyTorch 신경망 분류)  
**작성 범위:** CP-04~CP-08

> Notebook에서 확인한 실제 수치와 관찰을 기록합니다. `___`는 실제 값 또는 1~2문장으로 바꾸세요. Precision·Recall·F1은 구매 클래스(`Revenue=1`) 기준입니다.

## 1. 기준 모델 및 평가 지표 (CP-04)

- **양성 클래스:** 구매 (`Revenue=1`)  
  - 모델이 찾아내야 하는 대상이 "구매 세션"이라는 뜻

- **주 평가 지표:** F1 (구매 클래스)  
  - Precision과 Recall을 함께 반영한 값  
  - 구매가 15.5%뿐이라 Accuracy 대신 F1을 기준으로 삼음 

- **보조 지표:** Precision, Recall, Accuracy  
  - Precision: 구매라고 예측한 것 중 실제 구매 비율  
  - Recall: 실제 구매 중 찾아낸 비율  
  - Accuracy: 전체 예측 중 맞힌 비율  

- **모델 비교에 사용한 데이터:** Train, Validation (Validation 기준으로 비교)  
  - Train은 학습한 데이터라 결과가 좋게 나오기 쉬움  
  - 처음 보는 데이터인 Validation으로 비교해야 공정함  
  - Test는 마지막 최종 평가용으로 아껴 둠  

- **DummyClassifier — Validation Accuracy / F1:** 0.845 / 0.000  
  - 모두 미구매로 찍어서 Accuracy는 높지만 구매를 하나도 못 찾아 F1은 0

- **LogisticRegression — Train F1 / Validation F1:** 0.505 / 0.49  
  - 두 값이 비슷해서 과적합은 크지 않음  
  - 다만 F1 자체가 낮아 구매 예측 성능은 아직 부족함  
- **LogisticRegression — Validation Precision / Recall:** 0.734 / 0.367  
  - 구매라고 예측한 것의 약 73%는 실제 구매  
  - 실제 구매 고객은 약 37%만 찾아냄 (약 63%는 놓침)  
- **LogisticRegression의 Validation 혼동행렬:** TN 1526 / FP 38 / FN 181 / TP 105  
  - TN: 미구매를 미구매로 맞힘  
  - FP: 미구매를 구매로 잘못 예측  
  - FN: 구매를 미구매로 잘못 예측 (놓친 구매 고객)  
  - TP: 구매를 구매로 맞힘  
  - FN(181)이 TP(105)보다 많아서 Recall이 낮음  

**분석:** 불균형 데이터에서 기준 모델의 Accuracy와 F1이 다르게 나타나는 이유를 작성합니다.  
- DummyClassifier는 모든 세션을 미구매로 예측해서, 다수 클래스(84.5%)만 맞혀 Accuracy가 0.845로 높게 나옴
- 하지만 구매 세션을 하나도 찾지 못해 구매 클래스의 Recall과 F1이 0이 되므로, Accuracy만으로는 이 사실이 드러나지 않음
___

## 2. PyTorch MLP-A 구현 및 학습 (CP-05)

- **신경망 구조:** 입력 → 64 → 32 → 1 (ReLU, Dropout 0.1)  
  - 입력 특성 73개를 64개, 32개로 줄여 가며 학습하고, 마지막에 구매 점수 1개를 출력함  
  - ReLU: 복잡한 패턴을 학습할 수 있게 하는 활성화 함수  
  - Dropout 0.1: 학습 중 노드의 10%를 무작위로 꺼서 과적합을 줄임  

- **손실 함수 / 옵티마이저:** `BCEWithLogitsLoss` / Adam  
  - `BCEWithLogitsLoss`: 구매/미구매 이진 분류용 손실 함수 (Sigmoid 포함)
  - Adam: 손실이 줄어드는 방향으로 가중치를 갱신하는 방법

- **학습 조건:** 최대 25 epoch, batch size 256, learning rate 0.001
  - 25 epoch: 전체 Train 데이터를 최대 25바퀴 학습
  - batch size 256: 256개씩 묶어서 가중치를 갱신
  - learning rate 0.001: 한 번에 가중치를 바꾸는 크기

- **Validation F1이 가장 높았던 에포크:** 25
  - 마지막 에포크에서 최고라 아직 성능이 오르는 중임

- **선택된 MLP-A의 Train F1 / Validation F1:** 0.709 / 0.633
  - 두 값의 차이가 약 0.08로 크지 않아 과적합은 심하지 않음
  - LogisticRegression의 Validation F1(0.49)보다 높음

- **MLP-A Validation Precision / Recall:** 0.717 / 0.566
  - 구매라고 예측한 것의 약 72%가 실제 구매
  - 실제 구매 고객의 약 57%를 찾아내고, 약 43%는 놓침

**분석:** Train loss와 Train·Validation F1 그래프에서 확인한 학습 경향을 작성합니다.  

- Train loss는 초반에 빠르게 줄다가 이후 완만하게 계속 감소함
- Train F1과 Validation F1은 초반(2~4 epoch)에 크게 오른 뒤 천천히 상승하며, 둘의 차이는 약 0.08로 크지 않아 과적합이 심하지 않음
- 최고 Validation F1이 마지막인 25 epoch에서 나왔고 아직 오르는 중이라, epoch을 늘리면 더 좋아질 여지가 있음
- LogisticRegression의 Validation F1(0.49)보다 높아 신경망이 구매 예측을 더 잘함
___

## 3. 클래스 가중 손실과 MLP 비교 (CP-06)

- **MLP-B 변경 사항:** 양성 클래스 가중치(`pos_weight`)를 적용한 `BCEWithLogitsLoss`
  - 구매를 틀렸을 때의 손실을 더 크게 계산해, 소수 클래스인 구매를 더 신경 쓰게 함
- **Train 데이터의 `pos_weight`(미구매 수 ÷ 구매 수):** 5.46
  - 구매 1건의 오답을 미구매 약 5.5건의 오답만큼 크게 취급함
- **MLP-B의 최적 에포크:** 17
  - MLP-A(25)보다 초반에 F1이 빠르게 올라가지만(1 epoch: 약 0.47 vs 0.0), 10 epoch 이후에는 두 모델의 Validation F1이 거의 비슷함
- **MLP-B Train F1 / Validation F1:** 0.692 / 0.645
  - 차이가 약 0.05로 과적합은 크지 않음
- **MLP-B Validation Precision / Recall:** 0.575 / 0.734
  - 구매라고 예측한 것의 약 58%가 실제 구매
  - 실제 구매 고객의 약 73%를 찾아냄
- **MLP-A와 MLP-B의 Validation F1 차이:** 0.012 (0.633 → 0.645)
  - MLP-B가 약간 높지만 차이가 작아, 그래프에서도 10 epoch 이후 두 선이 거의 겹침
- **LogisticRegression과 두 MLP의 성능 비교에서 발견한 점:**
  - Validation F1: LogisticRegression 0.490 / MLP-A 0.633 / MLP-B 0.645
  - 두 MLP가 LogisticRegression보다 확실히 높음
  - LogisticRegression은 Precision(0.734)은 높지만 Recall(0.367)이 낮아, 구매 고객을 많이 놓침
  - Accuracy는 LogisticRegression(0.882)과 두 MLP(0.875~0.898)가 비슷해, F1로 봐야 차이가 드러남
- **Train·Validation F1 차이에서 관찰한 점:**
  - MLP-A 0.077 / MLP-B 0.047 / LogisticRegression 0.016
  - MLP-A의 차이가 가장 커서 과적합 경향이 가장 큼
  - LogisticRegression은 차이는 작지만 F1 자체가 낮아, 구매 패턴을 충분히 학습하지 못한 것으로 보임


**분석:** 양성 클래스 가중치를 적용한 뒤 Precision·Recall·F1이 어떻게 달라졌는지 작성합니다.  
- Recall이 0.566에서 0.734로 크게 올라 구매 고객을 더 많이 찾게 됨
- 대신 Precision이 0.717에서 0.575로 내려가, 구매가 아닌 세션을 구매로 잘못 예측하는 경우가 늘어남
- Recall 상승이 Precision 하락을 상쇄해 F1은 0.633에서 0.645로 소폭 오름 (Accuracy는 0.898에서 0.875로 내려감)

___

> 전체 실험 결과는 Notebook의 `results_df`와 `results/02_model_comparison.csv`를 참고합니다.

## 4. 예측 오류 및 임계값 분석 (CP-07)

- **비교한 두 신경망:** MLP-A / MLP-B
  - MLP-A: 기본 BCE, MLP-B: 구매에 가중치를 준 weighted BCE
- **기본 임계값 0.5에서 MLP-A의 FP / FN:** 64 / 124
  - FP: 미구매인데 구매로 잘못 예측한 세션 수
  - FN: 구매인데 놓친 세션 수
- **기본 임계값 0.5에서 MLP-B의 FP / FN:** 155 / 76
  - MLP-A보다 FP는 91 늘고 FN은 48 줄어, 더 많은 구매 고객을 찾는 대신 오탐이 늘어남
- **임계값 0.3에서 Precision·Recall 및 FP·FN 변화:** (0.5 -> 0.3 기준)
  - MLP-A: Precision 0.717 -> 0.608, Recall 0.566 -> 0.678, FP 64 -> 125, FN 124 -> 92
  - MLP-B: Precision 0.575 -> 0.459, Recall 0.734 -> 0.839, FP 155 -> 283, FN 76 -> 46
  - 임계값을 낮추면 구매로 예측하는 기준이 느슨해져 Recall은 오르고 Precision은 내려감
- **임계값 0.7에서 Precision·Recall 및 FP·FN 변화:** (0.5 -> 0.7 기준)
  - MLP-A: Precision 0.717 -> 0.835, Recall 0.566 -> 0.353, FP 64 -> 20, FN 124 -> 185
  - MLP-B: Precision 0.575 -> 0.639, Recall 0.734 -> 0.612, FP 155 -> 99, FN 76 -> 111
  - 임계값을 높이면 확실한 경우만 구매로 예측해 Precision은 오르고 Recall은 내려감
- **Validation F1이 가장 높은 MLP·임계값 조합:** MLP-B, 임계값 0.5 (F1 0.645)
  - Precision 0.575 / Recall 0.734
  - 2위는 MLP-A, 임계값 0.3 (F1 0.641)으로 차이가 0.004뿐임
- **선택한 MLP의 최적 에포크 수:** 17
  - 선택한 MLP-B의 Validation F1이 가장 높았던 에포크
- **(선택) PR 곡선 또는 추가 임계값 실험에서 확인한 점:**
  - MLP-A는 임계값에 따라 F1이 0.496 ~ 0.641로 크게 변하지만, MLP-B는 0.593 ~ 0.645로 변화가 작음
  - MLP-A를 0.3으로 낮춘 결과(Precision 0.608 / Recall 0.678)가 MLP-B 0.5(0.575 / 0.734)와 비슷해, 가중치를 주는 것과 임계값을 낮추는 것이 비슷한 효과를 냄


**분석:** 모델과 임계값의 선택에서 Precision·Recall 및 FP·FN을 어떻게 고려했는지 작성합니다.  

- 주 평가 지표인 F1은 Precision과 Recall을 함께 보므로, FP와 FN이 모두 너무 많지 않은 조합을 골랐음
- 임계값 0.3은 구매 고객을 더 찾지만(MLP-B FN 46) 오탐이 크게 늘고(FP 283), 0.7은 오탐이 적지만(MLP-A FP 20) 구매 고객을 많이 놓침(FN 185)
- 그래서 Test 결과를 보기 전에, Validation F1이 가장 높은 MLP-B와 임계값 0.5를 최종으로 선택함
- 선택 후 Test에서도 F1 0.649(Validation 0.645)로 비슷했고, 구매 고객 286명 중 241명을 찾음(Recall 0.843, FN 45, FP 216)

___

> 임계값별 상세 결과는 `threshold_df`와 `results/02_threshold_comparison.csv`를 참고합니다.

## 5. 최종 신경망과 Test 평가 (CP-08)

- **Test 확인 전에 확정한 MLP 설정 / 임계값:** MLP-B (weighted BCE) / 0.5
  - 구조는 64 -> 32 -> 1, Dropout 0.1
  - Validation F1이 가장 높은 조합이라 선택함
- **최종 재학습 데이터(Train + Validation):** 10,480행
  - 8,630(Train) + 1,850(Validation)
  - 전처리도 이 데이터로 다시 학습했고, Test에는 변환만 적용함
- **최종 재학습 에포크 수 / 가중 손실 적용 여부:** 17 / 적용
  - 17은 MLP-B의 Validation F1이 가장 높았던 에포크
  - `pos_weight`는 Train+Validation 기준으로 다시 계산함
- **Test Accuracy / Precision / Recall / F1:** 0.859 / 0.527 / 0.843 / 0.649
  - 구매라고 예측한 것의 약 53%가 실제 구매
  - 실제 구매 고객의 약 84%를 찾아냄
- **Test 혼동행렬:** TN 1348 / FP 216 / FN 45 / TP 241
  - 구매 고객 286명 중 241명을 찾고 45명을 놓침
  - 미구매 세션 216건을 구매로 잘못 예측함
- **모델 번들(`02_final_classifier.joblib`)·설정 JSON 저장 및 재로딩 확인:** 저장 후 다시 불러와 Test 예측이 저장 전과 완전히 일치함 (`True`)
  - 번들에는 전처리와 모델 가중치가 함께 들어 있음
- **`predict_purchase()` 실행 결과:** 구매 확률 0.0058, 예측 0 (임계값 0.5), 실제 Revenue 0
  - Test 첫 번째 세션 1건 기준이며, 함수가 끝까지 동작하는지 확인한 용도

**해석:** 최종 Test 결과와 동일한 모델·임계값의 Validation 결과를 비교해 알게 된 점을 작성합니다.  

- F1은 Test 0.649, Validation 0.645로 거의 같아서, Validation으로 고른 모델과 임계값이 Test에서도 비슷하게 동작함
- 다만 내용은 달라져서 Recall은 0.734에서 0.843으로 올랐고, Precision은 0.575에서 0.527로, Accuracy는 0.875에서 0.859로 내려감
- Validation 값은 Train만으로 학습한 모델, Test 값은 Train+Validation으로 재학습한 모델의 결과라서 완전히 같은 모델의 비교는 아님
___

> Test 결과는 모델 선택에 사용하지 않습니다. 최종 Test 수치는 Notebook과 이 문서에만 기록합니다.

## 6. 실험 요약

- **가장 중요한 실험 결과:**
  - Validation F1이 LogisticRegression 0.490, MLP-A 0.633, MLP-B 0.645로, 신경망이 기준 모델보다 구매 고객을 훨씬 잘 찾음
  - 구매에 가중치를 준 MLP-B는 Recall이 0.566에서 0.734로 올라 구매 고객을 더 많이 찾음
  - 최종 모델의 Test F1은 0.649로 Validation과 비슷했음
- **남아 있는 모델의 한계:**
  - Test Precision이 0.527이라, 구매라고 예측한 세션의 약 47%(216건)가 실제로는 구매가 아님
  - 구매 고객 45명(약 16%)은 여전히 놓침
  - 임계값을 0.3, 0.5, 0.7 세 가지만 비교했고, 상위 조합 간 F1 차이(0.645 vs 0.641)가 작아 선택이 확실하다고 말하기 어려움
  - `PageValues`는 세션이 끝나기 전에는 알기 어려워, 실시간 예측에 그대로 쓰기 어려움


> **최종 MLP 설정·임계값의 선택과 근거**는 `adr.md`의 의사결정 ④에 작성합니다.
