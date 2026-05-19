<h2>DMMN - Deep Matching Model with feature-aware external Memory Network</h2>

리뷰 텍스트와 메타데이터 간 matching degree를 외부 메모리 네트워크로 학습하는 멀티모달 베이스라인

<hr>

<h2>Architecture</h2>
<p align="center">
  <img src="DMMN_picture.png" width="800" />
</p>
<p align="center">
  <i>Wang, S., & Qiu, J. (2023). Utilizing a feature-aware external memory network for helpfulness prediction in e-commerce reviews. Applied Soft Computing, 148, 110923.</i>
</p>

<hr>

<h2>Overview</h2>

DMMN은 리뷰 유용성 예측에서 리뷰 텍스트와 메타데이터 간의 matching degree 관점을 도입하고, 외부 메모리 네트워크(External Memory Network, EMN)에 item의 전형적 특성을 저장, 갱신, 매칭함으로써 두 모달 사이 상호작용을 학습할 수 있음을 입증한 모델이다. 원논문에서 item은 Amazon 상품을 의미하지만, 본 구현에서는 데이터 도메인에 맞춰 작성자 메타데이터(item = Meta)로 적용했다.

<hr>

<h2>Input Modality</h2>

- 2개 모달: 텍스트 + 메타데이터

<h3>텍스트 모달리티</h3>

- clean_review를 NLTK tokenizer로 토큰화
- 사전학습된 Word2Vec (300차원)으로 초기화한 Embedding layer
- Bidirectional LSTM (150 units, return_sequences=True)으로 양방향 문맥 추출
- 시간축에 대해 mean pooling -> 300차원 리뷰 특징 벡터 review_feature (rf)
- Dropout 적용

<h3>메타데이터 모달리티</h3>

- 6개 feature
  1. 작성자 누적 리뷰 수
  2. 보유 게임 수
  3. 총 플레이타임
  4. 리뷰 작성 시점 플레이타임
  5. 리뷰 작성 시각
  6. 무료 제공 여부
- 분포 안정화: log1p 변환
- 정규화: train split만으로 fit한 StandardScaler 적용
- MLP: Dense(64, ReLU) -> Dense(300, tanh)
- 텍스트 임베딩과 동일한 300차원 메타 특징 벡터 meta_feature (mf)로 사영
- Dropout 적용

<hr>

<h2>Architecture Details (Controller + External Memory)</h2>

<h3>1) Controller 벡터 생성</h3>

- controller = rf와 mf의 element-wise multiply

<h3>2) External Memory Network</h3>

- 학습 가능한 외부 메모리 행렬: 100 슬롯 x 300차원
- Read: softmax 기반 read weight로 메모리 슬롯을 가중합 -> meta_memory 벡터
- Write: add, erase 벡터 생성 -> 메모리를 점진적으로 갱신
- 갱신식: new_memory = memory * (1 - ew * ee) + ew * ea
- 학습이 진행되는 동안 작성자-아이템 상호작용을 메모리에 누적

<hr>

<h2>Regression Head</h2>

- output_feat = meta_memory와 rf의 element-wise multiply
- Dense(1, linear)

<hr>

<h2>Target</h2>

target = log(votes_up + 1)

<hr>

<h2>Code</h2>

- 구현 파일: <code>09_DMMN_nomans.py</code>
- 실행 절차: grid search (5 lr x 3 dropout x 3 batch = 45조합) -> best 조합 10회 반복 (fixed42 5회 + varied 5회) -> best run을 test에 평가
