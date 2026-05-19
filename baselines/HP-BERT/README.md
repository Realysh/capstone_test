<h2>HP-BERT - Helpfulness Prediction BERT</h2>

사전학습 BERT의 [CLS] 표현을 frozen 상태로 활용해 리뷰 유용성을 회귀하는 단일 텍스트 모달 베이스라인

<hr>

<h2>Architecture</h2>
<p align="center">
  <img src="HP-BERT_picture.png" width="800" />
</p>
<p align="center">
  <i>Bilal, M., & Almazroi, A. A. (2023). Effectiveness of fine-tuned BERT model in classification of helpful and unhelpful online customer reviews. Electronic Commerce Research, 23, 2737-2757.</i>
</p>

<hr>

<h2>Overview</h2>

HP-BERT는 사전학습 언어모델 BERT를 활용하여 리뷰 유용성 분류 성능을 유의미하게 개선함을 Yelp 리뷰 데이터셋을 통해 입증한 모델이다. 본 구현에서는 Steam 리뷰 도메인에 적용하기 위해 BERT 인코더를 frozen 상태로 두고, 사전학습 의미 표현을 회귀 task에 그대로 활용한다.

<hr>

<h2>Input Modality</h2>

- 단일 텍스트 모달리티 (리뷰 텍스트)
- 토큰화: bert-base-uncased tokenizer
- 시퀀스 길이: 최대 256 토큰 (padding / truncation)

<hr>

<h2>Architecture Details</h2>

<h3>1) BERT 인코더 (frozen)</h3>

- 12층 Transformer Encoder, bert-base-uncased
- 본 프로젝트와 Yelp 데이터의 도메인 차이를 고려하여 Yelp fine-tuned 가중치를 사용하지 않음
- 가중치 frozen, <code>torch.no_grad()</code> 환경에서 순전파만 수행
- 추가 학습 파라미터 없이 대규모 사전학습 의미 표현만 보존적으로 활용

<h3>2) [CLS] 표현 추출</h3>

- self-attention 기반 양방향 문맥 정보를 압축한 [CLS] 토큰의 <code>pooler_output</code>
- 마지막 hidden state에 tanh 활성을 거친 표현
- 768차원 벡터로 한 리뷰의 통합 표현 확보

<hr>

<h2>Regression Head</h2>

- Dropout
- Dense(1, linear)

<hr>

<h2>Target</h2>

target = log(votes_up + 1)

<hr>

<h2>Code</h2>

- 구현 파일: <code>03_BERT_nomans.py</code>
- 실행 절차: BERT [CLS] 임베딩 사전 추출 -> grid search (5 lr x 3 dropout x 3 batch = 45조합) -> best 조합 10회 반복 (fixed42 5회 + varied 5회)
