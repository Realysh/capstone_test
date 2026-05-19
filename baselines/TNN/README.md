<h1>TNN - Text-based 1D Convolutional Neural Network</h1>

<hr>

<h2>🏗️ Architecture</h2>
<p align="center">
  <img src="TNN_picture.png" width="800" />
</p>
<p align="center">
  <i>Olmedilla, M., Martinez-Torres, M. R., & Toral, S. (2022). Prediction and modelling online reviews helpfulness using 1D Convolutional Neural Networks. Expert Systems with Applications, 198, 116787.</i>
</p>

<hr>

<h2>Overview</h2>

- TNN은 전통적 1D CNN, 즉 다양한 kernel size의 병렬
합성곱이 텍스트 모달리티의 의미 패턴을 포착할 수 있음을 입증한 모델임. 
- 본 구현에서는 Steam 리뷰 데이터에 동일 구조를 적용해 리뷰 유용성(votes_up)을 회귀한다.

<hr>

<h2>Using tool</h2>

* <code>Python 3.10-3.12</code>
* <code>tensorflow==2.18.1</code>
* <code>gensim>=4.3.0</code>
* <code>nltk>=3.8.0</code>
* <code>scikit-learn>=1.3</code>
* <code>numpy&lt;2.0</code>
* <code>pandas>=2.0</code>
* <code>scipy</code>
* <code>tqdm</code>

<hr>

<h2>📝Architecture Details</h2>

<h3>1) Embedding Single Text Modality</h3>

단일 텍스트 모달리티(리뷰 텍스트)를 입력으로 사용한다. 먼저 NLTK tokenizer로 리뷰 텍스트를 토큰화한 후, 학습 데이터(train split)로만 fit한 100차원 Word2Vec으로 초기화된 Embedding layer를 통과시켜 임베딩 행렬을 구성한다.

<hr>

<h3>2) Parallel 1D Conv (3 branch)</h3>

동일한 임베딩 행렬에 세 개의 Conv1D를 병렬로 적용한다. 각 branch는 kernel size 1, 2, 3을 가지며, branch당 100개의 필터와 ReLU 활성화 함수로 구성된다. 이 세 branch는 각각 개별 단어, 인접한 두 단어, 세 단어 범위의 지역적 패턴을 동시에 추출하는 역할을 한다.

<h3>3) GlobalMaxPooling1D</h3>

각 branch의 출력에 GlobalMaxPooling1D를 적용하여 가장 큰 활성화 값만을 추출한다. 이를 통해 리뷰 전체에서 각 필터가 가장 강하게 반응한 위치의 신호만이 단일 벡터로 압축된다.

<h3>4) Concat</h3>

세 개의 100차원 pooled 벡터를 이어붙여 최종적으로 300차원의 통합 벡터를 구성한다.

<hr>

<h2>Regression Head</h2>

Concat된 300차원 벡터는 회귀 헤드를 거쳐 단일 스칼라 값으로 변환된다. 회귀 헤드는 Dropout -> Dense(32, ReLU) -> Dropout -> Dense(1, linear) 순서로 적용되며, 중간 Dense layer가 차원 축소 및 비선형성을 부여하고, 마지막 linear layer가 유용성 점수를 출력한다.

<hr>

<h2>Target</h2>

학습 타깃은 원시 votes_up 값을 안정적으로 회귀하기 위해 로그 변환을 적용한 <code>log(votes_up + 1)</code>이다. 또한 votes_up 분포의 long-tail 영향을 줄이기 위해 학습 데이터(train split)에 한해 99 퍼센타일을 초과하는 샘플을 제거하는 q99 cutoff를 적용한다. 이 cutoff는 train split에만 적용되며 validation/test에는 적용하지 않아 데이터 누수를 방지한다.

<hr>