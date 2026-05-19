<h2>TNN - Text-based 1D Convolutional Neural Network</h2>

전통적 1D CNN의 병렬 합성곱으로 리뷰 텍스트의 의미 패턴을 포착하는 단일 텍스트 모달 베이스라인

<hr>

<h2>Architecture</h2>
<p align="center">
  <img src="TNN_picture.png" width="800" />
</p>
<p align="center">
  <i>Olmedilla, M., Martinez-Torres, M. R., & Toral, S. (2022). Prediction and modelling online reviews helpfulness using 1D Convolutional Neural Networks. Expert Systems with Applications, 198, 116787.</i>
</p>

<hr>

<h2>Overview</h2>

- TNN은 전통적 1D CNN, 즉 다양한 kernel size의 병렬
합성곱이 텍스트 모달리티의 의미 패턴을 포착할 수 있음을 입증한 모델이다. 본 구현에서는 Steam 리뷰 데이터에 동일 구조를 적용해 리뷰 유용성(votes_up)을 회귀한다.

<hr>

<h2>Using tool</h2>

- Python 3.10-3.12, RTX 4090 / CUDA 12.x (RunPod)
- <code>tensorflow==2.18.1</code> - Keras layers, EarlyStopping, Adam
- <code>gensim>=4.3.0</code> - Word2Vec (100d, train split만 fit)
- <code>nltk>=3.8.0</code> - word_tokenize
- <code>scikit-learn>=1.3</code> - split, MAE/MSE/MAPE 메트릭
- <code>numpy&lt;2.0</code>, <code>pandas>=2.0</code>, <code>scipy</code>, <code>tqdm</code>
- 호환성: TF 2.18과 torch 2.5.1+cu121의 cuDNN 충돌 회피 위해 <code>nvidia-cudnn-cu12>=9.3</code> override (자세히는 루트 <code>pyproject.toml</code>)

<hr>

<h2>Architecture Details</h2>

<h3>1) Embedding Single Text Modality </h3>

- 단일 텍스트 모달리티 (리뷰 텍스트)
- 전처리: NLTK tokenizer로 토큰화
- 임베딩: 학습 데이터로만 fit한 Word2Vec (100차원)으로 초기화한 Embedding layer

<hr>

<h3>2) Parallel 1D Conv (3 branch)</h3>

- 동일 임베딩에 세 개의 Conv1D를 병렬 적용
- kernel size = 1, 2, 3
- 각 branch당 100개 필터, ReLU 활성화 함수 적용
- 각각 개별 단어 / 인접 두 단어 / 세 단어의 지역적 패턴 추출

<h3>3) GlobalMaxPooling1D</h3>

- 각 branch에서 가장 큰 활성화 값을 추출
- 리뷰 전체에서 각 필터가 가장 강하게 반응한 위치의 신호만 단일 벡터로 압축

<h3>4) Concat</h3>

- 세 개의 100차원 pooled 벡터를 이어붙여 300차원 통합 벡터 구성

<hr>

<h2>Regression Head</h2>

- Dropout
- Dense(32, ReLU)
- Dropout
- Dense(1, linear)

<hr>

<h2>Target</h2>

target = log(votes_up + 1)

<hr>