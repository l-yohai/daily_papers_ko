# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using GPT-4o.

Thanks to [@AK391](https://github.com/AK391) for great work.

## Daily Papers (2024-10-21)

### [How Do Training Methods Influence the Utilization of Vision Models?](https://arxiv.org/abs/2410.14470)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.14470.png)

Vote: 4

Authors: ['Janis Keuper', 'Margret Keuper', 'Paul Gavrikov', 'Shashank Agnihotri']

- ***What's New***: 이 연구는 시각 모델(Vision Models)의 사용 방식이 교육 방법(Training Methods)에 어떻게 영향을 받는지를 조사하였습니다. 특히, 이미지넷-1k(ImageNet-1k) 분류 모델을 대상으로 아키텍처와 교육 데이터를 고정한 상태에서 다양한 교육 파이프라인을 적용하여 실험적 평가를 수행하였습니다.
- ***Technical Details***: 연구는 각 레이어의 가중치(Weights)를 임의의 값으로 대체한 후 모델의 결정 기능에 얼마나 영향을 미치는지를 평가하는 방법론을 사용하였습니다. 구체적으로, 로지츠(Logits)에 소프트맥스(Softmax)를 적용하여 나온 확률 벡터의 코사인 거리(Cosine Distance)를 측정하였고, 이 거리 변화가 각 레이어의 중요성(크리티컬리티; Criticality)을 나타냅니다.
- ***Performance Highlights***: 연구 결과에 따르면, 교육 방법에 따라 초기 레이어의 중요성이 증가하거나 감소하고, 자가 감독 학습(Self-Supervised Learning) 또는 적대적 학습(Adversarial Training) 등의 다른 방법에 따라 상충되는 효과가 나타났습니다. 특정 교육 방법의 결과로, 모델의 결정 기능이 초기에 집중되는 경향이 있는 것으로 관찰되었습니다.

### [FiTv2: Scalable and Improved Flexible Vision Transformer for Diffusion Model](https://arxiv.org/abs/2410.13925)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13925.png)

Vote: 13

Authors: ['Cai Zhou', 'Wanli Ouyang', 'Di Huang', 'ZiDong Wang', 'Zeyu Lu', 'and Lei Bai']

- ***What's New***: FiTv2는 이미지 해상도가 다양한 경우에도 효과적으로 학습할 수 있는 Flexible Vision Transformer(FiT)의 업그레이드 버전으로, 향상된 학습 전략과 새로운 아키텍처 디자인을 도입하여 Diffusion Model에서 사용 가능합니다. 이미지 크기와 가로세로비에 제한이 없도록 설계되었습니다.
- ***Technical Details***: FiTv2는 쿼리-키 벡터 정규화(Query-Key Vector Normalization), 적응형 계층 정규화 적응 모듈(AdaLN-LoRA), 교정된 흐름 스케줄러(rectified flow scheduler), 로그-노멀 샘플러(Logit-Normal sampler)를 포함한 여러 혁신적인 설계를 통합합니다. 또한, 효과적인 혼합 데이터 전처리 전략을 적용하여 다양한 해상도의 이미지 합성 효율성을 높였습니다.
- ***Performance Highlights***: FiTv2는 이전 모델들에 비해 2배의 수렴 속도를 보입니다. 다양한 해상도의 이미지 생성에서 뛰어난 성능을 보였으며, 320×320, 224×448, 160×480 해상도에서 FID 점수 기준으로 최상의 결과를 달성했습니다. 또한, high-resolution 이미지 생성 및 텍스트-이미지 생성 작업에서도 우수한 성능을 입증했습니다.

### [A Common Pitfall of Margin-based Language Model Alignment: Gradient Entanglement](https://arxiv.org/abs/2410.13828)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13828.png)

Vote: 3

Authors: ['Huazheng Wang', 'Yifan Zeng', 'Mengdi Wang', 'Liu Leqi', 'Hui Yuan', 'Yue Wu']

- ***What's New***: 이 논문은 인간 피드백을 통한 강화 학습(Reinforcement Learning from Human Feedback; RLHF) 방법에서 마진 기반 정렬(Margin-based Alignment) 방식의 언어 모델(Language Model; LM) 정렬에서 발생하는 일반적인 함정, 즉 'Gradient Entanglement'을 식별합니다. 이 문제는 선호되는(Preferred) 답변과 비선호되는(Dispreferred) 답변의 확률이 동기화되어 증가하거나 감소하는 문제를 초래합니다.
- ***Technical Details***: 마진 기반 목표의 본질적인 효과를 'Gradient Entanglement'으로 정의하고, 선호되는 로그 확률의 기울기(inner product)와 비선호되는 기울기 사이의 내적이 클 때 이 효과가 문제를 일으킨다는 조건을 도출했습니다. 다양한 선호 최적화 알고리즘의 학습 동역학을 설명하고 개선 가능한 방법을 제안합니다.
- ***Performance Highlights***: 이 논문은 마진 기반 목표가 LLM, 특히 DPO(Direct Preference Optimization)를 사용하는 경우 특정 상황에서 선호와 비선호 확률을 개별적으로 변경하는 데 어려움이 있음을 실험적으로 입증합니다. 선호 데이터를 동일한 길이로 정규화했을 때 성능이 개선되었음을 확인했습니다.

### [SHAKTI: A 2.5 Billion Parameter Small Language Model Optimized for Edge AI and Low-Resource Environments](https://arxiv.org/abs/2410.11331)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.11331.png)

Vote: 2

Authors: ['Kruthika KR', 'Rakshit Aralimatti', 'Syed Abdul Gaffar Shakhadri']

- ***What's New***: SHAKTI는 25억 개의 파라미터를 가진 소형 언어 모델(Small Language Model)로, 엣지 AI(Edge AI) 및 낮은 자원 환경에 최적화되어 있습니다. 이는 실시간 AI 애플리케이션을 위해 높은 성능의 NLP와 효율적이고 정밀한 최적화를 결합하여 성능과 효율성을 모두 제공하는 솔루션입니다.
- ***Technical Details***: SHAKTI는 Variable Grouped Query Attention(VGQA), Pre-Normalization, SwiGLU 활성화 함수, 그리고 Rotary Positional Embeddings(RoPE)을 통해 엣지 장치에서 효율적으로 작동합니다. VGQA는 주어진 키에 대해 여러 쿼리를 그룹화되어 메모리 사용을 줄이고 추론 시간을 가속화합니다. 또한, RoPE는 긴 텍스트 시퀀스를 적은 메모리 사용으로 처리할 수 있게 하여 긴 문서 요약이나 복잡한 쿼리에 적합합니다.
- ***Performance Highlights***: SHAKTI는 2.5억 파라미터를 가진 모델이면서도 MMLU에서 71.7%의 점수를 기록하여 더 큰 모델인 Phi-3 Mini-4k와 Gemma 7B보다 우수한 성능을 보여주었습니다. PIQA에서는 86.2%의 성과를 나타내며 Shakti-LLM은 자원 제약 환경에서 효율적인 성능을 제공합니다.

### [Web Agents with World Models: Learning and Leveraging Environment Dynamics in Web Navigation](https://arxiv.org/abs/2410.13232)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13232.png)

Vote: 31

Authors: ['Kai Tzu-iunn Ong', 'Dongha Lee', 'Jihoon Kim', 'Namyoung Kim', 'Sunghwan Kim', 'Gwanwoo Song', 'Hyungjoo Chae', 'Jinyoung Yeo', 'Minju Gwak']

- **What's New**: 이 연구는 LLM 기반의 웹 에이전트에 세계 모델 (World Model)을 처음으로 통합하여 현재 LLM의 환경 동적 이해 능력 부족 문제를 해결합니다. 시뮬레이션을 통해 에이전트의 행동 결과를 예측함으로써 결정 과정을 향상시키며, 트리 탐색 기반 에이전트보다 비용 및 시간 효율성을 강조합니다.
- **Technical Details**: 본 연구에서는 전환 중심 관찰 추상화 (Transition-focused Observation Abstraction)를 제안하여 세계 모델이 시뮬레이션할 다음 관찰을 자유형 자연어로 예측하도록 훈련합니다. 각 행동 후보의 결과를 예측하고 가치 함수 (Value Function)를 통해 보상을 추정하여 최적의 행동을 선택합니다.
- **Performance Highlights**: WebArena와 Mind2Web 벤치마크 실험에서 WMA 웹 에이전트는 정책 모델 훈련 없이도 성능을 개선하고, 최근 트리 탐색 에이전트에 비해 6.8배 높은 비용 효율성과 5.3배 빠른 시간 효율성을 입증했습니다. Mind2Web에서 이전 SOTA 성과를 능가했습니다.

### [Context is Key(NMF): Modelling Topical Information Dynamics in Chinese Diaspora Media](https://arxiv.org/abs/2410.12791)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.12791.png)

Vote: 3

Authors: ['Rebecca M. M. Hicke', 'Ross Deans Kristensen-McLachlan', 'Mette Thunø', 'Márton Kardos']

- ***What's New***: 이 논문은 중국 디아스포라 미디어의 정보 역학을 효율적으로 분석하기 위해 설계된 새로운 방법론인 KeyNMF를 소개합니다. 특히, 2024년 유럽 의회 선거 시기가 다가오는 기간 동안 이 미디어의 주제적 변화를 모델링하여 중국 정부가 디아스포라를 대상으로 하는 정보 제어를 어떻게 수행하는지를 조사합니다.
- ***Technical Details***: KeyNMF는 Transformer 기반의 문맥적 임베딩 모델을 활용하여 고정 및 동적 주제 모델링을 구현하는 새로운 접근 방식입니다. 이 방법론은 NMF(Non-negative Matrix Factorization)의 신뢰성, 안정성, 확장성 및 해석가능성을 바탕으로 문서에서 키워드 중요도를 계산하고 이를 비음수 행렬 분해로 분해하여 주제를 추출합니다.
- ***Performance Highlights***: KeyNMF는 여러 중국어 데이터셋과 메트릭에서 높은 성능을 보였습니다. 특히 외부 일관성 측면에서 우수한 결과를 기록했으며, 기존의 고전적인 주제 모델에 비해 성능이 크게 개선되었습니다.

### [Teaching Models to Balance Resisting and Accepting Persuasion](https://arxiv.org/abs/2410.14596)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.14596.png)

Vote: 1

Authors: ['Mohit Bansal', 'Elias Stengel-Eskin', 'Peter Hase']

- ***What's New***: 이 연구에서는 LLMs(대형 언어 모델)가 설득에 취약하며, 이것이 모델이 적대적인 대화 상대와 마주할 때 위험을 초래할 수 있음을 지적합니다. 새로운 훈련 방법인 PBT(Persuasion-Balanced Training)를 소개하여 부정적 설득에 저항하면서도 긍정적 설득을 수용할 수 있도록 훈련 방법을 제안하였습니다.
- ***Technical Details***: PBT는 다중 에이전트 재귀적 대화 트리를 활용하여 데이터 생성하고, 선호 최적화를 통해 설득을 적절히 수용하도록 모델을 훈련시킵니다. 이 과정에서 QA(Question-Answering) 설정을 통해 두 LLM이 논쟁하며 대화 트리를 생성합니다. 각 턴의 다양한 답변을 비교해 긍정적 및 부정적 설득 데이터를 도출합니다.
- ***Performance Highlights***: PBT로 훈련된 모델은 Misinformation에 대한 저항성이 향상되고, 'Are you sure?' 프롬프트에도 높은 정확성을 유지하여 Flipflopping을 방지합니다. 이는 Mistral-7B, Llama-3.1-8B, Llama-3.1-70B 모델을 대상으로 평균 정확도 63.88%를 기록하며, 원래 모델의 48.87%보다 성능이 월등함을 보여줍니다. 팀 환경에서도 설득할 수 있는 모델은 협업 속에서 성능을 전반적으로 향상시킬 수 있음을 확인했습니다.

### [Montessori-Instruct: Generate Influential Training Data Tailored for Student Learning](https://arxiv.org/abs/2410.14208)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.14208.png)

Vote: 1

Authors: ['Chenyan Xiong', 'Xiaochuan Li', 'Zichun Yu']

- ***What's New***: Montessori-Instruct는 학생(학생)의 학습 선호도에 맞춘 고유한 데이터 합성을 수행하는 혁신적인 데이터 합성 프레임워크입니다. 이 접근 방식은 Synthetic Data의 효과를 향상시키고, 학생의 학습 향상에 기여하는 더욱 영향력 있는 합성 데이터를 생성합니다.
- ***Technical Details***: Montessori-Instruct는 영향 함수(influence functions)를 활용해 학생 모델의 데이터 선호도를 정확히 측정하고 이를 기반으로 교사 모델(teacher model)을 최적화합니다. Direct Preference Optimization (DPO)을 통해 학생의 학습에 직접적으로 연결된 데이터를 생성하도록 교사를 최적화합니다. 이 과정은 영향력 있는 데이터를 식별하고 교사 모델을 점진적으로 최적화하여 반복됩니다.
- ***Performance Highlights***: Montessori-Instruct는 Alpaca Eval과 MT-Bench에서 각각 18.35%와 46.24%의 성능 향상을 달성하며, 기존 합성 방법보다 우수한 성능을 보입니다. 다양한 NLP 작업에서도 일반화 능력이 입증되었습니다. 이는 학생의 데이터 선호도를 반영한 교사의 최적화가 학습 성능을 향상시킴을 나타냅니다.

### [Mini-Omni2: Towards Open-source GPT-4o with Vision, Speech and Duplex Capabilities](https://arxiv.org/abs/2410.11190)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.11190.png)

Vote: 13

Authors: ['Changqiao Wu', 'Zhifei Xie']

- ***What's New***: Mini-Omni2는 GPT-4o와 유사한 기능성을 가진 최초의 오픈 소스 다중 모드 언어 모델(multi-modal language model)로, 시각, 청각, 텍스트 및 청각 중단 메커니즘을 제공합니다. 이 모델은 실시간 스트리밍 음성 출력을 지원하며, 명령 기반의 사용자와의 상호작용을 통해 더욱 유연한 사용자 상호작용을 가능케 합니다.
- ***Technical Details***: Mini-Omni2는 CLIP의 시각 컴포넌트와 Whisper의 음성 인코더를 활용하여 시각 및 청각 모달리티에 대한 고효율 데이터 사용을 최적화합니다. 모델은 세 단계의 훈련 과정을 통해 모달리티 확장을 위한 효율적인 훈련 방식을 채택하며, 명령 기반의 중단 메커니즘을 통해 스트리밍 토큰을 사용하여 모델이 자신의 오디오 출력 스트림을 외부 의미에 따라 제어할 수 있도록 하였습니다.
- ***Performance Highlights***: Mini-Omni2의 음성 인식 정확도는 librispeech-other 데이터셋에서 Whisper 모델을 능가하며, 이는 모델의 견고한 성능을 보여줍니다. 시각 모달리티 추가 후 Mini-Omni 대비 약간의 정확도 저하는 발생하지만, 이는 데이터 비중 감소에 기인할 수 있습니다.

### [SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs](https://arxiv.org/abs/2410.13276)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13276.png)

Vote: 15

Authors: ['Zhichen Zeng', 'Ting Cao', 'Mao Yang', 'Dayou Du', 'Yizhao Gao', 'Shijie Cao', 'Hayden Kwok-Hay So', 'Fan Yang']

- ***What's New***: SeerAttention은 대형 언어 모델(LLMs)에서 내재된 스파스 주의(attention sparsity)를 학습하여 효율성을 개선하는 새로운 주의 메커니즘입니다. 기존의 스파스 주의가 미리 정의된 패턴이나 휴리스틱에 의존했던 것과 달리, SeerAttention은 학습 가능한 게이트를 도입하여 동적으로 중요한 블록을 선택할 수 있습니다. 이는 정확도와 속도 향상을 효과적으로 균형 잡습니다.
- ***Technical Details***: SeerAttention은 블록-스파스 주의를 사용하여 계산과 메모리 오버헤드를 줄입니다. 이를 위해 맞춤형 FlashAttention 구현을 개발하여 블록 레벨의 주의 맵을 최소 오버헤드로 추출합니다. 학습 가능한 'Attention Gate'는 Q와 K 입력 매트릭스를 풀링하여 중요한 블록을 식별하며, 이러한 블록 정보는 이후 주의 계산에 사용됩니다. 훈련 과정에서 AttnGate는 표준 주의에서 생성된 주의 맵을 자신만의 방식으로 학습합니다.
- ***Performance Highlights***: SeerAttention은 post-training 단계에서 최신 방법을 능가하며, 긴 문맥의 미세 조정에서도 뛰어난 성능을 보입니다. 특히 YaRN과 결합하여 긴 문맥의 모델에서 스파시티 비율 90%를 달성하면서도 최소한의 손실을 보입니다. FlashAttention-2에 비해 최대 5.67 배의 속도 향상을 이루었습니다.

### [Are AI Detectors Good Enough? A Survey on Quality of Datasets With Machine-Generated Texts](https://arxiv.org/abs/2410.14677)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.14677.png)

Vote: 7

Authors: ['Anastasia Voznyuk', 'Yury Chekhovich', 'Andrey Grabovoy', 'German Gritsai']

- ***What's New***: 이 연구는 기계 생성 텍스트(AI-generated Texts)의 탐지 품질을 평가하기 위해 사용되는 데이터셋의 질에 대해 체계적인 리뷰를 제공하고 있습니다. 특히 데이터셋이 낮은 평가 품질을 제공하는지와 같은 문제를 해결하고 AI 탐지기를 더 잘 지원하기 위한 고품질 데이터셋 개발 방법을 제안하고 있습니다.
- ***Technical Details***: 연구진은 AI 탐지기(AI Detectors)의 신뢰성을 개선하기 위해 여러 데이터셋을 분석하고 경쟁 대회 및 연구 논문에서 사용된 데이터셋을 체계화했습니다. 탐지의 질을 평가하는 방법으로 PHD(Topological Time Series), KL-Divergence, Attention Maps 등을 사용하여 데이터셋의 전반적인 품질을 측정했습니다. 또한, 문장 순서를 임의로 바꾸거나 단어를 대체하는 등 텍스트의 소규모 수정이 AI 모델에 미치는 영향을 평가하여 데이터셋의 로버스트성을 검토했습니다.
- ***Performance Highlights***: 복잡한 데이터셋에서 모델들의 탐지 성능은 데이터셋의 질에 크게 영향을 받는 것으로 나타났습니다. RuATD와 같은 일부 데이터셋에서는 AI 탐지기의 성능이 0.82의 정확도에 그쳤으며, 반면 다른 데이터셋에서는 탐지기가 거의 완벽에 가까운 성능을 보였습니다. 이는 데이터셋의 품질이 탐지기의 성능에 큰 영향을 미친다는 점을 시사하며, 보다 정교한 탐지기에 대한 필요성을 강조합니다.

### [BiGR: Harnessing Binary Latent Codes for Image Generation and Improved Visual Representation Capabilities](https://arxiv.org/abs/2410.14672)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.14672.png)

Vote: 2

Authors: ['Shaozhe Hao', 'Kwan-Yee K. Wong', 'Xuantong Liu', 'Bojia Zi', 'Shihao Zhao', 'Xianbiao Qi', 'Rong Xiao', 'Kai Han']

- ***What's New***: BiGR는 바이너리 잠재 코드를 활용한 새로운 조건부 이미지 생성 모델로, 생성과 표현 능력을 동시에 향상시킵니다. 이 모델은 생성과 분별 작업을 동일한 프레임워크 내에서 통합하는 최초의 모델입니다.
- ***Technical Details***: BiGR는 바이너리 토크나이저, 마스킹 모델링 메커니즘, 그리고 바이너리 트랜스코더를 사용하여 바이너리 코드 예측을 수행합니다. 새로운 샘플링 방법인 엔트로피 정렬 샘플링을 도입해 효율적인 이미지 생성을 가능합니다. 전체 프레임워크는 언어 모델 아키텍처로 구성되며, 이미지 토큰의 바이너리 코드를 생성하고 예측하는 과정을 통해 작동합니다.
- ***Performance Highlights***: BiGR는 FID-50k 및 리니어-프로브 정확도에서 기존의 모델에 비해 더 우수한 성능을 보이며, 다양한 시각 작업에서 제로-샷 범용성을 보여줍니다. 특히, 고해상도 이미지 생성, 인페인팅, 아웃페인팅, 편집, 보간 및 추가 기능에서 뛰어난 성능을 기록했습니다. 실험 결과는 BiGR의 생성 및 분별 작업에서의 뛰어난 성능을 보여주며, 이 모델의 확장 가능성을 제시합니다.

### [DPLM-2: A Multimodal Diffusion Protein Language Model](https://arxiv.org/abs/2410.13782)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13782.png)

Vote: 6

Authors: ['Zaixiang Zheng', 'Shujian Huang', 'Fei Ye', 'Quanquan Gu', 'Dongyu Xue', 'Xinyou Wang']

- ***What's New***: DPLM-2는 단백질 구조와 서열의 동시 생성을 목표로 하는 멀티모달 단백질 기초 모델로, 기존의 단일 모달 모델의 한계를 넘어 구조와 서열을 함께 이해하고 생성할 수 있습니다. DPLM-2는 기존의 DPLM을 확장하여 실험적 구조와 고품질 합성 구조를 학습하여 구조 및 서열의 결합 분포를 학습합니다.
- ***Technical Details***: DPLM-2는 루프프리 양자화(lookup-free quantization) 기반의 토크나이저를 사용하여 3D 좌표를 이산 토큰으로 변환하여 구조 학습을 가능하게 합니다. 효율적인 웜업 전략을 통해 대규모 진화 데이터와 사전 학습된 서열 기반 모델의 구조적 귀납 편향을 활용합니다. 서열과 구조를 공동으로 처리하여 고도의 상관성을 가진 단백질 구조와 서열을 동시에 생성합니다.
- ***Performance Highlights***: DPLM-2는 다양한 조건부 생성 작업에서 경쟁력 있는 성능을 보여줍니다. 무조건적인 단백질 생성에서 높은 디자인 가능성과 접히는 능력을 가지며, 특정 문제들에서는 기타 강력한 모델과 견줄 수 있는 성능을 나타내며, 대장형 및 모티프 유지 생성 과제에서도 뛰어난 성공률을 보입니다.

### [UCFE: A User-Centric Financial Expertise Benchmark for Large Language Models](https://arxiv.org/abs/2410.14059)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.14059.png)

Vote: 40

Authors: ['Honghai Yu', 'Yuzhe Yang', 'Yilin Guo', 'Jimin Huang', 'Benyou Wang', 'Qianqian Xie', 'Mingcong Lei', 'Yueru He', 'Yifei Zhang', 'Haining Wang', 'Xiao Zhang', 'Yan Hu', 'Ruoli Gan']

- ***What's New***: UCFE 벤치마크(UCFE Benchmark)는 대형 언어 모델(Large Language Models; LLMs)의 금융 분야에서 복잡한 실제 작업처리 능력을 평가하기 위한 새로운 프레임워크입니다. 이 벤치마크는 인간 전문가의 평가와 동적이고 작업에 따른 상호작용을 결합하여 금융 시나리오의 복잡성을 시뮬레이션합니다.
- ***Technical Details***: UCFE 벤치마크는 사용자 중심의 설계(User-Centric Design)를 채택하여 분석가, 금융 전문가, 규제 전문가, 일반 대중 등 4가지 사용자 그룹대표 역할을 시뮬레이션하도록 LLMs를 평가합니다. 이 벤치마크는 17가지 사용자 프로필에 맞춘 작업 유형을 개발하여, 제로샷(Zero-shot) 및 퓨샷(Few-shot) 설정에서 다중 회환 대화를 포함하는 330개의 데이터 포인트를 제공합니다.
- ***Performance Highlights***: UCFE 벤치마크에서 LLMs의 성능은 사용자 선호도와 높은 상관성을 보였으며, 피어슨 상관 계수 0.78을 기록했습니다. 특히, 금융 텍스트에 대해 훈련된 LLMs는 전문적인 금융 개념 이해 및 사용자 의도 해석에서 유의한 향상을 보였습니다. 중형 모델(7B~14B 파라미터)은 컴퓨팅 효율성과 도메인 전문성 간의 균형을 유지하면서 특히 우수한 성과를 보였습니다.

### [Looking Inward: Language Models Can Learn About Themselves by Introspection](https://arxiv.org/abs/2410.13787)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13787.png)

Vote: 3

Authors: ['Tomek Korbak', 'Miles Turpin', 'John Hughes', 'Henry Sleight', 'Owain Evans', 'Ethan Perez', 'Felix J Binder', 'James Chua', 'Robert Long']

- ***What's New***: 이 논문에서는 '내성(Interospection)'을 통해 대형 언어 모델(LLMs)이 자신에 대한 정보를 획득할 수 있는 가능성을 조사합니다. 이는 모델이 훈련 데이터로부터 직접 파생되지 않은 내부 상태에서 기인한 지식을 습득할 수 있음을 의미하며, 이는 모델 해석 가능성을 향상시킬 수 있습니다.
- ***Technical Details***: 연구진은 LLM을 미세 조정하여 가설 시나리오에서 자신의 행동의 속성을 예측하도록 학습시켰습니다. 예를 들어 '주어진 입력 P에 대해 당신의 출력은 단기 혹은 장기 옵션에 더 유리합니까?'와 같은 질문을 통해 내성을 시험하였습니다. 실험에서는 GPT-4, GPT-4o, Llama-3 모델이 사용되었으며, 각 모델이 자체 행동을 예측하는 성능을 비교하였습니다.
- ***Performance Highlights***: 실험 결과, '자기예측(self-prediction)'이 '교차예측(cross-prediction)'보다 우수하였으며, 이는 모델이 내성을 통해 자기 행동의 경향에 대한 특권적 접근을 하고 있음을 시사합니다. 예를 들어, Llama-3-70B 모델이 GPT-4o 모델보다 17% 높은 정확성을 보였습니다.

### [NaturalBench: Evaluating Vision-Language Models on Natural Adversarial Samples](https://arxiv.org/abs/2410.14669)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.14669.png)

Vote: 23

Authors: ['Ranjay Krishna', 'Wenxuan Peng', 'Jean de Dieu Nyandwi', 'Zixian Ma', 'Deva Ramanan', 'Zhiqiu Lin', 'Graham Neubig', 'Daniel Jiang', 'Baiqi Li', 'Simran Khanuja']

- ***What's New***: NaturalBench는 시각-언어 모델(Vision-Language Models, VLMs)을 평가하는 새로운 척도로, 인간이 쉽게 해결할 수 있는 자연적인 이미지와 질문에 VLM이 어려움을 겪는지 확인하기 위한 자료를 제공합니다. 이 벤치마크는 두 개의 이미지와 질문을 쌍으로 구성하여 '눈먼' 솔루션이 시각적 입력 없이 답을 제시하지 못하도록 구성되어 있어 더 도전적입니다.
- ***Technical Details***: NaturalBench는 자연 이미지-텍스트 코퍼스에서 기존 모델(CLIP, ChatGPT 등)을 활용하여 자동 생성된 10,000개의 인간 검증 VQA 샘플로 구성됩니다. 각 질문에는 서로 다른 답을 생성하는 두 개의 이미지가 연관되어 있으며, 모델이 정밀한 평가를 받을 수 있도록 27개의 세부 기술 태그를 포함합니다. 또한, 비언어적 소스를 포함하여 다양한 데이터 출처를 이용해 지속적으로 업데이트할 수 있는 동적 평가 시스템을 제공합니다.
- ***Performance Highlights***: 53개의 최신 VLM을 평가한 결과, 모델의 성능은 인간의 50-70% 수준에 그쳐 여전히 높은 도전 과제가 남아 있음을 확인했습니다. 예를 들어, 모델들은 논리적 추론 및 속성 바인딩과 같은 복합적 기술에 어려움을 겪으며, 자연 반대 샘플에서 심각한 편향을 드러냈습니다. 디바이어싱을 하면 성능이 크게 개선됨을 보여주어 향후 연구에 중요한 인사이트를 제공합니다.

### [HART: Efficient Visual Generation with Hybrid Autoregressive Transformer](https://arxiv.org/abs/2410.10812)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.10812.png)

Vote: 5

Authors: ['Enze Xie', 'Haotian Tang', 'Zhuoyang Zhang', 'Han Cai', 'Song Han', 'Junsong Chen', 'Yecheng Wu', 'Junyu Chen', 'Shang Yang', 'Yao Lu']

- ***What's New***: HART(Hybrid Autoregressive Transformer)는 1024x1024 이미지 생성을 직접적으로 수행하면서, 디퓨전 모델과 비교 가능한 품질을 제공하는 효율적인 저작권 모형입니다. 이는 하이브리드 토큰화(hybrid tokenization) 및 잔여 확산(residual diffusion)을 통해, 이미지의 전체 구조를 포착하는 이산 토큰과 세부 사항을 다듬는 잔여 토큰을 생성함으로써 달성되었습니다.
- ***Technical Details***: HART의 하이브리드 토크나이저는 연속적인 특성을 디코딩할 수 있도록 하여 전통적인 이산 토크나이저의 한계를 극복합니다. 이는 VAR에서 확장된 확장 가능한 해상도의 자회귀 변환기를 사용하여 이산 토큰을 모델링하고, 37M 파라미터만으로 잔여 환산을 하는 경량 잔여 확산 모듈을 통해 잔여 토큰을 학습합니다. 이 과정에서 VAR의 절대 위치임베딩을 상대 임베딩으로 전환하여 높은 해상도에서 훈련 비용을 효과적으로 줄입니다.
- ***Performance Highlights***: HART는 1024x1024 해상도에서 디퓨전 모델에 비해 4.5-7.7배 높은 처리량과 3.1-5.9배 낮은 지연 시간을 달성했습니다. 또한, MJHQ-30K와 ImageNet 데이터셋에서의 평가에서도 이를 뒷받침하는 품질을 자랑하며, MACs에서 10.7배 적은 계산량으로 MAR을 능가하는 성능을 보여주었습니다.

### [DAWN: Dynamic Frame Avatar with Non-autoregressive Diffusion Framework for Talking Head Video Generation](https://arxiv.org/abs/2410.13726)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13726.png)

Vote: 7

Authors: ['Chenyu Liu', 'Jia Pan', 'Jun Du', 'Hanbo Cheng', 'Limin Lin', 'Jiefeng Ma', 'Pengcheng Xia', 'Pengfei Hu']

- ***What's New***: DAWN은 비자동회귀(diffusion) 프레임워크를 통해 동적 프레임 아바타를 생성하여 포트레이트와 음성이 주어졌을 때 생동감 있는 토킹 헤드 비디오를 제작하는 획기적인 접근 방법을 소개합니다. 이는 일반적인 토킹 헤드 생성에 맞춰 설계된 최초의 비자동회귀 솔루션으로, 더 빠른 추론 속도와 높은 품질을 보장합니다. 또한, DAWN은 비자동회귀 접근 방식의 가능성을 제공하며, 방대한 실험 결과로부터 그 효과와 잠재적인 영향을 입증합니다.
- ***Technical Details***: DAWN은 세 가지 주요 컴포넌트로 구성되어 있습니다: (1) 잠재 흐름 생성기(Latent Flow Generator; LFG) (2) 조건부 오디오-비디오 흐름 확산 모델(Audio-to-Video Flow Diffusion Model; A2V-FDM), 그리고 (3) 포즈 및 깜박임 생성 네트워크(Pose and Blink generation Network; PBNet)입니다. LFG는 비디오의 잠재 모션 표현을 모델링하며, A2V-FDM은 오디오에서 모션 표현을 생성하고, PBNet은 오디오에서 자연스러운 포즈와 깜박임 시퀀스를 생성합니다. 이러한 구조는 오디오와 포즈/깜박임 신호를 기반으로 생동감 있는 토킹 헤드 비디오를 생성하도록 설계되었습니다.
- ***Performance Highlights***: 우리 방법은 실험에서 FID, FVD32, FVD16, CSIM, BAS, Blink/s 지표에서 가장 우수한 성능을 보였으며, 이는 시각적 품질, 리듬 있는 포즈, 자연스러운 깜박임을 포함한 다양한 평가에서 기존 최첨단 방법을 능가합니다. 또한 비자동회귀 전략을 사용하여 더 빠른 생성 속도를 달성하였으며, 잠재적인 오류 축적 문제를 해결하여 긴 비디오에서도 안정적으로 높은 품질의 결과를 유지했습니다.

## License

This project is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.ko) license.

## Citation

If this work is helpful, please kindly cite as:

```bibtex
@misc{daily_papers,
  title = {Huggingface Daily Papers},
  author = {AK391},
  howpublished = {\url{https://huggingface.co/papers}},
  year = {2023}
}

@misc{daily_papers_ko,
  title = {Automatically translate and summarize huggingface's daily papers into korean},
  author = {l-yohai},
  howpublished = {\url{https://github.com/l-yohai/daily_papers_ko}},
  year = {2023}
}
```

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=l-yohai/daily_papers_ko&type=Date)
