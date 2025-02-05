# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using GPT-4o.

Thanks to [@AK391](https://github.com/AK391) for great work.

## Daily Papers (2025-02-05)

### [OmniHuman-1: Rethinking the Scaling-Up of One-Stage Conditioned Human Animation Models](https://arxiv.org/abs/2502.01061)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01061.png)

Vote: 134

Authors: Jiaqi Yang, Jianwen Jiang, Chao Liang, Zerong Zheng, Gaojie Lin

- ***What's New***: OmniHuman-1는 새로운 트레이닝 전략 'omni-conditions training'을 통해 다양한 모션 관련 조건을 통합하여 스케일링 문제를 해결한 새로운 혼합 조건 인간 비디오 생성 모델입니다. 이 모델은 기존 오디오 및 포즈 기반 모델의 한계를 넘어 보편적인 휴먼 애니메이션을 가능케 합니다.
- ***Technical Details***: OmniHuman-1은 DiT(Diffusion Transformer) 구조를 활용하여 텍스트, 오디오, 포즈 등 3개의 모션-관련 조건을 신호로 사용합니다. omni-conditions training 전략에 따르면 강한 조건은 약한 조건을 활용하여 트레이닝을 강화할 수 있으며, 강한 조건의 비중이 너무 높을 경우 다른 조건들은 학습이 잘 이루어지지 않기 때문에 각 조건에 따라 비율을 조정하여 트레이닝합니다.
- ***Performance Highlights***: OmniHuman-1은 초상화 및 반신 애니메이션 등 다양한 비율과 스타일의 입력에서 선도적인 특화 모델과 비교해 뛰어난 성능을 보였습니다. 여러 데이터셋에서 평가된 결과, OmniHuman은 FID와 FVD 기준으로 가장 낮은 점수를 기록하며, 특히 Sync-C 지표(7.443)에서도 최상의 성능을 보여줍니다.

### [The Differences Between Direct Alignment Algorithms are a Blur](https://arxiv.org/abs/2502.01237)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01237.png)

Vote: 100

Authors: Viacheslav Sinii, Alexey Gorbatovski, Alexey Malakhov, Daniil Gavrilov, Boris Shaposhnikov

- ***What's New***: Direct Alignment Algorithms (DAAs)는 인간 피드백을 통한 강화 학습(RLHF)에서 보상 모델링과 강화 학습 단계를 대체하여 정책 최적화만을 사용하여 대형 언어 모델(LLM)의 정렬을 간소화합니다. 본 논문에서는 단일 단계 방법에 명시적 감독 미세 조정(SFT)을 추가하여 성능을 개선하였으며, ORPO와 ASFT에 β 파라미터를 도입하여 최적화를 통일하는 것이 두 단계 방법과 유사한 성능을 제공한다는 것을 보여주었습니다.
- ***Technical Details***: 본 연구에서는 DAA 방법을 점수의 사용 기법(예: 확률 비율)과 SFT 단계의 필요 여부에 따라 분류하였습니다. 또한, β가 0에 근접하면 DAA의 손실 함수들의 기울기가 두 가지 카테고리 내에서 방향이 일치한다는 것을 이론적으로 증명하였습니다. 그리고 ASFT와 ORPO를 단일 단계에서 다단계로 확장할 수 있으며 β 스케일링 옵션이 가능하다는 것을 제안했습니다. 실험적으로, SFT 단계의 포함이 성능을 상당히 개선하며 5~10%의 데이터만으로도 거의 최적의 결과를 얻을 수 있음을 발견했습니다.
- ***Performance Highlights***: SFT 초기화된 ORPO는 +9.3 LC / +6.9 AH, ASFT는 +1.9 LC / +3.1 AH의 성능 향상을 보였으며, Pairwise 방법이 Pointwise 방법보다 특히 대규모 모델에서 더 나은 성능을 나타냈습니다. 또한, β를 튜닝함으로써 선호 최적화의 강도를 조절하여 성능을 향상시킬 수 있음을 보여주었습니다.

### [Process Reinforcement through Implicit Rewards](https://arxiv.org/abs/2502.01456)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01456.png)

Vote: 50

Authors: Weize Chen, Maosong Sun, Kaiyan Zhang, Ning Ding, Yuan Yao, Jiarui Yuan, Bingxiang He, Zefan Wang, Lifan Yuan, Ganqu Cui, Bowen Zhou, Yuchen Fan, Xu Han, Zhiyuan Liu, Xingtai Lv, Yu Cheng, Tianyu Yu, Huayu Chen, Shuo Wang, Hao Peng, Hanbin Wang, Wendi Li, Qixin Xu

- ***What's New***: 이 논문에서는 기존의 결과 중심 보상(Outcome Rewards) 대신 암묵적 보상(Implicit Rewards)을 통해 강화 학습을 할 수 있는 새로운 기법인 PRIME(Process Reinforcement through Implicit Rewards)을 제안했습니다. PRIME은 복잡한 수학적 문제와 코딩 문제에서 대형 언어 모델(LLMs)의 추론 능력을 대폭 개선할 수 있음을 보여줍니다.
- ***Technical Details***: PRIME은 정책 롤아웃과 결과 레이블만을 사용하여 온라인 PRM(Process Reward Models)을 업데이트하는 프레임워크로, 별도의 보상 모델 학습 과정을 생략합니다. PRIME은 다양한 RL 알고리즘과 결합할 수 있으며, 토큰 수준의 밀집 프로세스 보상과 결과 수준의 희소 보상을 함께 사용할 수 있습니다. 특히 Qwen2.5-Math-7B-Base 모델로 시작하여 성능을 개선하였습니다.
- ***Performance Highlights***: PRIME을 통해 Eurus-2-7B-PRIME 모델은 SFT 모델 대비 15.1%의 평균 성능 향상을 이루었으며, 수학 추론 벤치마크에서 Qwen2.5-Math-7B-Instruct보다 우수한 성과를 보였습니다. 특히 AMC 및 AIME 경쟁에서 20% 이상의 성능 향상을 달성했습니다.

### [Preference Leakage: A Contamination Problem in LLM-as-a-judge](https://arxiv.org/abs/2502.01534)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01534.png)

Vote: 30

Authors: Yue Huang, Bohan Jiang, Wei Wang, Jiawei Han, Huan Liu, Dawei Li, Renliang Sun, Xiangliang Zhang, Ming Zhong

- ***What's New***: 이 연구는 LLM-as-a-judge의 새로운 문제인 선호 누출(Preference Leakage)를 탐지 및 분석하여, 데이터 생성과 평가에 관련된 LLM 간의 유사성이 시스템적인 평가 편향을 야기할 수 있음을 밝혔습니다.
- ***Technical Details***: 이 연구에서는 데이터 생성기 LLM과 판사 LLM 사이의 세 가지 유형의 관련성을 정의합니다: 동일 모델, 상속 관계 및 동일 모델 패밀리 내 소속입니다. 또한 다양한 LLM과 벤치마크를 통해 선호 누출로 인한 평가 편향을 실험적으로 확인하고, 이에 대한 선호 누출 점수를 도입하여 평가합니다.
- ***Performance Highlights***: 실험 결과, Arena-Hard와 AlpacaEval 2.0에서 선호 누출 점수는 대부분의 모델 쌍에서 긍정적인 값을 보여주었으며, 이는 데이터 생성기와 평가자가 동일한 경우에 널리 퍼진 편향을 보여줍니다. 특히 더 큰 학생 모델일수록 판사 LLM에서 더 큰 편향을 유발할 가능성이 높았습니다. 이러한 결과는 선호 누출의 검출이 어렵고, 특히 주관적 질문 및 판단 차원에 더욱 영향을 미친다는 것을 보여줍니다.

### [AlignVLM: Bridging Vision and Language Latent Spaces for Multimodal Understanding](https://arxiv.org/abs/2502.01341)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01341.png)

Vote: 30

Authors: Xiangru Jian, Pierre-André Noël, Akshay Kalkunte Suresh, Sathwik Tejaswi Madhusudhan, Marco Pedersoli, Bang Liu, Enamul Hoque, David Vazquez, Ahmed Masry, Yoshua Bengio, Spandana Gella, Perouz Taslakian, Suyuchen Wang, Abhay Puri, Christopher Pal, Nicolas Chapados, Issam H. Laradji, Tianyu Zhang, Sai Rajeswar, Juan A. Rodriguez, Chao Wang, Aarash Feizi

- ***What's New***: ALIGNVLM은 새로운 비전-텍스트 정렬 방법으로, 기존의 비전 인코더가 생성한 시각적 피처를 LLM의 텍스트 임베딩과 정렬하는 데 효과적입니다. 이는 특히 문서 이해 작업에서, 스캔된 문서 이미지를 텍스트 콘텐츠와 정확하게 맵핑하는 데 유리합니다.
- ***Technical Details***: ALIGNVLM은 시각적 피처를 LLM의 사전 학습된 텍스트 어휘 임베딩에 확률 분포로 맵핑함으로써 LLM의 텍스트 공간 내에서 시각적 피처를 효과적으로 해석할 수 있도록 합니다. 이는 주어진 시각적 문맥을 LLM 임베딩 공간의 볼록 껍질에 포함되도록 하여 노이즈나 분포 외 입력의 리스크를 줄입니다.
- ***Performance Highlights***: ALIGNVLM 모델은 다양한 문서 이해 벤치마크에서 기존 최첨단 방법보다 우수한 성능을 입증하였으며, DocVQA, InfoVQA, DeepForm 등에서 높은 점수를 기록합니다. ALIGNVLM-Llama-3.2-3B 모델은 비슷한 크기의 모델들 중 최고 성능을 제공하며, 특히 Instruct VLMs와 비교하여 매우 경쟁력 있는 성과를 보여줍니다. ALIGN 모듈의 설계는 다양한 벤치마크에서 표현 간 정렬과 멀티모달 문서 이해 성능을 향상시키는 데 기여했습니다.

### [SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model](https://arxiv.org/abs/2501.18636)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2501.18636.png)

Vote: 25

Authors: Sensen Zhang, Shichao Song, Jason Zhaoxin Fan, Simin Niu, Zhiyu Li, Jiawei Yang, Bo Tang, Feiyu Xiong, Mengwei Wang, Xun Liang, Hanyu Wang

- ***What's New***: SafeRAG는 Retrieval-Augmented Generation(RAG)의 보안 취약성을 평가하기 위해 설계된 첫 번째 중국어 벤치마크입니다. 네 가지 새로운 공격 기법 - silver noise, inter-context conflict, soft ad, white DoS -을 도입하여 RAG 시스템이 현실적인 공격 시나리오에 얼마나 취약한지를 검증합니다.
- ***Technical Details***: SafeRAG 데이터셋은 네 가지 공격 작업(잡음, 충돌, 독성, 서비스 거부)에 대해 주로 인간 및 차세대 언어 모델(LLM)의 도움을 통해 수동으로 구성되어 있으며, 각 작업에 대해 RAG 파이프라인의 다른 단계에서 공격이 수행됩니다. 클라우드 뉴스 웹사이트에서 수집된 원시 데이터를 바탕으로 포괄적인 질문-컨텍스트를 생성하여 공격용 텍스트를 구성합니다.
- ***Performance Highlights***: 실험 결과에 따르면 RAG는 모든 공격 작업에 대해 상당한 취약성을 보였습니다. Baichuan 13B 모델은 여러 공격 작업에서 우수한 평가 메트릭을 기록했으며, 특히 white DoS 및 DoS 변종 메트릭에서 두각을 나타냈습니다. 다른 모델들과 비교하여 Baichuan 13B가 상대적으로 안전한 모델로 평가되었습니다.

### [VideoJAM: Joint Appearance-Motion Representations for Enhanced Motion Generation in Video Models](https://arxiv.org/abs/2502.02492)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.02492.png)

Vote: 25

Authors: Uriel Singer, Adam Polyak, Yaniv Taigman, Amit Zohar, Lior Wolf, Shelly Sheynin, Hila Chefer, Yuval Kirstain

- ***What's New***: VideoJAM은 기존 비디오 생성 모델에 강력한 운동 사전 (Motion Prior)을 부여하여 비디오 생성 시 운동 일관성을 크게 향상시키는 새로운 프레임워크입니다. 이 시스템은 운동 일관성을 다양한 동작 유형에 걸쳐 크게 증진시키며, 기존의 소유 모델들을 능가하는 성능을 자랑합니다.
- ***Technical Details***: VideoJAM은 학습 과정에서 외관과 운동을 동시 예측하는 목표로 확장함으로써, 모델이 공동 외관-운동 표현 (Joint Appearance-Motion Representation)을 학습하게 합니다. 추론 시에는 Inner-Guidance라는 메커니즘을 도입하여, 모델 자체의 변화하는 운동 예측을 동적 가이던스 신호로 활용합니다. 두 개의 선형 레이어를 아키텍처에 추가함으로써 기존 층을 크게 변경하지 않고도 이 기능을 통합할 수 있습니다.
- ***Performance Highlights***: VideoJAM은 운동 일관성 면에서 매우 경쟁적인 소유 모델을 능가하는 상태-of-the-art 성능을 달성했으며, 외관의 시각적 품질 또한 향상시켰습니다. 다양한 모델 크기와 모션 유형에 걸쳐 사전 훈련된 비디오 모델을 적용한 결과, Motion Coherence에 있어 큰 향상을 보여 주었습니다.

### [MatAnyone: Stable Video Matting with Consistent Memory Propagation](https://arxiv.org/abs/2501.14677)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2501.14677.png)

Vote: 25

Authors: Peiqing Yang, Chen Change Loy, Shangchen Zhou, Jixin Zhao, Qingyi Tao

- ***What's New***: MatAnyone는 고유의 메모리 기반 패러다임을 활용해 목표 객체 매트(matting)를 수행하는 새로운 프레임워크입니다. 이 방법은 지역 적응 메모리 융합 메커니즘(region-adaptive memory fusion)을 도입하여 비디오의 주요 영역에서 의미적 안정성을 보장하는 동시에, 객체 경계의 세밀한 디테일을 보존합니다.
- ***Technical Details***: MatAnyone은 메모리 기반 기법을 사용하여 이전 프레임의 정보를 통합하는 일정한 메모리 전파 모듈(consistent memory propagation module)을 제안합니다. 또한, 다양한 스타일을 포괄하는 새로운 고품질의 VM800 데이터셋과 세그멘테이션 데이터를 사용한 새로운 훈련 전략을 도입하여 강력한 매팅 성능을 제공합니다.
- ***Performance Highlights***: MatAnyone은 기존의 비디오 매트팅 방법을 능가하며, 복잡한 배경에서도 온전한 주요 영역과 경계 디테일을 유지하여 우수한 성능을 입증합니다. 특히다양한 해상도와 복잡한 실제 환경에서도 안정적이고 정확한 결과를 구현합니다.

### [Inverse Bridge Matching Distillation](https://arxiv.org/abs/2502.01362)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01362.png)

Vote: 22

Authors: Daniil Selikhanovych, Nikita Gushchin, Alexander Korotin, Dmitry Baranchuk, David Li, Evgeny Burnaev

- ***What's New***: 역방향 브리지 매칭 증류(Inverse Bridge Matching Distillation; IBMD)는 확산 브리지 모델(DBMs)의 추론 속도를 크게 개선하기 위한 새로운 증류 기법을 제안합니다. 이 기술은 기존 DBM 증류 방법과는 달리 조건부 및 비조건부 DBM 모두에 적용 가능하며, 오염된 이미지만을 사용하여 단일 단계 생성기로 모델을 증류할 수 있습니다.
- ***Technical Details***: IBMD는 주어진 교사 모델의 드리프트와 같은 드리프트를 갖는 혼합 브리지(Πθ)를 찾는 역 문제로 정의됩니다. 이 최적화 문제는 실용적인 양상으로 재구성되어, 모델의 자동 미분을 통해 최적화가 가능해졌습니다. 특히, 학습된 브리지 매칭 모델을 단일 단계 생성기로 증류하는 과정을 통해 맥락적 추론을 개선합니다.
- ***Performance Highlights***: 실험 결과, IBMD는 DBMs의 추론 속도를 최대 100배까지 향상시킬 수 있으며, 경우에 따라 이전의 교사 모델보다 더 나은 생성 품질을 제공합니다. JPEG 복원 및 초해상도, 스케치에서 이미지 변환과 같은 다양한 이미지-이미지 변환 작업에서의 성능이 검증되었습니다. 예를 들어, 4x 초해상도 작업에서 1000 스텝의 I2SB 모델보다 1 스텝의 IBMD가 더 나은 FID를 기록합니다.

### [SliderSpace: Decomposing the Visual Capabilities of Diffusion Models](https://arxiv.org/abs/2502.01639)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01639.png)

Vote: 19

Authors: Nick Kolkin, Zongze Wu, David Bau, Eli Shechtman, Rohit Gandikota, Richard Zhang

- ***What's New***: 이 논문은 SliderSpace라는 새로운 프레임워크를 소개하여 확산 모델(Diffusion Models)의 시각적 능력을 자동으로 분해하고 이를 제어 가능한 방향으로 소개합니다. 기존의 방식과 달리, SliderSpace는 단일 텍스트 프롬프트로부터 여러 해석 가능한 방향을 동시에 발견할 수 있습니다. 각 방향은 저차순 어댑터(Low-Rank Adaptor)로 학습되어 모델의 잠재 공간 내에서 놀라운 가능성을 발견하게 합니다.
- ***Technical Details***: SliderSpace는 텍스트 프롬프트만을 사용하여 모델의 지식 내의 의미 있는, 다양한, 제어 가능한 방향을 발견합니다. 각 방향은 저차원 어댑터로 구현되며 사용자들이 모델의 시각적 다양성을 탐색하고 결합할 수 있도록 합니다. 의미적 정규성(Semantic Orthogonality)과 분포 일관성(Distribution Consistency)을 위해 슬라이더(joined sliders)에 의한 변형은 임베딩에서 정렬됩니다. 이를 통해 모델의 지식 분포를 해석 가능한 방향으로 효과적으로 분해할 수 있습니다.
- ***Performance Highlights***: SliderSpace는 다양한 개념 분해, 예술 스타일 탐험, 다양성 향상에서 효과적임을 실험적으로 증명했습니다. 테스트 실험에서 SliderSpace는 기존의 방법에 비해 더 다양한 시각적 변형을 만들어내며, 사용자가 모델의 조형 가능성을 보다 직관적이고 해석할 수 있도록 방향성을 제시합니다. 또한 실험 결과 사용자 연구는 SliderSpace가 기초 모델에 비해 더욱 다양한 변형을 생성함을 확인했습니다.

### [Self-supervised Quantized Representation for Seamlessly Integrating Knowledge Graphs with Large Language Models](https://arxiv.org/abs/2501.18119)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2501.18119.png)

Vote: 19

Authors: Mengling Feng, Tianzhe Zhao, Kai He, Ling Huang, Zhen Peng, Fangzhi Xu, Qika Lin, Jingying Ma

- ***What's New***: 이 논문에서는 지식 그래프(Knowledge Graph; KG)를 대형 언어 모델(Large Language Models; LLMs)과 원활하게 통합하기 위한 셀프 슈퍼바이즈드 양자화 표현(Self-supervised Quantized Representation; SSQR) 방법을 제안합니다. 이는 KG의 구조적 및 의미론적 정보를 불연속 코드(discrete codes)로 압축하여 시맨틱 텍스트의 형식과 일치시킴으로써 달성됩니다.
- ***Technical Details***: SSQR 방법은 그래프 컨볼루셔널 네트워크(Graph Convolutional Network; GCN)를 인코더로 사용하여 KG의 이웃 구조를 모델링하고, 벡터 양자화(vector quantization)를 통해 KG의 양자화 표현 학습을 수행합니다. 학습된 코드들은 LLM의 입력 요소로 사용되어 KG 작업을 지원하는 데이터로 통합됩니다.
- ***Performance Highlights***: SSQR은 기존의 비지도 양자화(Unsupervised Quantized) 방법보다 뛰어난 성능을 보여주며, LLaMA2 및 LLaMA3.1 모델들은 KG 링크 예측 및 삼중 분류 작업에서 우수한 성능을 발휘합니다. 이는 기존 방법들이 수천 개의 토큰이 필요했던 반면, SSQR은 엔티티당 16개의 코드만 사용하여 효율적인 성능을 제공합니다.

### [MM-IQ: Benchmarking Human-Like Abstraction and Reasoning in Multimodal Models](https://arxiv.org/abs/2502.00698)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.00698.png)

Vote: 18

Authors: Winston Hu, Huanqia Cai, Yijun Yang

- ***What's New***: MM-IQ는 인간의 추상화 및 추론 능력을 다중모달 모델(Multimodal Models)에서 평가하기 위한 최초의 포괄적인 평가 프레임워크입니다. 총 2,710개의 테스트 문제와 8개의 독특한 추론 패러다임으로 구성되어, 현재 모델들의 한계점을 밝혀내어 미래 연구의 방향을 제시합니다.
- ***Technical Details***: MM-IQ는 8가지 구체적인 추론 패러다임에 걸쳐 문제를 수집하여, 모델의 추상적 시각 추론(Abstract Visual Reasoning; AVR) 능력을 포괄적으로 평가합니다. 이 벤치마크는 국가 공무원 시험에서 문제를 수집하고, 각 문제의 패턴 기억이 없도록 문제 구성을 다양화하여 현실적인 평가를 목표로 합니다. 각 문제의 응답은 정규 표현식(regex)을 통해 정답을 추출하며, 모든 모델에 동일한 질문 프롬프트를 제시하여 평가합니다.
- ***Performance Highlights***: MM-IQ에서 현재 최고 수준의 다중모달 모델들조차 인간 성능의 절반도 미치지 못하며, 가장 잘 수행한 모델이 27.49%의 정확도에 그쳤습니다. 이는 랜덤 초이스와 거의 차이가 없는 결과로, LMM들이 근본적인 인간의 추론 능력을 근접하게 모방하는데 여전히 큰 한계가 있음을 보여줍니다. 더 큰 모델 크기가 성능 향상에 기여하지만, 여전히 객체 구체 추론과 같은 일부 영역에서 제한적인 성능을 보이며, 단순한 규칙에 의존하는 경우가 많습니다.

### [Scalable-Softmax Is Superior for Attention](https://arxiv.org/abs/2501.19399)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2501.19399.png)

Vote: 17

Authors: Ken M. Nakanishi

- ***What's New***: Scalable-Softmax(SSMax)는 Transformer 기반 언어 모델에서 Softmax를 대체하는 새로운 방법으로, 입력 벡터 크기가 변동하는 상황에서도 주의(attention) 희미화 문제를 완화하여 더 긴 문맥에 대한 일반화를 향상시키는 데 초점을 맞추고 있습니다. SSMax는 입력 벡터 크기가 커져도 주의 분포가 평탄해지지 않도록 설계되었습니다.
- ***Technical Details***: SSMax는 Softmax의 문제를 피하기 위해 도입된 기술로, 입력 벡터 크기 n에 따라 지수 기반을 설정하여 주의 집중을 유지합니다. SSMax는 기존 아키텍처에 쉽게 통합될 수 있으며, 작은 폭의 코드를 수정하여 모든 Transformer 기반 모델에 적용할 수 있습니다. SSMax는 지수 함수 기반을 입력 벡터 크기와 결합함으로써 이러한 주의 희미화 문제를 해결합니다.
- ***Performance Highlights***: 실험 결과, SSMax를 적용한 모델은 사전 학습 동안 더 낮은 손실 값을 보였고, 문맥 크기가 훈련 순서 길이를 크게 초과할 때에도 더 낮은 테스트 손실을 유지했습니다. 또한, 주요 정보 검색 작업에서도 향상된 정확성을 보였으며, 훈련 중에 본 적 없는 길이의 문맥에서도 효과적으로 관련 정보를 추출할 수 있음을 입증했습니다. 사전 학습 초기에 SSMax를 도입한 모델은 특히 긴 문맥에서 뛰어난 일반화 능력을 보여주었습니다.

### [ACECODER: Acing Coder RL via Automated Test-Case Synthesis](https://arxiv.org/abs/2502.01718)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01718.png)

Vote: 15

Authors: Huaye Zeng, Haozhe Wang, Dongfu Jiang, Wenhu Chen, Xiaotong Chen, Ping Nie

- ***What's New***: ACECODER는 자동화된 대규모 테스트 케이스 생성(Automated Large-Scale Test-Case Synthesis)을 활용하여 코드 모델(Code Models)의 학습을 강화하는 새로운 접근 방식을 제안합니다. 이는 기존의 강화 학습(RL) 기법이 코딩 분야에서 잘 활용되지 못했던 문제를 해결하여 코드 생성 모델의 성능 향상을 목적으로 합니다.
- ***Technical Details***: ACECODER는 ACECODE-89K라는 대규모 데이터셋을 구축하여 신뢰할 수 있는 테스트 케이스와 질문 쌍을 생성합니다. 질문에 대한 테스트 케이스를 통과율(pass rate)에 따라 선호도 쌍을 구성하였으며, 이를 사용하여 보상 모델(reward model)을 Bradley-Terry 손실을 통해 학습했습니다. 이 보상 모델을 활용하여 강화 학습을 수행하며, 특히 Proximal Policy Optimization(PPO)과 Reinforcement++ 알고리즘을 사용하여 학습 효율성을 높였습니다.
- ***Performance Highlights***: ACECODER는 다양한 코딩 벤치마크(HumanEval, MBPP, BigCodeBench, LiveCodeBench)에서 일관된 성능 개선을 보여줍니다. 특히, Qwen2.5-Coder-7B-base 모델에 대한 강화 학습은 보다 높은 성능 향상을 이뤄냈으며, HumanEval-plus에서 25%, MBPP-plus에서 6% 이상의 성능 향상을 달성했습니다. 이로 인해 ACECODER는 대규모 코드 생성 모델에 RL 트레이닝의 잠재력을 열 수 있음을 시사합니다.

### [PixelWorld: Towards Perceiving Everything as Pixels](https://arxiv.org/abs/2501.19339)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2501.19339.png)

Vote: 14

Authors: Xueguang Ma, Wenhu Chen, Zhiheng Lyu

- ***What's New***: PixelWorld는 모든 데이터를 픽셀(pixel)로 처리하여 기존의 텍스트 또는 토큰 기반 입력 방식을 대체할 수 있는 새로운 평가 스위트를 소개합니다. 이 방법은 다중모달(multimodal) 데이터셋에서 토큰 기반 입력보다 더 나은 성능을 발휘하며, 인간의 지각 방식과 유사한 일관성을 가지도록 설계되었습니다.
- ***Technical Details***: PixelWorld 평가 스위트는 텍스트, 구조, 다중모달 태스크를 픽셀 기반 입력으로 전환하여 모델 성능을 분석합니다. 이미지를 통해 입력을 통일함으로써 OCR 오류를 줄이고 문맥 레이아웃을 보존하는 방식으로, 다양한 비디오/이미지 입력을 지원합니다. 중요한 기발한 기법으로는 PEAP-Fast 알고리즘이 도입되어, 빈 픽셀 영역을 제거하여 처리 속도를 크게 개선합니다.
- ***Performance Highlights***: PixelWorld에서 적은 크기의 모델(예: Phi-3.5-비전)은 픽셀 기반 입력에서 성능 저하를 보이나, 큰 모델(예: GPT-4o)은 텍스트와 픽셀 간의 성능 전이가 우수합니다. 다중모달 이미지 입력을 활용할 때 성능 향상(특히 거대 모델에서)을 보였지만, 복잡한 추론 태스크에서는 여전히 도전 과제가 존재합니다.

### [MakeAnything: Harnessing Diffusion Transformers for Multi-Domain Procedural Sequence Generation](https://arxiv.org/abs/2502.01572)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01572.png)

Vote: 14

Authors: Yiren Song, Cheng Liu, Mike Zheng Shou

- ***What's New***: MakeAnything는 텍스트 설명이나 조건화된 이미지를 기반으로 회화, 공예 및 요리와 같은 다양한 작업에 대한 절차적 튜토리얼을 현실적으로 생성하는 도구입니다. 이 연구는 확산 변환기(Diffusion Transformers; DiT)를 활용하여 멀티 도메인 절차적 시퀀스를 생성하는 최초의 DIT 기반 아키텍처를 도입했습니다.
- ***Technical Details***: MakeAnything는 21개의 카테고리에서 24,000개 이상의 절차적 시퀀스를 포함한 멀티 도메인 데이터셋을 사용하며, 비대칭 로우랭크 적응(Asymmetric LoRA)과 ReCraft 모델을 통해 텍스트 및 이미지 기반 생성 패러다임을 지원합니다. 이 기술은 대규모 데이터에서 사전 훈련된 인코더와 작업별 최적화된 디코더를 결합하여 일반화 기능과 도메인 특정 성능을 균형 있게 유지합니다.
- ***Performance Highlights***: MakeAnything는 광범위한 실험을 통해 기존의 모든 방법을 능가하며 절차적 생성 작업에 대한 새로운 성능 기준을 설정했습니다. 이 모델은 이미지 조건 시퀀스 생성에서 높은 일치성과 논리적 일관성을 보여주었으며, 반도메인에서의 일반화 능력도 확인되었습니다.

### [AIN: The Arabic INclusive Large Multimodal Model](https://arxiv.org/abs/2502.00094)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.00094.png)

Vote: 14

Authors: Ahmed Heakl, Omkar Thawkar, Salman Khan, Sara Ghaboura, Rao Muhammad Anwer, Fahad Shahbaz Khan, Hisham Cholakkal

- ***What's New***: AIN은 아랍어와 영어에 능통한 최초의 아랍어 포함 대형 멀티모달 모델(Large Multimodal Model; LMM)로, 다양한 분야에서 뛰어난 성능을 발휘하도록 설계되었습니다. 이 모델은 아랍어와 영어의 다중모달 데이터를 통해 아랍어 성능을 향상시키고, 다양한 응용 분야에서의 멀티모달 생성 AI 툴을 제공하는 중요한 단계로 자리매김하고 있습니다.
- ***Technical Details***: AIN 모델은 7억 개의 매개변수를 갖는 Qwen-2-VL-7B 아키텍처를 기반으로 개발되었으며, 3.6백만 개의 고품질 아랍어-영어 다중모달 데이터 샘플을 활용하여 훈련되었습니다. AIN은 다중언어 및 정교한 이미지-텍스트 정렬 작업에서 우수한 성능을 보이며, CAMEL-Bench 벤치마크에서 뛰어난 성능을 입증하였습니다.
- ***Performance Highlights***: AIN-7B 모델은 GPT-4o보다 8개 도메인과 38개 하위 도메인에서 3.4% 이상의 성능 향상을 보였으며, 특히 OCR과 문서 이해 분야에서 강력한 능력을 발휘했습니다. 이러한 성능은 다양한 벤치마크에서 두드러지며, 아랍어와 영어 모두에서 경쟁력 있는 성과를 거두었습니다.

### [FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation](https://arxiv.org/abs/2502.01068)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01068.png)

Vote: 14

Authors: Jae-Joon Kim, Jiwon Song, Dongwon Jo, Yulhwa Kim

- ***What's New***: FastKV는 긴 컨텍스트 시퀀스를 효과적으로 처리하기 위한 KV 캐시 압축 방법론을 도입하였습니다. 이는 LLMs의 초기 레이어에서는 전체 컨텍스트 정보를 유지하면서 심층 레이어에서는 선택된 정보만 전파하는 Token-Selective Propagation (TSP) 접근 방식을 채택하여 레이턴시를 개선합니다. 또한, FastKV는 GQA(Groued-Query Attention)와 호환되는 KV 캐시 압축을 포함하여 프로세싱 효율성을 높였습니다.
- ***Technical Details***: FastKV는 LLM의 초기 레이어에서 전체 컨텍스트 정보의 전파를 유지하면서 TSP를 사용하여 심층 레이어에서 선택된 토큰만 전파합니다. TSP 레이어는 주어진 레이어의 주의 맵을 활용하여 중요한 토큰을 식별합니다. 초기 레이어의 기존 KV 캐시 압축 기법은 주의 점수를 사용해 각 토큰의 중요성을 평가하며, GQA 대응 압축 기법을 적용해 주의-그룹 단위로 KV 데이터를 제거합니다.
- ***Performance Highlights***: FastKV는 최신 KV 캐시 압축 방식인 HeadKV와 비교하여 첫 번째 토큰까지의 시간(TTFT)을 2.00배, 처리량을 1.40배 개선하였습니다. 또한, 장기 컨텍스트 벤치마크에서 기존 방식과 유사한 수준의 정확도를 유지하였습니다. 이러한 결과는 FastKV가 실시간 애플리케이션의 KV 캐시 관리와 저 레이턴시 요구 사항을 효과적으로 충족할 수 있음을 시사합니다.

### [Scaling Embedding Layers in Language Models](https://arxiv.org/abs/2502.01637)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01637.png)

Vote: 14

Authors: Chiyuan Zhang, Yangsibo Huang, Edith Cohen, Ravi Kumar, Badih Ghazi, Pritish Kamath, Daogao Liu, Da Yu

- ***What's New***: SCONE라는 새롭고 확장 가능한 임베딩(Embedding) 레이어 확장 방법을 제안하며, 이는 언어 모델의 성능을 향상시키기 위한 방법론이다. SCONE은 입력 토큰에 대해 빈번하게 발생하는 n-그램의 임베딩을 추가적으로 도입함으로써, 모델의 크기가 커질 때에도 고정된 추론 시간 FLOPS를 유지하면서 인퍼런스 속도에 적은 영향을 주는 방식으로 작동한다. 새로운 스케일링 전략을 통해 캐시된 n-그램 임베딩의 수를 늘리거나, 이를 학습하는 n-그램 모델의 크기를 확장할 수 있다.
- ***Technical Details***: SCONE 메소드는 기존의 일반적인 입력 임베딩에 빈번하게 발생하는 n-그램을 위한 임베딩을 추가하는 방식으로 작동한다. 고정된 사전 정의된 n-그램 집합으로부터 컨텍스트화된 변형을 사용하여 확장된 입력 임베딩 테이블을 구축하며, 출력 레이어의 계산 비용에 영향을 미치지 않도록 설계되었다. 이 컨텍스트화된 토큰의 임베딩은 'f-gram 모델'이라 불리는 별도의 임베딩 변환 모델에서 생성된다. 이 구조는 전체 n-그램 업데이트의 희소성 문제를 해결하고, 추론 시 오프로드되어 메모리에 저장된다.
- ***Performance Highlights***: SCONE을 적용한 1B 파라미터 모델은 두 배의 추론 시간 FLOPS를 필요로 하는 기존 1.9B 파라미터 베이스라인 모델보다도 더 좋은 성능을 보여주었다. 왜냐하면, 빈번한 n-그램 임베딩과 f-gram 모델을 통해 작업을 수행함에 있어 메모리와 저장 공간을 효율적으로 사용하면서도, 속도의 병목 현상을 일으키지 않고 성능을 높일 수 있기 때문이다.

### [DeepRAG: Thinking to Retrieval Step by Step for Large Language Models](https://arxiv.org/abs/2502.01142)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01142.png)

Vote: 13

Authors: Jiali Zeng, Yaojie Lu, Chunlei Xin, Fandong Meng, Xianpei Han, Jie Zhou, Xinyan Guan, Hongyu Lin, Le Sun

- ***What's New***: DeepRAG는 다양한 대형 언어 모델(LLMs)의 사실적인 환상을 줄이기 위해 설계된 마크로프 결정 프로세스(Markov Decision Process; MDP) 기반의 검색 강화 추론(RAG) 프레임워크입니다. 이 방법은 질의(Query)를 단계별로 분해하여 각 단계에서 외부 지식을 검색할지, 아니면 매개변수 추론에 의존할지를 동적으로 결정합니다.
- ***Technical Details***: DeepRAG는 이진 트리 검색(BTS; Binary Tree Search) 방법을 사용하여 하위 질의를 생성하며, 모형이 자신의 지식 경계를 인식하도록 돕기 위해 자동화된 데이터 합성을 통해 모형을 미세 조정합니다. 모형은 주제 추론을 통해 끝난 단계별 절차를 모방하여 검색을 최적화합니다. 또한 이 과정에서 '검색 서사(Retrieval narrative)' 및 '원자 결정(atomic decisions)'을 사용해 질의에 대한 적응적 검색 흐름을 적용합니다.
- ***Performance Highlights***: DeepRAG는 다섯 개의 오픈 도메인 QA 데이터셋에서 가장 높은 정확도를 기록하며, 검색 효율성 또한 향상되었습니다. HotpotQA와 같은 데이터에서 기존 방법들보다도 21.99% 높은 정답 정확도를 달성했고, 검색 필요성과 매개변수 지식 간의 상관 관계가 높은 것을 나타냈습니다.

### [QLASS: Boosting Language Agent Inference via Q-Guided Stepwise Search](https://arxiv.org/abs/2502.02584)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.02584.png)

Vote: 11

Authors: Da Yin, Ziniu Hu, Yizhou Sun, Xingcheng Yao, Yao Tang, Kai-Wei Chang, Zongyu Lin

- ***What's New***: 이 연구는 QLASS라는 새로운 접근법을 제안하며, Q-값에 기반한 단계별 가이드라인을 통해 오픈 소스 언어 에이전트(language agents)의 추론 성능을 향상시킵니다. 이는 복잡한 장기 작업에서 한계점이 있는 결과 기반 보상 모델을 초월하여 단계별 피드백을 제공합니다.
- ***Technical Details***: QLASS는 탐색 트리(exploration tree)를 구축하고, 벨만 방정식(Bellman Equation)을 활용하여 벨만 최적화 규칙을 기반으로 모든 상태-액션 쌍의 Q-값을 추출하여 중간 주석을 생성합니다. 이 추출된 Q-값을 사용하여 QNet이라는 함수 근사(functio approximator)를 학습하고, 이는 추론 시 에이전트의 액션 선택을 Q 값에 의해 가이드합니다.
- ***Performance Highlights***: QLASS는 WebShop, ALFWorld, SciWorld 등의 다양한 에이전트 환경에서 강력한 성능을 발휘합니다. 제한된 주석 데이터 사용 시에도 강력한 성능을 유지하여 효율성과 강건함을 입증하였습니다. RFT와 ETO 같은 기존의 강력한 기법들보다 평균적으로 5% 이상의 성능 향상을 보여주었습니다.

### [ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning](https://arxiv.org/abs/2502.01100)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01100.png)

Vote: 10

Authors: Yejin Choi, Ronan Le Bras, Peter Clark, Ashish Sabharwal, Radha Poovendran, Bill Yuchen Lin, Kyle Richardson

- ***What's New***: ZebraLogic는 대형 언어 모델(LLMs)의 논리적 추론 능력과 확장성을 체계적으로 평가하기 위해 논리 퍼즐을 도입한 종합 평가 프레임워크입니다. 이 프레임워크는 모델의 성능을 논리 그리드 퍼즐을 통해 평가하여, 문제의 복잡성이 증가함에 따라 성능이 현저히 감소하는 '복잡성의 저주' 현상을 발견하였습니다.
- ***Technical Details***: ZebraLogic은 제약 만족 문제(CSPs)에서 유래한 논리 퍼즐로 구성되어 있습니다. 이 퍼즐은 주어진 단서에 따라 논리적 제약을 만족해야 하는 다양한 속성을 가진 집들로 구성됩니다. 퍼즐 생성 알고리즘은 초기 해답 격자를 설정하고, 가능한 모든 관계를 포착하는 단서를 생성한 후, 최소화 절차를 통해 중복되는 단서를 제거하여 고유한 해답을 찾습니다.
- ***Performance Highlights***: 주요 언어 모델 o1은 81%의 정확도를 달성하며, ZebraLogic 벤치마크에서 가장 높은 성능을 기록했습니다. 반면, 불리한 경우 성능이 빠르게 저하되며 '복잡성의 저주'에 따라 대다수의 모델이 높은 복잡성을 지닌 퍼즐을 해결하지 못하는 경향을 보였습니다. 특히, GPT-4o 모델은 대규모 샘플링을 통해 성능을 향상시킬 수 있음을 보여주었으나, 모델 크기만 증가시키는 것은 논리적 추론 과제를 해결하는 데 한계가 있음을 확인하였습니다.

### [Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search](https://arxiv.org/abs/2502.02508)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.02508.png)

Vote: 9

Authors: Zhenfang Chen, David Cox, Maohao Shen, Subhro Das, Wei Lu, Gregory Wornell, Zhenting Qi, Guangtao Zeng, Zhang-Wei Hong, Chuang Gan

- **What's New**: Satori는 강화 학습(Reinforcement Learning)을 활용하여 Chain-of-Action-Thought(COAT) 방법론을 도입한 새로운 접근법으로, 대형 언어 모델(LLM)의 추론 능력을 autoregressive 탐색을 통해 향상시키는 연구를 제안합니다. COAT는 문제 해결 과정에서 다양한 메타 행동(meta-action)을 가능하게 하며, LLM이 외부의 도움 없이 스스로 발전할 수 있도록 합니다.
- **Technical Details**: Satori는 두 단계의 학습을 통해 COAT를 구현합니다. 첫 번째는 소규모 형식 조정(format tuning) 단계로 COAT 추론 형식을 내재화합니다. 두 번째는 'Restart and Explore(RAE)' 기법을 활용한 대규모 자가 개선 단계로, 강화 학습을 통해 새로운 솔루션을 찾고 이전 오류를 교정합니다.
- **Performance Highlights**: Satori는 수학적 추론 벤치마크에서 최첨단 성능을 달성했으며, 학습되지 않은 영역에서도 강력한 일반화 능력을 보여줍니다. 특히, 수학 도메인 외에서도 강력한 성능을 발휘하여, 이는 대규모 강화 학습을 통한 일반적인 추론 능력의 향상을 나타냅니다.

### [The Jumping Reasoning Curve? Tracking the Evolution of Reasoning Performance in GPT-[n] and o-[n] Models on Multimodal Puzzles](https://arxiv.org/abs/2502.01081)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01081.png)

Vote: 9

Authors: Vernon Y. H. Toh, Soujanya Poria, Deepanway Ghosal, Yew Ken Chia

- ***What's New***: 이 논문은 OpenAI의 최신 LLM(GPT-[n]과 o-[n] 시리즈)의 멀티모달 퍼즐에 대한 추론 성능을 추적하는 혁신적인 연구를 제시합니다. 이는 특히 GPT-4o와 o1 모델의 뛰어난 성능 향상을 통해 이러한 모델이 복합적인 시각 및 언어 데이터를 처리하는 능력을 어떻게 향상시켰는지 조사합니다.
- ***Technical Details***: 이 연구는 PUZZLEVQA와 ALGOPUZZLEVQA 데이터셋을 사용하여 시각적 추론과 알고리즘적 문제 해결 능력을 평가합니다. 평가 과정에서 모델은 퍼즐을 다항 선택 및 개방형 질문 형식으로 풀어야 하며, 다양한 퍼즐 범주는 도형, 색상, 크기와 같은 요소를 포함합니다. 또한, 제로샷 체인 오브 쏘트(Chain of Thought; CoT) 프롬프트를 활용하여 GPT-[n] 모델의 추론 단계를 유도했습니다.
- ***Performance Highlights***: 최신 o1 모델은 추론 능력에서 비약적인 성과 향상을 보여주었으나, 이는 GPT-4o에 비해 750배 이상의 계산 비용을 초래했습니다. o1 모델은 색상 및 숫자 추론에서 높은 성과를 보였지만, 도형과 크기가 결합된 퍼즐에서는 제한된 성과를 보였습니다. 멀티모달 퍼즐에서의 전반적인 성능은 여전히 인간 성능에 미치지 못하며, 이는 시각적 이해 능력에 더 많은 개선이 필요함을 보여줍니다.

### [Improving Transformer World Models for Data-Efficient RL](https://arxiv.org/abs/2502.01591)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01591.png)

Vote: 8

Authors: Antoine Dedieu, J Swaroop Guntupalli, Xinghua Lou, Joseph Ortiz, Wolfgang Lehrach, Kevin Patrick Murphy, Carter Wendelken, Miguel Lazaro-Gredilla

- ***What's New***: 이 연구에서는 트랜스포머 월드 모델(Transformer World Models; TWM)을 기반으로 한 데이터 효율적인 모델 기반 강화 학습(Model-Based RL; MBRL) 기법을 개선하여 Craftax-클래식 벤치마크에서 새로운 성능 기록을 달성했습니다. 제안된 방법은 샘플 효율성을 높이기 위한 설계 선택을 통합하여, 인간 전문가가 달성한 보상을 초과하는 최초의 에이전트를 개발하였습니다.
- ***Technical Details***: 제안된 MBRL 알고리즘은 세 가지 주요 개선 사항을 포함합니다: 1) 실제와 상상된 데이터를 모두 사용하는 '웜업을 통한 다이나(Dyna with warmup)', 2) 이미지 패치에서 '근접 이웃 토크나이저(Nearest Neighbor Tokenizer; NNT)' 사용, 3) 같은 시점의 모든 토큰에 대해 병렬로 샘플링 가능한 '블록 교사 강제(Block Teacher Forcing; BTF)'. 이러한 방식으로 정책이 강화되며 데이터 효율성을 극대화합니다.
- ***Performance Highlights***: 개선된 방법은 Craftax-클래식 환경에서 1M 스텝 동안 이전 최첨단의 53.20% 보상을 67.42%로 높였으며, 인간 기준의 65.0% 보상을 초과했습니다. 제안된 방법은 다른 모든 모델 기반 및 모델 프리 강화 학습 방법을 능가하며, 최초로 인간 수준의 성능을 초과하는 결과를 보여줍니다.

### [Almost Surely Safe Alignment of Large Language Models at Inference-Time](https://arxiv.org/abs/2502.01208)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01208.png)

Vote: 8

Authors: Xiaotong Ji, Shyam Sundhar Ramesh, Ilija Bogunovic, Matthieu Zimmer, Haitham Bou Ammar, Jun Wang

- ***What's New***: 이 논문에서는 대형 언어 모델(LLM)의 추론 시간(inference-time)에 안전한 정렬(safe alignment)을 보장하는 새로운 접근 방식을 소개합니다. 전통적인 정렬 기술(예: RLHF)이 LLM을 재훈련하는데 비용이 많이 들고 과적합(overfitting)의 위험이 있는 반면, 이 새로운 방법은 추론 시 안전한 응답을 거의 확실히(generating safe responses almost surely) 생성하도록 함으로써 이를 해결합니다.
- ***Technical Details***: 이 논문은 LLM에서 안전한 응답 생성을 제약된 마르코프 결정 과정(constrained Markov decision process; cMDP)으로 정식화합니다. 안전 상태(safety state)를 증강하여 제한 없는 마르코프 결정 과정으로 변경된 새로운 MDP를 제시합니다. 이를 통해 LLM의 모델 가중치를 수정하지 않고도 안전 정렬이 가능합니다. 이 방법론을 바탕으로 'InferenceGuard'라는 실용적인 구현을 제안하며, 이는 LLM의 가중치를 유지한 채 안전한 정렬을 제공합니다.
- ***Performance Highlights***: InferenceGuard는 Alpaca-7B에서 98.02%, Beaver-7B-v3에서 100%의 높은 안전 비율을 기록하며 주어진 보상과의 균형을 잘 맞춥니다. 이는 기존 추론 시간 정렬 방법보다 안전하고 정렬된 응답을 생성하는 데에 있어 우수한 성능을 보여줍니다.

### [PhD Knowledge Not Required: A Reasoning Challenge for Large Language Models](https://arxiv.org/abs/2502.01584)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01584.png)

Vote: 7

Authors: Molly Q Feldman, Arjun Guha, Federico Cassano, Francesca Lucchetti, Carolyn Jane Anderson, Aleksander Boruch-Gruszecki, Zixuan Wu, Joydeep Biswas

- ***What's New***: 이 논문은 박사 수준의 지식을 요구하지 않는 새로운 벤치마크를 제시합니다. NPR Sunday Puzzle Challenge를 기반으로 한 이 벤치마크는 일반 지식으로 해결할 수 있는 문제들로 구성되어 있으며, 사람과 모델 모두에게 도전적인 과제를 제공합니다. OpenAI o1 모델이 다른 추론 모델보다 우수한 성능을 보였음을 확인하였습니다.
- ***Technical Details***: 이 벤치마크는 600개의 문제로 구성되어 있으며, NPR Sunday Puzzle Challenge의 오프 에어 주간 과제들에서 파생되었습니다. 이러한 문제들은 일반적인 지식으로도 문제를 이해할 수 있도록 설계되어 있으며, 머신으로 검증할 수 있는 문제들입니다. 다양한 최신 모델들이 이 벤치마크에서 평가되었습니다.
- ***Performance Highlights***: 벤치마크 결과, OpenAI o1 모델이 59%의 정확도로 다른 모델들보다 상당히 우수한 성능을 보였습니다. DeepSeek R1 모델은 일부 문제를 해결하기 전에 '포기'하는 경향을 보여주었고, '추론 길이'에 따른 정확도의 변화를 계량화하여 특정 길이 넘어서는 추가적인 추론이 정확도를 크게 향상시키지 않음을 확인하였습니다.

### [Improved Training Technique for Latent Consistency Models](https://arxiv.org/abs/2502.01441)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01441.png)

Vote: 7

Authors: Di Liu, Quan Dao, Khanh Doan, Dimitris Metaxas, Trung Le

- ***What's New***: 이 연구는 Latent Consistency Models의 훈련 성능을 개선하는 새로운 접근법을 제시합니다. 저자들은 latent space에서 발생하는 impulsive outliers를 효과적으로 처리하기 위해 Cauchy losses를 도입하였으며, 초기 타임스텝에서 diffusion loss와 optimal transport(OT) coupling을 적용하여 성능을 향상시켰습니다.
- ***Technical Details***: 연구는 latent space와 pixel space 간의 통계적 차이를 분석하면서 시작됩니다. 주된 기법으로는 Pseudo-Huber losses를 Cauchy losses로 대체하여 outliers의 영향을 줄이며, early timesteps에서 diffusion loss를 도입하고 mini-batch에서 OT coupling을 사용해서 학습의 안정성을 높였습니다. 또한 adaptive scaling-c scheduler와 Non-scaling LayerNorm을 도입하여 모델의 훈련 과정을 보다 견고하게 관리합니다.
- ***Performance Highlights***: 개선된 기술을 통해 latent consistency 모델은 1~2 스텝만으로도 높은 품질의 샘플을 생성할 수 있게 되었으며, CelebA-HQ, LSUN Church 및 FFHQ 데이터셋에서 FID와 Recall 성능이 이전보다 크게 향상되었습니다. 제안된 방법론은 iLCT 모델에 비해 현저히 낮은 FID 스코어와 높은 Recall을 보여주었습니다.

### [RandLoRA: Full-rank parameter-efficient fine-tuning of large models](https://arxiv.org/abs/2502.00987)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.00987.png)

Vote: 7

Authors: Anton van den Hengel, Hemanth Saratchandran, Frederic Z. Zhang, Cristian Rodriguez-Opazo, Ehsan Abbasnejad, Paul Albert

- ***What's New***: RandLoRA는 대형 모델의 파인 튜닝 과정에서 풀-랭크(full-rank) 업데이트가 가능하면서도 파라미터 효율성을 유지할 수 있는 새로운 방법론을 제안합니다. 이 방법론은 비학습(random) 저랭크(low-rank) 행렬의 선형 결합을 학습하여 기존 LoRA에서 발생했던 랭크 제약을 극복합니다.
- ***Technical Details***: RandLoRA는 학습 가능한 파라미터를 최소화하기 위해 고정된 랜덤 행렬에 대각 스케일링(matrix diagonal scaling)을 적용하여 최적화합니다. 이 방법은 시각적(Vision), 언어적(Language), 시각 언어(Vision-Language) 벤치마크에서 LoRA의 저랭크 한계를 극복하고자 하는 풀랭크 업데이트를 가능케 합니다.
- ***Performance Highlights***: RandLoRA는 다양한 실험을 통해, 풀-랭크 업데이트가 비전 및 언어 과제에서 유익하며, 특히 비전-언어 과제에서 LoRA와의 성능 차이를 크게 줄이며 때로는 완전히 해소함을 보입니다. 실험 결과, 낮은 샷(few-shot) 환경에서 RandLoRA가 경쟁력 있는 성능을 보였습니다.

### [Can LLMs Maintain Fundamental Abilities under KV Cache Compression?](https://arxiv.org/abs/2502.01941)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01941.png)

Vote: 7

Authors: Zeyu Li, Bo Li, Xiaowen Chu, Zhenheng Tang, Xiang Liu, Xuming Hu, Hong Chen, Xiuze Zhou, Peijie Dong

- **What's New**: 본 논문은 대형 언어 모델(LLMs)의 핵심 능력에 미치는 KV 캐시 압축(KV Cache Compression) 방법의 영향을 첫 번째로 체계적으로 연구하였습니다. 이를 통해 다양한 작업에서 주요 KV 캐시 압축 방법들의 성능 저하 패턴을 분석하였으며, 작업 특유의 성능 열화가 나타남을 발견했습니다.
- **Technical Details**: 여러 작업(세계 지식, 상식 추론, 산술 추론, 코드 생성, 안전성 및 긴 문맥 이해 및 생성)에 걸쳐 유명한 KV 캐시 압축 방법을 종합적으로 연구했습니다. 주목할 점은 DeepSeek R1 Distill 모델이 명령어 튜닝된 모델보다 더 강한 압축 허용성을 보였다는 것입니다. ShotKV라는 새로운 압축 접근 방식을 제안하였으며, 이는 프리필과 디코딩 단계를 각각 다르게 처리하면서 샷 수준의 의미적 일관성을 유지합니다.
- **Performance Highlights**: 산술 추론 작업은 특히 공격적인 압축에 민감하며, 성능 저하는 17.4%에서 43.3%까지 다양합니다. 그러나 ShotKV는 공격적인 압축 비율에서 긴 문맥 생성 작업에서 9%에서 18%의 성능 향상을 달성했습니다.

### [COCONut-PanCap: Joint Panoptic Segmentation and Grounded Captions for Fine-Grained Understanding and Generation](https://arxiv.org/abs/2502.02589)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.02589.png)

Vote: 6

Authors: Linjie Yang, Xiaojie Jin, Xiaohui Shen, Qihang Yu, Ali Athar, Chenglin Yang, Liang-Chieh Chen, Xueqing Deng

- ***What's New***: COCONut-PanCap는 파노픽 분할(Panoptic Segmentation)와 수반된 캡션(Grounded Captions)을 통한 정밀한 이미지 이해와 생성 지원을 목표로 하는 데이터셋을 소개합니다. COCO 데이터셋과 최신 COCONut 파노픽 마스크들을 기반으로 하여, 이 데이터셋은 더 정밀하고 장면 포괄적인 장면 설명을 제공하기 위해 설계되었습니다.
- ***Technical Details***: COCONut-PanCap 데이터셋은 인간 편집을 거쳐 세밀하게 주석된 설명들을 포함하여 파노픽 분할 마스크에 기반을 둔 캡션을 제공합니다. 이 데이터셋은 14만3천 개의 이미지를 주석하여, 완전한 객체 커버리지를 보장하며, 정교한 분할 마스크를 포함합니다. 모델은 LLaVA(라바)나 BERT와 같은 상업적 VLM(비전-언어 모델)을 사용하여 결합된 데이터로부터 자세한 캡션을 생성합니다.
- ***Performance Highlights***: COCONut-PanCap은 이미지 이해와 생성 작업에서 성능을 크게 향상시켰습니다. PanCaper-Pro 모델은 캡션 품질 지표에서 최상의 결과를 보이며, 특히 파노픽 기반 분할에서 PQ 점수 0.61을 달성하였습니다. 이 데이터셋은 다양한 멀티모달 이해 및 생성 작업에서 모델의 사전 학습 및 미세 조정의 성능을 크게 향상시켜, 향후 연구 방향성을 제시합니다.

### [SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders](https://arxiv.org/abs/2501.18052)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2501.18052.png)

Vote: 6

Authors: Kamil Deja, Bartosz Cywiński

- ***What's New***: SAeUron은 희소 오토인코더(Sparse Autoencoders; SAEs)를 활용하여 텍스트-이미지 확산 모델(Text-to-Image Diffusion Models)에서 원치 않는 개념을 제거할 수 있는 새로운 방법을 제안합니다. 이는 기존의 기계 학습이 종종 불투명하여 모델 변화의 이해가 어려운 점을 개선하고, 더욱 해석 가능한 방법을 제공합니다.
- ***Technical Details***: SAeUron은 고정밀 불필요 개념 제거를 위해 확산 모델의 내부 활성화에 대해 비지도 방식으로 훈련된 희소 오토인코더를 사용합니다. 이는 특정 개념과 관련된 해석 가능한 희소 피쳐를 포착함으로써 개념의 추출과 제거를 가능하게 합니다. 포괄적인 UnlearnCanvas 벤치마크를 통해 개체 및 스타일 제거에서의 최첨단 성능을 입증했습니다.
- ***Performance Highlights***: SAeUron은 스타일 제거에서 경쟁 방법들 대비 뛰어난 성능을 보여주었고, 다중 개념 제거에서도 높은 견고성을 나타냈습니다. 이는 적대적 공격에도 불구하고 원치 않는 콘텐츠 생성을 효과적으로 막는 것을 시사합니다. 또한, 다중 개념을 동시에 제거할 수 있는 능력을 증명했습니다.

### [The Surprising Agreement Between Convex Optimization Theory and Learning-Rate Scheduling for Large Model Training](https://arxiv.org/abs/2501.18965)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2501.18965.png)

Vote: 5

Authors: Umut Simsekli, Fabian Schaipp, Alexander Hägele, Francis Bach, Adrien Taylor

- ***What's New***: 이 논문은 대규모 모델 학습의 학습률 스케줄링(learning-rate scheduling)이 비평활(convex) 최적화 이론의 성능 경계와 놀라울 정도로 유사한 방식으로 작동한다는 것을 보여줍니다. 특히, 상수 스케줄과 선형 쿨다운(linear cooldown)의 이론적 성능 경계를 제공합니다.
- ***Technical Details***: 논문에서는 비평활 확률적(convex) 설정에서의 서브옵티멀리티 경계(suboptimality bound)를 통해 학습률 스케줄링의 성능을 재현할 수 있음을 보여줍니다. 이를 활용해 124M과 210M Llama 모델의 학습을 최적 학습률을 사용하여 지속적으로 개선할 수 있습니다.
- ***Performance Highlights***: 제안된 'wsd' 스케줄은 쿨다운 기간 이후 손실이 급격히 감소하는 특징이 있으며, 이는 비평활 확률적(convex) 최적화 이론에 의해 설명될 수 있습니다. 실험 결과 최적 학습률을 사용하는 것이 성능을 개선한다는 것을 입증했습니다.

### [Rethinking Mixture-of-Agents: Is Mixing Different Large Language Models Beneficial?](https://arxiv.org/abs/2502.00674)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.00674.png)

Vote: 5

Authors: Mengzhou Xia, Wenzhe Li, Yong Lin, Chi Jin

- ***What's New***: 이 연구에서는 다양한 대형 언어 모델(Large Language Models; LLMs)을 혼합하는 것이 정말로 유익한지를 재검토합니다. 이를 위해 단일 최고 성능 모델만을 사용하는 자기 혼합(Mixture-of-Agents; MoA) 기법인 Self-MoA를 제안합니다. 실험 결과, Self-MoA가 기존의 MoA를 성능 면에서 능가함을 발견하였습니다.
- ***Technical Details***: Self-MoA는 단일 최고 성능 모델에서 반복적으로 샘플링된 출력을 집계하여 높은 품질의 응답을 생성합니다. 이는 AlpacaEval 2.0 및 여러 벤치마크(MMLU, CRUX, MATH)에서 3.8% 이상의 개선을 보였으며, 다양한 MoA 설정에서 품질 대 다양성의 상충 관계를 체계적으로 조사했습니다.
- ***Performance Highlights***: AlpacaEval 2.0 벤치마크에서 Self-MoA는 기존의 MoA보다 6.6% 높은 성능을 기록하였으며, 이는 Mixed-MoA를 능가하는 결과를 보여줍니다. 또한, TaskBest 설정에서는 더욱 향상된 성능을 보여주며, 다중 작업 벤치마크에서도 뛰어난 성과를 달성했습니다.

### [Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion](https://arxiv.org/abs/2501.18804)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2501.18804.png)

Vote: 4

Authors: Dian Chen, Vitor Guizilini, Greg Shakhnarovich, Muhammad Zubair Irshad, Rares Ambrus

- ***What's New***: MVGD(Multi-View Geometric Diffusion)는 신규 관점에서 이미지와 깊이 맵을 생성하는 최첨단 메소드로서, 입력 뷰의 임의적인 수에 기반하여 새로운 시점을 직접 픽셀 단위로 생성합니다. 본 연구는 중간 3D 표현 없이 스케일 인식 및 다중 시점 일관성을 유지하여 새로운 뷰 및 깊이 예측을 가능케 합니다.
- ***Technical Details***: 본 연구에서는 효율적인 학습을 위해 60백만 개 이상의 다양한 시나리오에서 수집된 다중 뷰 샘플을 사용하며, 이질적인 조건 하에서도 일관된 학습이 가능하도록 새로운 기술을 제안합니다. MVGD 아키텍처는 학습 가능한 태스크 임베딩을 사용해 서로 다른 모달리티에 대해 디퓨전 과정을 안내하며, 최종적인 지표에 도달하도록 합니다. 또한 작은 모델을 점진적으로 미세 조정함으로써 큰 모델을 효율적으로 학습할 수 있는 전략이 제안됩니다.
- ***Performance Highlights***: MVGD는 다양한 신규 뷰 합성 벤치마크에서 기존 기준을 초과하며, 스테레오와 비디오 깊이 추정 작업에서도 최고 수준의 결과를 보입니다. MVGD는 최대 수천 개의 컨디셔닝 뷰를 처리할 수 있으며, 점진적 컨디셔닝 전략을 통해 긴 시퀀스에서도 다중 시점 일관성을 더욱 향상시킵니다. 모델 복잡도가 증가함에 따라 성능이 개선되는 유망한 스케일링 특성을 보여줍니다.

### [Lifelong Sequential Knowledge Editing without Model Degradation](https://arxiv.org/abs/2502.01636)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01636.png)

Vote: 4

Authors: Gopala Anumanchipalli, Ahmed Alaa, Akshat Gupta, Thomas Hartvigsen, Phudish Prateepamornkul, Maochuan Lu

- **What's New**: 이 논문은 대규모 연속적 지식 편집(sequential knowledge editing)이 가져오는 모델 저하의 문제를 해결하면서 오리지널 모델의 다운스트림 성능을 유지하는 ENCORE라는 새로운 방법론을 제시합니다. ENCORE는 초기 정지와 노름 제한을 결합하여 10,000개의 연속적인 지식 편집을 수행하면서도 성능 저하가 없어 빠르고 강력한 지식 편집을 가능하게 합니다.
- **Technical Details**: ENCORE는 'locate-then-edit' 방식의 두 가지 주요 문제를 해결합니다. 첫 번째는 편집된 사실에 대한 과적합을 줄이기 위한 Most-Probable Early Stopping (MPES)입니다. 두 번째는 편집 매트릭스의 과도한 노름 증가를 제어하기 위한 프로베니우스 노름 제한(Frobenius-norm constraint)입니다. ENCORE는 Llama3-8B 모델에서 MEMIT과 AlphaEdit보다 각각 61%, 64% 더 빠르게 작동합니다.
- **Performance Highlights**: ENCORE는 Llama2-7B와 Llama3-8B 모델에서 각각 89%와 93%의 편집 점수(Edit Score)를 기록하며, 이는 기존 방법들보다 향상된 성능을 나타냅니다. 또한, ENCORE는 높은 편집 점수와 더불어 10,000개의 편집 후에도 안정적인 다운스트림 성능을 유지했습니다.

### [Concept Steerers: Leveraging K-Sparse Autoencoders for Controllable Generations](https://arxiv.org/abs/2501.19066)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2501.19066.png)

Vote: 4

Authors: Deepti Ghadiyaram, Dahye Kim

- ***What's New***: 이 연구는 k-희소 오토인코더(k-Sparse Autoencoders; k-SAE)를 활용하여 텍스트-이미지 생성기의 컨트롤러블 개념 조작을 구현한 새로운 프레임워크를 제안합니다. 이 접근법은 기존 모델을 재훈련하거나 LoRA 어댑터 없이도 개념을 효율적이고 해석 가능하게 조작할 수 있습니다.
- ***Technical Details***: 연구에서는 k-SAE를 사용하여 텍스트 임베딩의 잠재 공간에서 해석 가능한 단일 의미 개념을 식별하고, 이 개념을 통해 생성 프로세스를 특정 개념(예: 노출, 폭력)으로부터 멀어지거나 가까워지도록 조정하거나 새로운 개념을 도입할 수 있게끔 합니다. k-SAE는 한 번 훈련되면 불안전한 개념 제거에서 20.01%의 개선을 이루고, 스타일 변형에서 5배 빠르게 처리되며, 시각적 품질 손상을 최소화합니다.
- ***Performance Highlights***: Concept Steerer는 불안전한 개념 제거 분야에서 기존 최고 방법 대비 5배 빠르게 성과를 발휘하며, 시각적 품질을 훼손하지 않으면서 안정적인 이미지 생성을 지원합니다. 또한, 적대적 프롬프트 조작에 대해 20.01%의 성능 향상이 있습니다.

### [Fast Encoder-Based 3D from Casual Videos via Point Track Processing](https://arxiv.org/abs/2404.07097)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2404.07097.png)

Vote: 3

Authors: Haggai Maron, Yoni Kasten, Wuyue Lu

- ***What's New***: TRACKSTO4D는 캐주얼 비디오(Casual Videos)로부터 3D 구조와 카메라 위치를 빠르게 추론할 수 있는 새로운 접근법입니다. 이는 기존의 방법보다 최대 95%의 실행 시간을 절감합니다.
- ***Technical Details***: TRACKSTO4D는 2D 점 트랙(point tracks)을 입력으로 사용하여 이를 처리하기 위한 특별한 아키텍처를 설계하였습니다. 이 아키텍처는 입력 데이터의 대칭성을 고려하고, 저차수 근사(low-rank approximation)를 통해 동적 콘텐츠의 움직임을 효과적으로 표현할 수 있도록 설계되었습니다. 또한, 3D 감독 감독 없이 2D 점 트랙을 활용하여 비지도 학습(unsupervised learning) 방식으로 훈련됩니다.
- ***Performance Highlights***: TRACKSTO4D는 최신 기법들과 비슷한 수준의 정확한 3D 포인트 클라우드와 카메라 위치를 재구성하면서도 대폭적인 실행 시간 단축을 달성하였습니다. 실험 결과, 새로운 비디오와 새로운 의미 카테고리에 대해서도 우수한 일반화 성능을 보였습니다.

### [A Study on the Performance of U-Net Modifications in Retroperitoneal Tumor Segmentation](https://arxiv.org/abs/2502.00314)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.00314.png)

Vote: 3

Authors: David J. Foran, Ehsan Khodapanah Aghdam, Alexander Manzella, Wenjin Chen, Ilker Hacihaliloglu, Moein Heidari, Rebecca Scalabrino, Daniel Hsu

- ***What's New***: 본 연구는 새로운 CT 데이터셋을 소개하며, 스테이트 오브 더 아트(SOTA) U-Net 및 변형을 평가하여 복막 뒤 종양(segmentation)에 대한 성능을 조사하였습니다. 특히, Vision-xLSTM(ViLU-Net) 아키텍처가 도입되어, 변형된 U-Net 기반 네트워크가 우수한 성능을 보이며 대부분의 최신 방법들과 비교하여 향상된 정확성을 제공합니다.
- ***Technical Details***: 새로운 데이터셋에는 82개의 3D 스캔 사례가 포함되어 있으며, 전문가가 세그먼트 맵을 정밀하게 주석 달았습니다. ViLU-Net 모델은 CNN과 Vision-xLSTM을 통합하여 세밀한 텍스처와 지역 패턴을 캡처하는 동시에 글로벌 문맥을 효과적으로 인코딩합니다. ViL 블록은 공간 및 시간적 종속성을 포착하며, 인스턴스 정규화 및 LeakyReLU가 포함된 CNN 기반의 컨볼루션 스템과 encoder-decoder 구조를 띄고 있습니다.
- ***Performance Highlights***: ViLU-Net은 최신 모델들 중 평균 Dice Similarity Coefficient(DSC), Normalized Surface Distance(NSD), Intersection over Union(IoU)에서 최고 점수를 기록하고, Hausdorff Distance(HD)에서는 가장 낮은 점수를 기록하여 종양 경계를 정밀하게 구분할 수 있음을 보여주었습니다. 특히 복부 CT 데이터셋에서 ViLU-Net은 DSC에서 0.8594, NSD에서 0.8944의 우수한 성능을 보였습니다.

### [LongDPO: Unlock Better Long-form Generation Abilities for LLMs via Critique-augmented Stepwise Information](https://arxiv.org/abs/2502.02095)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.02095.png)

Vote: 3

Authors: Shanghang Zhang, Jiali Zeng, Shuo Wang, Fandong Meng, Jie Zhou, Bowen Ping

- ***What's New***: LongDPO는 대형 언어 모델(LLMs)의 장문 생성 능력을 향상시키기 위한 새로운 접근 방식으로, 과정 슈퍼비전을 통해 더욱 세부적으로 학습할 수 있도록 합니다. 이 방법은 외부 비판을 통합하여 선택된 후보의 질을 높이고자 합니다.
- ***Technical Details***: LongDPO는 몬테카를로 트리 탐색(Monte Carlo Tree Search; MCTS)을 통해 단계적 우선 순위 데이터를 수집하며, 글로발 메모리 풀(Global Memory Pool)을 사용하여 일관성을 유지합니다. 선택된 후보의 질을 개선하기 위해 외부 비판을 활용하고, 수집된 단계별 우선 순위 데이터를 이용하여 훈련하는 스텝별 DPO를 적용합니다.
- ***Performance Highlights***: LongDPO를 적용한 모델은 LongBench-Write-en 및 LongGenBench와 같은 장문 생성 벤치마크에서 뛰어난 성능을 보여주었으며, 일반적인 벤치마크에서도 거의 손실 없는 성능을 유지했습니다. 이는 주로 각 스텝에서의 세부적인 학습을 통해 높은 품질의 데이터를 얻을 수 있었기 때문입니다.

### [Learning to Generate Unit Tests for Automated Debugging](https://arxiv.org/abs/2502.01619)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01619.png)

Vote: 3

Authors: Justin Chih-Yao Chen, Mohit Bansal, Archiki Prasad, Elias Stengel-Eskin, Zaid Khan

- ***What's New***: 이 연구에서는 UTGEN이라는 시스템을 소개하며, 이는 대형 언어 모델(LLMs)이 오작동하는 코드에서 오류를 드러내는 유닛 테스트(Unit Tests; UTs) 입력을 생성하고 이에 대한 적절한 출력도 예측할 수 있게 하는 새로운 접근법입니다. 또한, 자동화된 디버깅 파이프라인인 UTDEBUG에 통합되어 모델이 생성한 테스트로부터 피드백을 받아 디버깅을 효과적으로 수행할 수 있도록 합니다.
- ***Technical Details***: UTGEN은 코드 생성 벤치마크에서 기존 오류를 시뮬레이션하여 실패하는 유닛 테스트를 생성하고, 시나리오별 체인-오브-사고(Chain-of-Thought) 방식의 논리를 추가하여 학습 데이터를 보강합니다. UTDEBUG는 다중 생성된 UT 기반으로 수정을 검증하고 역추적하는 과정을 통해 과적합을 피하는 방식을 사용합니다. 또한 테스트 시 계산을 확장함으로써 UT 출력 예측을 향상시킵니다.
- ***Performance Highlights***: UTGEN은 오류를 드러내는 유닛 테스트 입력과 정확한 유닛 테스트 출력을 포함하여 7.59%의 향상을 보이며, UTTDEBUG와 함께 사용할 때, UTGEN의 유닛 테스트 피드백을 통해 Qwen-2.5 7B 모델의 pass@1 정밀도를 HumanEval-Fix에서 3% 이상, MBPP+의 자사 고난도 디버깅 세트에서 12.35% 이상 증가시켰습니다.

### [Unraveling the Capabilities of Language Models in News Summarization](https://arxiv.org/abs/2501.18128)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2501.18128.png)

Vote: 3

Authors: Abdurrahman Odabaşı, Göksel Biricik

- ***What's New***: 이번 논문은 최근 20개의 언어 모델(Language Models)을 뉴스 요약(news summarization) 작업에 대해 벤치마킹한 결과를 상세히 다루고 있습니다. 특히 소형 모델(small models)을 집중적으로 분석하여 뉴스 기사의 요약 작업에서의 성능을 평가하였습니다.
- ***Technical Details***: 뉴스 요약 작업을 위해 3개의 데이터셋(CNN/Daily Mail, Newsroom, Extreme Summarization)을 활용하였으며, 각기 다른 스타일의 기사를 대상으로 zero-shot 학습과 few-shot 학습 설정에서 모델의 성능을 평가했습니다. 모델들은 자동화된 메트릭(ROUGE, METEOR, BERTScore) 및 인간 평가, AI 기반 평가를 통해 테스트되었습니다. 특히 demonstration examples를 포함함으로써 성능이 향상되지 않고 오히려 악화되는 현상이 관찰되었습니다.
- ***Performance Highlights***: GPT-3.5-Turbo와 GPT-4는 탁월한 성능을 보였으며, 일부 공개 모델, 예를 들어 Qwen1.5-7B, SOLAR-10.7B-Instruct-v1.0, Meta-Llama-3-8B 및 Zephyr-7B-Beta는 많은 모델들과 비교해 유망한 결과를 보여주었습니다. 이들 모델은 대형 모델에 비해 뉴스 요약 작업에서 경쟁력 있는 대안으로 자리매김하고 있습니다. 소형 모델이 자동 메트릭에서는 높은 점수를 받았으나 인간 평가에서는 낮은 점수를 받는 경우도 관찰되었습니다.

### [INT: Instance-Specific Negative Mining for Task-Generic Promptable Segmentation](https://arxiv.org/abs/2501.18753)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2501.18753.png)

Vote: 2

Authors: Jian Hu, Shaogang Gong, Zixu Cheng

- ***What's New***: 본 연구는 INT(Instance-Specific Negative Mining for Task-Generic Promptable Segmentation)를 제안하여, 이미지 세분화 작업에서 일반적인 프로프트(generic prompt)를 사용해 보다 정확한 인스턴스별 프로프트(instance-specific prompt)를 생성하는 방식을 소개합니다. 이는 다양한 이미지 내에서 주어진 작업에 따라 여러 대상을 효과적으로 분할할 수 있는 기술을 제공합니다.
- ***Technical Details***: INT는 두 가지 주요 구성요소로 이루어져 있습니다: (1) 인스턴스별 프로프트 생성으로, 이는 잘못된 정보를 점진적으로 필터링하며 보다 정확한 프로프트 생성을 목표로 합니다. (2) 의미적 마스크 생성으로, 인스턴스별 프로프트가 이미지 인스턴스의 의미와 정확히 일치하도록 합니다. 이를 위해 VLM(Vision-Language Model)을 이용해 기존 출력과 마스킹된 출력의 변화를 분석하여 오류를 발견하고 수정합니다.
- ***Performance Highlights***: INT는 카멜레온(CHAMELEON), CAMO, COD10K와 같은 6개의 데이터셋을 기반으로 한 실험을 통해 그 효과성과 강력함을 입증하였습니다. INT는 특히 캠플리지 객체 검출(COD) 및 의료 이미지 세분화(MIS)와 같은 복잡한 분할 작업에서 기존 방법보다 높은 성능을 보였습니다. 본 연구에서는 VLM의 예측을 점진적 오류 채굴을 통해 보정하며, 이는 주어진 작업에서 정확한 인스턴스별 프로프트 생성을 가능하게 했습니다.

### [Generating Multi-Image Synthetic Data for Text-to-Image Customization](https://arxiv.org/abs/2502.01720)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01720.png)

Vote: 2

Authors: Jun-Yan Zhu, Xi Yin, Ishan Misra, Nupur Kumari, Samaneh Azadi

- ***What's New***: 이 논문은 텍스트-이미지 모델의 개인화 개선을 위해 멀티 이미지 기반의 합성 데이터를 생성하는 새로운 접근법을 제안합니다. 이는 고품질의 Synthetic Customization Dataset (SynCD)을 생성하여 다양한 실험을 통해 기존의 방법보다 우수한 성능을 보입니다.
- ***Technical Details***: 이 연구는 기존의 텍스트-이미지 모델을 활용하여 동일한 객체의 서로 다른 조명, 배경 및 자세로 구성된 여러 이미지를 포함한 Synthetic Customization Dataset (SynCD)을 생성합니다. 공유 주의 메커니즘(shared attention mechanism)에 기반하여 입력 이미지에서 세밀한 시각적 정보를 통합하는 새로운 인코더 아키텍처를 제안하며, 인퍼런스 단계에서 텍스트 및 이미지 안내 벡터를 정규화하여 노출 과다 문제를 완화합니다.
- ***Performance Highlights***: 제안된 모델은 합성 데이터셋과 새로운 인코더 및 인퍼런스 알고리즘을 사용하여 표준 개인화 벤치마크에서 기존의 튜닝-프리(tuning-free) 메서드를 능가합니다. 인간 평가 및 다양한 자동화된 평가 지표에서 텍스트와 이미지 조건에 대한 준수도가 상승하였음을 확인했습니다.

### [Current Pathology Foundation Models are unrobust to Medical Center Differences](https://arxiv.org/abs/2501.18055)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2501.18055.png)

Vote: 2

Authors: Edwin D. de Jong, Eric Marcus, Jonas Teuwen

- ***What's New***: 본 연구는 병리학 기반 모델(Pathology Foundation Models; FMs)이 서로 다른 의료 센터 간의 차이에 따라 얼마나 견고하지 않은지를 평가합니다. 새로운 견고성 지표(Robustness Index)를 제안하여 생물학적 특징과 혼란을 초래하는 특징 간의 우위를 측정합니다. 현재 연구된 모든 병리학 기반 모델들은 특정 의료 센터의 영향을 강하게 받는다고 밝혀졌습니다.
- ***Technical Details***: 제안된 견고성 지표는 생성된 임베딩 공간의 생물학적 이웃과 의료 센터 이웃의 비율을 계산하는 방식으로, 10개의 공공 데이터 기반 병리학 모델들을 평가하였습니다. 결과적으로, 대부분의 모델들이 생물학적 정보를 초월하여 의료 센터에 기반한 패턴을 강하게 나타내고 있음을 시각적으로 확인하였습니다. 코사인 거리(cosine distance)를 사용하여 k-nearest neighbors 분석을 수행하였습니다.
- ***Performance Highlights***: 10개의 모델 중 Virchow2만이 견고성 지표가 1을 초과하였으며, 이는 생물학적 정보가 혼란을 초래하는 정보보다 더 강하게 표현되고 있음을 나타냅니다. 다른 모든 모델들은 의료 센터 정보에 강하게 조직되어 높은 수준의 동일 센터 혼란(same-center confounders)을 보여 주고 있습니다. t-SNE를 이용하여 임베딩 공간을 2D로 시각화한 결과, 대부분의 모델은 암 유형보다 의료 센터별 클러스터링(clustering)을 더 명확히 나타냅니다.

### [Language Models Prefer What They Know: Relative Confidence Estimation via Confidence Preferences](https://arxiv.org/abs/2502.01126)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01126.png)

Vote: 2

Authors: Percy Liang, Vaishnavi Shrivastava, Ananya Kumar

- ***What's New***: 이 연구는 언어 모델(Language Models; LMs)의 자신감 추정 방식을 개선하기 위해 상대적 자신감 추정(Relative Confidence Estimation)을 제안하였습니다. 이는 모델이 단일 질문에 대한 자신감을 직접 평가하는 절대적 자신감 추정(Absolute Confidence Estimation)보다 일관성 있는 자신감 점수를 제공함을 입증하였습니다.
- ***Technical Details***: 상대적 자신감 추정은 모델이 여러 질문에 대해 자신감을 비교하도록 하여 어느 질문에 더 자신 있는지를 묻는 방식입니다. 그런 다음 이런 비교를 엘로 레이팅(Elo Rating)이나 브래들리-테리(Bradley-Terry)와 같은 순위 집계 방법(Rank Aggregation Methods)을 사용하여 자신감 점수로 변환합니다. 본 연구는 5개의 최첨단 언어 모델(GPT-4, GPT-4o, Gemini 1.5 Pro, Claude 3.5 Sonnet, Llama 3.1 405B)에서 이러한 방법을 테스트했습니다.
- ***Performance Highlights***: 상대적 자신감 추정은 직접적인 절대적 자신감 추정 방식보다 평균 선택적 분류 AUC(Selective Classification AUC)에서 3.5% 향상된 결과를 보였습니다. 특히 Llama 3.1 405B 모델에서는 6.1%의 AUC 향상이 관찰되었습니다. 이는 상대적 자신감 추정이 다양한 AI 모델에 걸쳐 더 신뢰성 있는 성능을 제공할 수 있음을 보여줍니다.

### [ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](https://arxiv.org/abs/2502.00299)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.00299.png)

Vote: 1

Authors: Zeyu Li, Bo Li, Xiaowen Chu, Zhenheng Tang, Xiang Liu, Xuming Hu, Peijie Dong

- ***What's New***: ChunkKV는 대형 언어 모델(LLM)의 긴 문맥 추론 시 메모리 비용을 줄이기 위해 제안된 새로운 키-값(KV) 캐시 압축 방법입니다. 이 방법은 연속된 토큰 블록을 기본 압축 단위로 사용하여 가장 중요한 의미적 블록을 유지하는 방식으로 설계되었습니다.
- ***Technical Details***: ChunkKV는 토큰을 청크(chunk) 단위로 그룹화하여 보존할지 폐기할지를 결정합니다. 그리고 여러 계층에 걸쳐 보존된 인덱스의 유사성이 높다는 점을 이용하여 계층별 인덱스 재사용(layer-wise index reuse) 기술을 제안하여 추가적인 계산 비용을 줄입니다. 이 방법은 LongBench 및 Needle-In-A-HayStack와 같은 최신 벤치마크를 통해 평가되었으며, 최대 10%의 성능 개선을 이루었습니다.
- ***Performance Highlights***: ChunkKV는 기존의 KV 캐시 압축 방법보다 효율성과 정확성 면에서 뛰어난 성능을 보입니다. 긴 문맥에서 필요한 중요한 정보를 선택적으로 보존함으로써, 다양한 모델과 벤치마크에서 현재의 최첨단 성능을 달성했습니다.

### [Federated Sketching LoRA: On-Device Collaborative Fine-Tuning of Large Language Models](https://arxiv.org/abs/2501.19389)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2501.19389.png)

Vote: 1

Authors: Liangqi Yuan, Christopher G. Brinton, Wenzhi Fang, Seyyedali Hosseinalipour, Dong-Jun Han

- ***What's New***: 이 논문은 FSLoRA(Federated Sketching LoRA)를 소개하며, 이는 대형언어모델(LLM)의 협력적 미세조정을 위한 새로운 방법론입니다. 기기에서 글로벌 LoRA 모듈의 하위행렬을 선택적으로 업데이트할 수 있게 하는 스케칭 메커니즘을 사용하여, 디바이스 특정한 통신 및 계산 제약에 유연하게 대응합니다.
- ***Technical Details***: FSLoRA는 스케칭 비율을 통해 디바이스에서의 하위행렬 랭크를 조절하는 기계를 도입하여, 애플리케이션 요구사항에 맞춘 최적화를 가능하게 합니다. 개인 디바이스는 시스템 자원 제약 내에서 스케칭 비율을 조정하여 로컬 개선을 맞춤화할 수 있습니다. FSLoRA는 수렴 속도에 대한 스케칭 비율의 영향을 철저히 분석하며, 이는 통신 및 계산 비용과의 복잡한 균형을 보여줍니다.
- ***Performance Highlights***: RoBERTa 모델과 GLUE 벤치마크, LLaMA-3.2-3B 모델과 Commonsense Reasoning 벤치마크에서, FSLoRA는 다양한 베이스라인과 비교하여 높은 테스트 정확도 및 자원 효율성을 보여주며, 특히 테스트 정확도와 학습 시간에서 우수한 성능을 입증하였습니다.

### [Sample, Scrutinize and Scale: Effective Inference-Time Search by Scaling Verification](https://arxiv.org/abs/2502.01839)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.01839.png)

Vote: 1

Authors: Pranjal Awasthi, Eric Zhao, Sreenivas Gollapudi

- ***What's New***: 이 논문은 테스트 시점 계산을 활용하여 추론 성능을 강화하는 샘플링 기반 검색(Sampling-Based Search)의 스케일링 트렌드를 연구했습니다. 특히, 랜덤 샘플링과 직접적인 자기 검증을 사용하는 최소한의 구현만으로도 지속적인 성능 향상을 얻을 수 있다는 점을 발견했으며, Gemini v1.5 Pro 모델의 추론 성능이 o1-Preview보다 뛰어나다는 것을 보여줍니다.
- ***Technical Details***: 이 연구는 모델이 생성한 여러 응답 중 최고를 직접 검증을 통해 선택하는 샘플링 기반 검색 방법론을 사용합니다. 샘플링 수(k_inf)를 조정하여 검색을 확장하고, 검증 수(k_verif)를 조정하여 검증 능력을 확장합니다. 또한, 서로 다른 응답 비교를 통해 오류와 환각이 발생한 위치를 식별하고, 필요에 따라 답변의 스타일을 변경하여 검증 가능성을 높입니다.
- ***Performance Highlights***: Gemini v1.5 Pro 모델은 AIME와 같은 복잡한 벤치마크에서 샘플링 기반 검색을 활용해 Consistency@200을 능가하는 Verification@200 결과를 보였으며, 특히 정확한 결정을 요구하는 문제에서 강력한 성능을 발휘했습니다. 이는 모델이 손쉽게 올바른 솔루션을 필터링하여 찾을 수 있는 능력을 보여줍니다.

### [Activation Approximations Can Incur Safety Vulnerabilities Even in Aligned LLMs: Comprehensive Analysis and Defense](https://arxiv.org/abs/2502.00840)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.00840.png)

Vote: 0

Authors: Lipeng He, Mingli Song, Jian Liu, Jian Lou, Zunlei Feng, Jiawen Zhang, Xiaohu Yang, Kui Ren, Kejia Chen, Dan Li

- ***What's New***: 이 연구에서는 대형 언어 모델(LLMs)의 활성화 근사(Activation Approximation) 기술이 안전성 문제를 초래할 수 있음을 밝혔습니다. 이는 현재의 안전성 정렬 메커니즘에서는 적절히 해결되지 않으며, 모델의 유용성을 저하시키지 않고 공격 성공률을 높일 수 있는 잠재적 취약성을 드러냅니다.
- ***Technical Details***: 이 논문에서는 활성화 근사 기법들을 세 가지 주요 범주(Activation Polynomialization, Sparsification, Quantization)로 나누어 10개의 안전 정렬된 LLM에서 실행된 첫 번째 체계적 안전 평가를 수행합니다. 분석 결과, 초기 몇 개의 레이어에서의 활성화 근사 오류가 안전성에 가장 큰 영향을 미친다고 밝혀졌습니다. 연구는 활성화 오류가 악성 프롬프트를 무해한 영역으로 이동시켜 안전 체크를 회피할 수 있음을 보입니다.
- ***Performance Highlights***: Llama-3.1-8B-Instruct 모델의 경우, 활성화 근사는 공격 성공률(ASR)을 0.19%에서 69.23%로 증가시키는 것으로 나타났으며, 유용성에 거의 영향을 미치지 않았습니다. 제안된 QuadA 방법은 다양한 공격 및 활성화 근사 정도에 대해 안전한 출력을 유지, 짧은 코드 수정만으로도 더욱 강력한 안전성을 제공할 수 있음을 실험적으로 증명합니다.

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
