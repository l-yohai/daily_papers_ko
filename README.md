# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using GPT-4o.

Thanks to [@AK391](https://github.com/AK391) for great work.

## Daily Papers (2025-02-19)

### [Soundwave: Less is More for Speech-Text Alignment in LLMs](https://arxiv.org/abs/2502.12900)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.12900.png)

Vote: 62

Authors: Benyou Wang, Yuhao Zhang, Fan Bu, Ruiyu Zhang, Haizhou Li, Zhiheng Liu

- ***What's New***: Soundwave는 고효율적인 훈련 전략과 새로운 아키텍처를 활용하여 적은 양의 데이터만으로도 높은 성능을 발휘하는 음성-텍스트 정렬 모델입니다. 이 연구는 기존의 방대한 규모의 주석 데이터에 의존하지 않고, 적은 데이터로도 우수한 성능을 나타내며, 특히 고급 음성 번역 및 AIR-Bench 음성 작업에서 향상된 성능을 보여줍니다.
- ***Technical Details***: Soundwave는 표현 공간 격차와 시퀀스 길이 불일치를 해결하기 위해 두 단계의 훈련 프레임워크를 제안합니다. 첫 번째 단계에서는 변환기 계층과 연속적 대화(adapters)를 사용하여 음성과 텍스트 간의 표현 공간을 일치시키고, 두 번째 단계에서는 음성 시퀀스의 길이를 줄입니다. 보다 효과적인 정렬을 위해 고품질의 음성 인식 데이터를 수집하고 수동으로 오디오 라벨을 주석합니다.
- ***Performance Highlights***: Soundwave 모델은 Qwen2-Audio를 능가하며, AI-R 벤치의 여러 음성 기초 작업에서 뛰어난 성능을 보여줍니다. 특히, 1만 시간의 제한된 훈련 데이터로도 최첨단 성능을 달성합니다. 다른 음성 LLM(Speech LLM)과 비교해 적은 데이터와 낮은 훈련 비용으로 문맥 추론 능력을 확보하며, AI를 활용한 다양한 음성 번역 작업에서도 우수한 성능을 나타냈습니다.

### [Cramming 1568 Tokens into a Single Vector and Back Again: Exploring the Limits of Embedding Space Capacity](https://arxiv.org/abs/2502.13063)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.13063.png)

Vote: 51

Authors: Yuri Kuratov, Mikhail Burtsev, Aydar Bulatov, Mikhail Arkhipov

- ***What's New***: 이 연구는 대형 언어 모델(LLMs)의 입력 임베딩 공간의 용량 한계에 대한 새로운 통찰력을 제공합니다. 연구자들은 1568개의 토큰을 단일 벡터로 압축하고 다시 복원하는 실험을 통해, 기존 방법보다 10배 더 높은 압축 비율(x1500)을 달성했음을 보여주었습니다.
- ***Technical Details***: 연구는 입력 토큰 시퀀스를 최적화된 [mem] 벡터로 압축하여 LLM에서 사용할 수 있도록 합니다. 이는 Transformer 아키텍처의 자기 주의 메커니즘의 계산 비용을 줄이기 위한 것으로, 언어 모델의 모든 매개변수가 고정된 상황에서 특정 입력 임베딩만을 최적화합니다. 이 연구는 입력 벡터로부터 재구성 가능한 최대 토큰 수를 평가하고, 정보 이득 및 토큰 이득을 측정하는 새로운 지표를 제안합니다.
- ***Performance Highlights***: 테스트 결과, Llama-3.1-8B 모델은 단일 벡터로부터 최대 1568개의 토큰을 정확하게 재구성할 수 있었으며, 여러 모델에서 토큰 압축 성능이 선형적으로 확장된다는 사실을 확인했습니다. 특히, Llama-3.2-1B는 단 16개의 벡터로 7168개의 토큰을 완벽히 재생성할 수 있었습니다. 이러한 결과는 대형 모델의 임베딩 공간 활용 능력에 대한 새로운 가능성을 제시합니다.

### [Continuous Diffusion Model for Language Modeling](https://arxiv.org/abs/2502.11564)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11564.png)

Vote: 41

Authors: Jaehyeong Jo, Sung Ju Hwang

- ***What's New***: 언어 모델링을 위한 새로운 연속 확산 모델(Continuous Diffusion Model)이 제안되었습니다. 이는 이산 데이터의 기하학적 구조를 통합한 모델로, 이산 확산과 연속 흐름 간의 연결을 통해 기존 이산 확산 모델을 일반화하는 접근을 소개합니다.
- ***Technical Details***: 이 모델은 Fisher-Rao 메트릭을 사용하여 통계적 다양체(Statistical Manifold) 상의 흐름을 정의합니다. 기저 카테고리 분포의 기하학을 활용하여 이산 데이터를 연속 매개변수로 재구성하며, 고차원 다양체의 문제를 해결하기 위해 방사형 대칭(radial symmetry) 기반의 시뮬레이션 없는 훈련 프레임워크를 제안합니다.
- ***Performance Highlights***: 실험 결과, 제안된 연속 확산 모델은 기존 이산 확산 모델을 능가하고 자동회귀 모델의 성능에 근접함을 보였습니다. Text8 및 One Billion Words 데이터셋에서 낮은 비트 당 문자 및 낮은 당혹지수(perplexity) 결과를 기록하였으며, 생물학적 시퀀스 설계 및 이미지 모델링에서도 우수한 성능을 발휘했습니다.

### [Phantom: Subject-consistent video generation via cross-modal alignment](https://arxiv.org/abs/2502.11079)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11079.png)

Vote: 41

Authors: Tianxiang Ma, Zhuowei Chen, Bingchuan Li, Xinglong Wu, Jiawei Liu, Qian He, Lijie Liu

- ***What's New***: Phantom은 주제 일관 비디오 생성(Subject-consistent Video Generation)을 위한 새로운 방법론을 제안합니다. 이는 참조 이미지를 기반으로 주제 요소를 추출하고 텍스트 명령에 따라 일관된 비디오를 생성하는 '주제-비디오' 접근 방식을 채택합니다. Phantom은 텍스트, 이미지 및 비디오의 삼중(triplet) 데이터 구조를 사용하여 교차 모드 정렬(cross-modal alignment)을 달성하며, 사람의 ID 보존 등 기존의 주제 일관 비디오 생성 기법보다 개선된 성능을 제공합니다.
- ***Technical Details***: Phantom은 기존 영상 생성 모델을 기반으로, 새로운 텍스트-이미지 주입 방식으로 교차 모드 데이터 형식을 효과적으로 학습하는 구조를 갖추고 있습니다. 비디오, 이미지 및 텍스트 삼중 데이터 구조를 구성하여 교차 모드 학습을 진행하며, MoiveGen 등에서 영감을 받아 영상의 키프레임에서 참조 이미지를 추출하는 방식과 다른, 교차 매칭 데이터 구조를 활용합니다. 이러한 접근 방식이 모델의 교차 모드 정렬 학습에 기여합니다.
- ***Performance Highlights***: Phantom은 현재 상용 솔루션들과 비교하여 주제 일관성(Subject Consistency) 및 명령 수행에서 우수한 성능을 보였습니다. 특히 여러 주제를 포함한 비디오 생성에서 사용자 시험 결과, 상용 솔루션과 유사한 성과를 기록하며, 일부 주제 일관성에서는 더 나은 점수를 받았습니다. 예를 들어, 얼굴 인식 평가에서 높은 ID 유사성 점수를 획득하였습니다.

### [Rethinking Diverse Human Preference Learning through Principal Component Analysis](https://arxiv.org/abs/2502.13131)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.13131.png)

Vote: 32

Authors: Huan Zhang, Jiarui Yao, Chunyuan Deng, Feng Luo, Jingyan Shen, Hanjie Chen, Rui Yang, Hao Sun

- ***What's New***: 이 연구는 인간의 다양한 선호를 이진 비교에서 추출하는 새로운 방법으로, 세분화된 주석 없이도 Principal Component Analysis (PCA)를 사용하여 인간의 선호를 벡터로 표현하는 Decomposed Reward Models (DRMs)를 제안합니다. 이는 전통적인 보상 모델보다 더욱 해석 가능하고 확장 가능한 대안을 제공합니다.
- ***Technical Details***: DRMs는 선호하는 반응과 거부된 반응 간의 임베딩 차이 데이터를 만들고, 그 차이에 PCA를 적용하여 서로 직교하는 기저 벡터를 식별합니다. 이러한 기저 벡터들은 다양한 사용자 요구에 맞게 조합될 수 있으며, 추가적인 학습 없이도 새로운 사용자에 적응할 수 있습니다.
- ***Performance Highlights***: DRMs는 다양한 인간 선호 속성을 효과적으로 추출하며, 테스트 시점에서 사용자 선호에 적응하는 데 있어 기존의 보상 모델을 능가합니다. Gemma-2B-RM을 사용한 DRMs는 RewardBench에서 단일 머리 RM보다 평균 8% 높은 성능을 보였으며, RPR에서도 26% 우수한 결과를 나타냈습니다.

### [Multimodal Mamba: Decoder-only Multimodal State Space Model via Quadratic to Linear Distillation](https://arxiv.org/abs/2502.13145)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.13145.png)

Vote: 26

Authors: Hongyuan Tao, Haoran Yin, Yingyue Li, Tianheng Cheng, Xinggang Wang, Bencheng Liao, Wenyu Liu, Qian Zhang

- ***What's New***: 이 논문은 멀티모달 맘바(mmMamba)라는 새로운 프레임워크를 소개하며, 복잡한 사전 학습 없이 MODA-2(Mamba-2)로 변환하여 선형-복잡도 디코더 전용 비전-언어 모델(Vision-Language Models; VLMs)을 구축하는 법을 제안하고 있습니다. 이는 비전 인코더나 선형-복잡도 언어 모델(Linear-Complexity Language Models; LLMs)에 대한 의존 없이 멀티모달 상태 공간 모델(State Space Models; SSMs)을 구현할 수 있도록 합니다.
- ***Technical Details***: mmMamba는 Transformer 기반 HoVLE에서 파생된 디코더-전용 VLM으로부터 지식을 증류(Distillation)하는 3단계 프로세스를 통해 구축됩니다. 1단계에서는 새로 도입된 SSM 매개변수를 학습하고, 2단계에서는 전체 Mamba-2의 계층 행동을 최적화합니다. 마지막 3단계에서는 최종 출력 로짓에 대한 KL-발산 손실을 사용하여 모델의 멀티모달 이해 역량을 강화합니다. 이 프레임워크는 순수한 Mamba-2 계층을 사용한 mmMamba-linear와 Transformer 및 Mamba-2 계층을 혼합한 mmMamba-hybrid 두 가지 아키텍처 변형을 지원합니다.
- ***Performance Highlights***: mmMamba-linear는 EVE-7B과 같은 기존의 선형 및 사각 복잡도 VLM과 성능을 비교할 수 있으며, mmMamba-hybrid는 이를 더욱 향상시키고 있습니다. 특히, 103K 토큰 길이에서 mmMamba-linear는 HoVLE와 비교해 20.6배의 속도 향상과 75.8%의 GPU 메모리 절약을 달성하며, mmMamba-hybrid는 13.5배의 속도 향상과 60.2%의 메모리 절약을 보여줍니다. 이는 긴 시퀀스를 처리할 때에도 높은 효율을 유지할 수 있음을 시사합니다.

### [SoFar: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation](https://arxiv.org/abs/2502.13143)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.13143.png)

Vote: 26

Authors: Xin Jin, Li Yi, Yufei Ding, Jiayuan Gu, Xialin He, Jiawei He, Zhizheng Zhang, He Wang, Guofan Fan, Runpei Dong, Kaisheng Ma, Xinqiang Yu, Lingyun Xu, Jiazhao Zhang, Zekun Qi, Wenyao Zhang, Jingwen Li, Baoyu Li

- ***What's New***: SOFAR는 최초로 객체의 방향성을 공간적 추론과 로봇 조작에 통합한 시스템입니다. 우리는 자연언어 기반의 객체 방향성(Semantic Orientation) 개념을 도입하여, 로봇이 포즈와 방향 제약을 통해 조작 작업을 효율적으로 수행할 수 있게 했습니다. 이를 지원하기 위해 OrienText300K라는 대규모 데이터셋을 구축했습니다.
- ***Technical Details***: PointSO는 Cross-Modal 3D Transformer 구조를 채택하며, PointNet과 같은 네트워크를 사용하여 3D 포인트 클라우드를 내포합니다. 비전과 언어의 특성을 융합하여 언어로 정의된 방향을 예측합니다. SOFAR 시스템은 RGB-D 이미지를 입력으로 받아 VLM과 통합하여 작업 지향적인 공간적 추론을 수행할 수 있습니다.
- ***Performance Highlights***: 실험 결과 SOFAR는 Open6DOR V2와 같은 대규모 벤치마크에서 최첨단 비전-언어 모델을 능가하는 성과를 보였습니다. 특히 6-DoF 공간적 이해 및 로봇 조작에서 뛰어난 성능을 발휘하여, 실험에서 정부 터보, GPT-4V와 비교했을 때 보다 높은 성능을 보여주었습니다.

### [SafeRoute: Adaptive Model Selection for Efficient and Accurate Safety Guardrails in Large Language Models](https://arxiv.org/abs/2502.12464)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.12464.png)

Vote: 25

Authors: Haebin Seong, Minki Kang, Seanie Lee, Tobias Bocklet, Sung Ju Hwang, Dominik Wagner, Dong Bok Lee, Juho Lee

- ***What's New***: SafeRoute는 대형 언어 모델(Large Language Models; LLMs) 배포에 있어 효율적인 안전 장치(Safety Guardrails)를 위해 적응형 모델 선택 메커니즘을 도입합니다. 이는 작고 효율적인 모델을 이용해 대부분의 쉬운 예제를 처리하고, '어려운' 예제에 대해서만 더 큰 모델을 사용하는 방식을 제안하여 정확성을 유지하면서 계산 비용을 줄이는 접근법입니다.
- ***Technical Details***: SafeRoute는 데이터의 난이도를 기반으로 하여 큰 모델과 작은 모델을 선택하는 이진 라우터입니다. 각 데이터 포인트를 '쉬운' 또는 '어려운'으로 라벨링하여 더 작은 모델이 틀린 예제에 대해서만 큰 안전 장치 모델을 선택적으로 적용합니다. 이를 위해 라우터를 훈련시키고, 효율성과 정확성 간의 트레이드오프를 최적화합니다.
- ***Performance Highlights***: SafeRoute는 다양한 벤치마크 데이터셋에서 작은 모델과 큰 모델 사이의 적응적 모델 선택을 통해 계산 비용과 안전 성능 간의 균형을 크게 향상시킵니다. 본 연구는 테스트 데이터의 5.09%만 큰 모델을 사용하면서도 F1 점수를 13% 향상시킨 결과를 보여줍니다. 이는 관련 기준선 대비 더 향상된 결과를 보이며, 특히 Out-of-Distribution(OOD) 시나리오에서도 뛰어난 성능을 나타냅니다.

### [Magma: A Foundation Model for Multimodal AI Agents](https://arxiv.org/abs/2502.13130)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.13130.png)

Vote: 25

Authors: Jianfeng Gao, Yuquan Deng, Mu Cai, Joel Jang, Reuben Tan, Yongyuan Liang, Qianhui Wu, Lars Liden, Yu Gu, Seonghyeon Ye, Baolin Peng, Jianwei Yang, Ruijie Zheng

- ***What's New***: Magma는 멀티모달 AI 에이전트의 새로운 기반 모델로, 디지털 및 물리적 세계에서의 에이전트 작업(Agentic Tasks)을 수행하는 능력을 지니고 있습니다. 이 모델은 시각-언어(VL) 이해 능력을 유지하는 동시에, 시각-공간적 세계에서 계획하고 행동할 수 있는 능력을 갖추고 있습니다.
- ***Technical Details***: Magma 모델은 이미지, 비디오, 로봇 데이터 등 다양한 이질적 데이터셋에 대해 사전 학습되었습니다. 특히, SoM(Set-of-Mark)과 ToM(Trace-of-Mark) 기법을 통해 행동 기반 시각 객체를 라벨링하고, 이 라벨들로부터 행동 계획을 수립할 수 있도록 하였습니다. 이러한 라벨링 및 계획 과정은 모델의 시공간 지능 획득에 크게 기여합니다.
- ***Performance Highlights***: Magma는 사용자 인터페이스(UI) 네비게이션 및 로봇 조작 작업에서 새로운 최첨단(State-of-the-Art) 결과를 달성했으며, 이전의 특정 작업을 위해 설계된 모델보다 우수한 성능을 보였습니다. 또한 이미지 및 비디오 관련 멀티모달 태스크에서도, 훨씬 더 많은 데이터셋에서 학습된 인기 있는 대형 멀티모달 모델들과 비교해 경쟁력 있는 성능을 나타냅니다.

### [You Do Not Fully Utilize Transformer's Representation Capacity](https://arxiv.org/abs/2502.09245)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.09245.png)

Vote: 24

Authors: Daniil Gavrilov, Yaroslav Aksenov, Nikita Balagansky, Gleb Gerasimov, Viacheslav Sinii

- ***What's New***: 이 연구에서는 기존의 Transformer가 즉각적으로 이전 레이어에서만 표현을 사용하는 한계를 해결하는 Layer-Integrated Memory (LIMe)를 제안합니다. LIMe는 모든 이전 레이어의 숨겨진 상태를 활용하여 표현 용량을 확장하며, 다양한 구조와 검색 메커니즘을 통한 실험에서 일관된 성능 향상을 보여줍니다.
- ***Technical Details***: LIMe는 마스크 된 다중-헤드 자기-주의(Masked Multi-Head Self-Attention)의 확장으로, 이전 모든 레이어의 표현을 검색하고 통합할 수 있도록 합니다. LIMe는 모든 이전 레이어 출력을 사용하여 키와 값을 생성하며, 특정 라우터를 통해 이러한 혼합을 형성하여 효율적으로 층간 정보를 블렌딩합니다.
- ***Performance Highlights***: LIMe는 표준 Transformer 및 최신 수정을 지속적으로 능가하며, 학습 엔트로피를 감소시키고 표현의 다양성을 향상시키며 대표적인 벤치마크에서 높은 성능을 기록합니다. 특히, LIMe가 적용된 모델은 각 층별 '의미적 회로'를 학습하여 표현 붕괴를 효과적으로 방지하고 더 나은 레이어간 정보 통합을 이끌어 냅니다.

### [FLAG-Trader: Fusion LLM-Agent with Gradient-based Reinforcement Learning for Financial Trading](https://arxiv.org/abs/2502.11433)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11433.png)

Vote: 23

Authors: Jimin Huang, Mingquan Lin, Yangyang Yu, Guojun Xiong, Zhiyang Deng, Haohang Li, Xiao-Yang Liu, Yupeng Cao, Xueqing Peng, Kaleb E Smith, Sophia Ananiadou, Keyi Wang, Qianqian Xie

- ***What's New***: 이번 연구는 금융 거래에 LLM(대형 언어 모델; Large Language Model)을 RL(강화 학습; Reinforcement Learning)과 결합하여 활용하는 FLAG-Trader라는 새로운 프레임워크를 제공합니다. 본 프레임워크는 RL 최적화를 통해 LLM을 세분화 거래 정책 네트워크로 활용, 작은 규모의 LLM도 대규모의 모델 성능을 능가하도록 설계되었습니다.
- ***Technical Details***: FLAG-Trader는 금융 데이터를 사용하여 부분적으로 미세조정된 LLM을 정책 네트워크로 삼고, 포괄적인 강화 학습 프레임워크로 최적화를 진행합니다. 이 프로세스는 훈련 가능한 상위 계층을 통해 금융 도메인 적응을 실시하며, 체계적인 강화학습 방법을 통해 초기 프롬프트의 영향을 줄여가며 안정적인 정책으로 수렴합니다.
- ***Performance Highlights***: FLAG-Trader는 미세 조정된 작은 규모의 LLM(135M 파라미터)이 대형 독점 모델을 능가하는 성과를 보였습니다. 여러 금융 거래 시나리오에서 기존의 LLM 에이전틱 프레임워크 및 전통적인 RL 기반 거래 에이전트를 능가하며, 수익 및 샤프 비율에서 우수한 결과를 달성했습니다.

### [RealSyn: An Effective and Scalable Multimodal Interleaved Document Transformation Paradigm](https://arxiv.org/abs/2502.12513)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.12513.png)

Vote: 14

Authors: Ziyong Feng, Chaoyi Zhang, Kaicheng Yang, Dongnan Liu, Yin Xie, Weidong Cai, Jiankang Deng, Xiang An, Tiancheng Gu

- ***What's New***: 이 논문은 대규모 멀티모달 문서에서 짝지어지지 않은 데이터를 활용하여 비전-언어 표현 학습(Vision-Language Representation Learning)을 개선하기 위한 새로운 접근법, RealSyn을 제안합니다. 이를 통해 현실적인 텍스트와 합성 텍스트를 결합한 대규모 데이터셋을 구축하였습니다.
- ***Technical Details***: RealSyn은 실제 및 합성 텍스트를 결합하여 15M, 30M, 100M의 세 가지 규모로 제공되는 데이터셋을 통해 학습됩니다. 데이터셋 구축을 위해 실제 이미지와 텍스트를 추출하는 Real-World Data Extraction 파이프라인과, 각 이미지에 의미론적으로 관련된 텍스트를 효율적으로 연관시키기 위한 계층적 검색 방법(Hierarchical Retrieval Method)을 설계했습니다. 또한, 합성 텍스트 생성을 위한 이미지 의미론 강화 생성 모듈을 제안합니다.
- ***Performance Highlights***: RealSyn으로 사전 학습된 모델은 다수의 다운스트림 작업에서 최첨단 성능(State-of-the-art Performance)을 달성했습니다. 특히, Flickr30k 및 MSCOCO 데이터셋에서의 제로샷 이미지-텍스트 검색 결과가 크게 개선되었음을 보여주며, 이는 다양한 시나리오에서 탁월한 확장성을 발휘합니다.

### [OctoTools: An Agentic Framework with Extensible Tools for Complex Reasoning](https://arxiv.org/abs/2502.11271)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11271.png)

Vote: 10

Authors: Rahul Thapa, Joseph Boen, James Zou, Bowen Chen, Pan Lu, Sheng Liu

- ***What's New***: OctoTools는 복잡한 추론 문제를 해결하기 위한 도구 사용 에이전트 프레임워크로, 훈련 없이 사용자 친화적이고 확장 가능한 오픈 소스입니다. 새로운 표준화된 툴카드(tool cards)를 도입하여 다양한 도구의 기능성을 캡슐화하고, 높은 수준 및 낮은 수준의 계획을 통제하는 계획자(planner)와 도구 사용을 수행하는 실행자(executor)를 추가하여 다양한 도메인에서 일반적인 응용 가능성을 입증하였습니다.
- ***Technical Details***: OctoTools는 16개의 다양한 작업(MathVista, MMLU-Pro, MedQA, GAIA-Text 포함)에서 효율적으로 동작하며, GPT-4o를 기준으로 평균 정확도가 9.3% 향상되었습니다. Task-specific 도구 집합 최적화 알고리즘을 통해 특정 작업에 유리한 도구 세트를 학습합니다. 새로운 도구는 훈련 없이 쉽게 통합, 교체 또는 확장 가능합니다.
- ***Performance Highlights***: OctoTools는 동일한 도구 세트를 사용할 때 AutoGen, GPT-Functions 및 LangChain과 비교하여 최대 10.6% 더 성능이 우수하며, 복잡한 추론과 도구 사용에서의 장점을 입증했습니다. 세부 분석을 통해 복잡한 문제 해결에서의 멀티스텝 계획과 전문 도구 사용의 효과를 분리하여 이해할 수 있었습니다.

### [PAFT: Prompt-Agnostic Fine-Tuning](https://arxiv.org/abs/2502.12859)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.12859.png)

Vote: 10

Authors: Yao Shu, Mingwen Ou, Fei Richard Yu, Ying Tiffany He, Chenxing Wei

- ***What's New***: PAFT(Prompt-Agnostic Fine-Tuning)는 대형 언어 모델(Large Language Models; LLMs)의 프롬프트 변동성에 대한 강건성을 향상시키기 위한 새로운 방법입니다. PAFT는 프롬프트와 독립적으로 작업의 기본 원리를 학습하도록 모델을 유도하며, 훈련 중 동적으로 프롬프트를 조정합니다.
- ***Technical Details***: PAFT는 두 단계로 작동합니다. 첫째, 다양한 의미를 포착하는 합성 프롬프트 후보를 생성합니다. 둘째, 이러한 프롬프트 집합에서 무작위로 샘플링하여 동적 입력을 만들어 훈련을 진행합니다. 이로 인해 모델은 특정 프롬프트 패턴에 과적합하지 않고 다양한 프롬프트에 대해 일반화할 수 있습니다.
- ***Performance Highlights***: PAFT를 적용한 모델은 다양한 테스트 프롬프트에 대해 강건한 성능을 보였으며, 댓글 검색과 같은 실제 응용에 더 높은 신뢰성과 효율성을 제공합니다. PAFT는 기준 방법에 비해 평균 정확도와 표준 편차 모두에서 뛰어난 성능을 보여주며, 테스트 프롬프트에서는 90% 이상의 정답률을 기록했습니다.

### [MUDDFormer: Breaking Residual Bottlenecks in Transformers via Multiway Dynamic Dense Connections](https://arxiv.org/abs/2502.12170)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.12170.png)

Vote: 10

Authors: Da Xiao, Shengping Li, Qingye Meng, Xingyuan Yuan

- ***What's New***: MUDDFormer는 새로운 MUltiway Dynamic Dense (MUDD) 연결 방식을 통해 Transformer의 잔여 연결(Residual Connections)에서 발생하는 병목 현상을 해결하고, 계층 간 정보 흐름을 강화하는 모델입니다. 기존의 밀집 연결(Dense Connection) 방식과는 달리, MUDD 연결은 시퀀스 위치별 및 Transformer 블록의 각 입력 스트림(쿼리, 키, 값, 잔여)에 따라 동적으로 연결 가중치를 생성합니다.
- ***Technical Details***: MUDD 연결은 Transformer 아키텍처 어디에도 매끄럽게 통합될 수 있으며, 모델의 각 계층 이후에 깊이 방향으로 집계 모듈(Depth-wise Aggregate Modules)을 두어 현재 계층의 입력 스트림을 여러 개의 입력 스트림으로 결합합니다. 이를 통해 기존의 잔여 스트림을 넘어서서 계층 간 통신 대역폭을 크게 확장시킵니다. 또한, DA 모듈 내에서 연산을 통해 동적 연결을 구현하고, 이를 통해 효율성과 확장성을 높입니다.
- ***Performance Highlights***: MUDDFormer는 다양한 모델 아키텍처와 규모에서 기존의 Transformer를 크게 능가하였습니다. 예를 들어, MUDDPythia-2.8B는 예비 훈련 퍼플렉서티(pre-training perplexity)와 다운스트림 작업에서 Pythia-6.9B와 유사한 성능을 보여주며, 0.23%의 파라미터와 0.4%의 계산만을 추가하였습니다. 또한, 모델 성능을 1.8~2.4배 높인 Transformer의 성능을 달성했습니다.

### [Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities?](https://arxiv.org/abs/2502.12215)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.12215.png)

Vote: 9

Authors: Yunhua Zhou, Zhiyuan Zeng, Qinyuan Cheng, Zhangyue Yin, Xipeng Qiu

- ***What's New***: 이 연구는 오픈AI의 o1 시리즈 등을 포함한 대형 언어 모델(LLMs)의 테스트 시 계산 리소스 스케일링 기능의 실제 성능을 재평가합니다. 기존에는 연산 리소스를 늘리면 정확도가 증가할 것이라 생각되었으나, 실제로는 정답 솔루션보다 오답 솔루션이 더 긴 경우가 많다는 역설적인 결과를 발견하였습니다.
- ***Technical Details***: 이 연구는 QwQ, Deepseek-R1(R1), LIMO 등의 오픈소스 후속 모델들을 평가하며, Chain-of-Thought(CoT) 길이와 정확도 간의 상관관계를 체계적으로 조사합니다. CoT 길이를 늘리면 모델의 정확도가 개선된다는 전통적인 관점을 도전하고 손상된 성능으로 이어지는 자기 수정(self-revision) 능력의 부재가 주된 문제임을 밝혀냈습니다. 이를 통해 'Shortest Majority Vote' 방법을 제안하여 병렬 스케일링을 CoT 길이 특성과 결합합니다.
- ***Performance Highlights***: 병렬 스케일링은 QwQ 및 R1 모델에 대해 더 나은 커버리지와 확장성을 제공하는 것으로 나타났습니다. 특히, 병렬 스케일링은 추가 솔루션을 샘플링하여 최적의 답을 선택하는 것을 통해 수준 높은 성능을 보였으며, 'Shortest Majority Vote'는 전통적인 다수결 방식보다 유의미하게 높은 테스트 시 스케일링 성능을 보여줍니다.

### [Text2World: Benchmarking Large Language Models for Symbolic World Model Generation](https://arxiv.org/abs/2502.13092)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.13092.png)

Vote: 8

Authors: Tianxing Chen, Mengkang Hu, Ming Li, Hongyuan Zhang, Wenqi Shao, Yude Zou, Qiguang Chen, Yuheng Lei, Ping Luo

- ***What's New***: 이 논문에서는 대형 언어 모델(Large Language Models; LLMs)을 텍스트 설명으로부터 기호적 세계 모델(Symbolic World Models)을 생성하는 데 사용하는 새로운 벤치마크인 TEXT2WORLD를 소개합니다. 이 벤치마크는 계획 도메인 정의 언어(Planning Domain Definition Language; PDDL)을 기반으로 하며, 수백 개의 다양한 도메인을 포함하여 LLM의 세계 모델링 능력을 평가합니다.
- ***Technical Details***: TEXT2WORLD 벤치마크는 구조적 유사성 및 컴포넌트별 F1 점수와 같은 고급 기법을 사용하여 LLM이 생성한 세계 모델의 정확성을 강력하게 평가합니다. LLM이 자연어 설명(Natural Language Description)에서 행동지향 및 제약을 암시해야 하는 과제를 제공하여 평가합니다.
- ***Performance Highlights***: 실험 결과, 대규모 강화 학습으로 훈련된 추론 모델이 다른 모델들보다 우수한 성능을 보여줬습니다. 그러나 가장 성능이 우수한 모델도 여전히 제한적인 세계 모델링 능력을 보였습니다. LLM의 오류 수정 능력을 활용한 실험에서는, 오류 수정 시도 횟수를 늘릴수록 성능이 향상됨을 발견했습니다.

### [HealthGPT: A Medical Large Vision-Language Model for Unifying Comprehension and Generation via Heterogeneous Knowledge Adaptation](https://arxiv.org/abs/2502.09838)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.09838.png)

Vote: 7

Authors: Wenqiao Zhang, Binhe Yu, Haoyuan Li, Tianwei Lin, Hui Lin, Wanggui He, Yuqian Yuan, Xiaohui Song, Hao Jiang, Beng Chin Ooi, Jun Xiao, Yueting Zhuang, Siliang Tang, Mengze Li, Sijing Li

- ***What's New***: HealthGPT는 의료 분야의 복합적인 시각-언어 이해(comprehension) 및 생성(generation) 작업을 통합하는 최초의 의료 대형 비전-언어 모델(Medical Large Vision-Language Model; Med-LVLM)입니다. 이 모델은 다양한 영상 모달리티의 종합적인 이해와 생성을 가능하게 하여 복잡한 의료 시나리오에서의 활용 가능성을 높입니다.
- ***Technical Details***: HealthGPT는 이기종 지식 적응(Heterogeneous Knowledge Adaptation) 방식을 통해 사전 훈련된 대형 언어 모델(LLMs)을 점진적으로 적응시키고, 이기종 저랭크 적응(Heterogeneous Low-Rank Adaptation; H-LoRA) 기법과 계층적 시각 지각(hierarchical visual perception)을 결합하여 의료 분야의 시각적 이해와 생성을 통합합니다. 이 모델은 시각적 세부 사항을 Vision Transformer(ViT)를 사용하여 압축하며, 세 가지 단계 학습 전략(Three-stage Learning Strategy; TLS)을 통해 다양한 다운스트림 과제에 빠르게 적응할 수 있습니다.
- ***Performance Highlights***: HealthGPT는 다양한 의료 시각-언어 과제에서 뛰어난 성능을 보여주었습니다. 예를 들어, CT에서 MRI로의 변환 작업에서 SSIM 79.38을 기록하며, 기존의 변화 모델들보다 우수한 성과를 보였습니다. 또한, VQA-RAD 등의 의료 질문 응답 과제에서도 높은 성과를 기록하며, Unified Model들과 비교했을 때도 우수한 성능을 입증했습니다.

### [Eager Updates For Overlapped Communication and Computation in DiLoCo](https://arxiv.org/abs/2502.12996)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.12996.png)

Vote: 7

Authors: Yanislav Donchev, Satyen Kale, Arthur Douillard

- ***What's New***: DiLoCo에서의 외부 최적화 단계 동안 발생하는 통신과 성능 저하 문제를 해결하기 위해, 'eager updates' 기법을 개발하여 학습 중 통신과 계산을 중첩시키는 방법을 제안합니다. 이를 통해 예전보다 통신 대역폭이 낮은 환경에서 DiLoCo와 같은 대규모 분산 학습에서도 성능 저하를 최소화할 수 있습니다.
- ***Technical Details***: 제안된 기법인 'eager updates'는 외부 최적화 단계 동안 발생하는 외부 그래디언트(Outer Gradients)와 내적 최적화 단계 작업을 병행합니다. 이때 외부 그래디언트를 각각의 작업자(Workers)가 자체적으로 계산하여 통신이 완료되기 전에도 로컬 외부 그래디언트를 활용하는 방식으로 최적화를 진행합니다. 이를 통해 통신이 완료되기 전에 이미 새로운 최적화 단계를 시작할 수 있게 되어, 통신 지연에 의한 성능 저하를 최소화합니다.
- ***Performance Highlights***: eager updates 기법을 통해 계산 활용도(Compute Utilization)가 기존 DiLoCo 대비 크게 향상되었으며, 통신 대역폭 요구사항도 크게 줄어듭니다. 예를 들어, 1-outer-step eager updates 기법은 데이터 병렬 방식에 비해 1,177배 더 적은 대역폭을 요구하며, 이는 훈련에 필요한 대역폭을 크게 줄이는 결과를 보여줍니다. 이를 통해 제한된 대역폭 환경에서의 훈련 효율성을 강화할 수 있습니다.

### [HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading](https://arxiv.org/abs/2502.12574)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.12574.png)

Vote: 6

Authors: Jiawei Zhao, Wen Xiao, Bo Yuan, Hanshi Sun, Beidi Chen, Cheng Luo, Zefan Cai, Anima Anandkumar, Jinqi Xiao, Junjie Hu

- ***What's New***: HEADINFER는 대형 언어 모델(LLMs)의 추론 시 메모리 효율성을 높이는 새로운 프레임워크입니다. 이 연구는 LLM 추론 중의 메모리 사용량을 줄이기 위해 개별 'attention heads' 수준에서 KV 캐시를 CPU RAM으로 오프로드하는 전략을 도입합니다.
- ***Technical Details***: HEADINFER는 'attention heads'의 독립성을 활용하여 주어진 시간에 단 하나의 머리의 KV 캐시만 GPU에 저장하고 나머지는 CPU로 오프로드합니다. 또한, 어댑티브 헤드 그룹핑(adaptive heads grouping), 청크드 프리필(chunked prefill), 비동기 데이터 전송(asynchronous data transfer)을 통합하여, 증가하는 문맥 길이에 맞춰 온전히 GPU 메모리 공간을 효율적으로 사용합니다.
- ***Performance Highlights***: RTX 4090과 같은 소비자 GPU에서 4백만 토큰 추론을 구현하며, GPU 메모리 사용량을 BF16 기준과 비교하여 92% 줄였습니다. Llama-3-8B 모델의 GPU 메모리 풋프린트를 128GB에서 1GB로 대폭 줄였고, 총 GPU 메모리 사용량은 207GB에서 17GB로 감소시켰습니다.

### [Atom of Thoughts for Markov LLM Test-Time Scaling](https://arxiv.org/abs/2502.12018)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.12018.png)

Vote: 6

Authors: Fengwei Teng, Jiayi Zhang, Chenglin Wu, Zhaoyang Yu, Quan Shi, Yuyu Luo

- ***What's New***: 이 논문에서는 대형 언어 모델(LLMs)의 추론 능력을 확장하기 위한 새로운 프레임워크인 Atom of Thoughts (AOT)를 제안합니다. AOT는 복잡한 추론 과정을 Markov 프로세스를 통해 독립적인 원자적 질문으로 변환하여 정보의 히스토리적 의존성을 제거합니다. 이렇게 함으로써, 모형은 현재의 질문 상태에 집중할 수 있으며, AOT는 독립적인 프레임워크로서뿐만 아니라 기존의 테스트 시간 스케일링 방법을 강화하는 플러그인으로도 작동할 수 있습니다.
- ***Technical Details***: AOT는 현재 질문을 의존성 기반의 유향 비순환 그래프(DAG)로 분해한 후, 부분 질문을 새로운 원자적 상태로 축소하는 이중 단계 전이 메커니즘을 사용합니다. 이 과정은 직접적으로 해결 가능한 원자적 질문에 도달할 때까지 계속됩니다. 수많은 벤치마크 실험에서 AOT는 독립형 프레임워크 및 플러그인으로서 모두 효과성을 입증하였으며, 특히 HotpotQA 데이터셋에서 gpt-4o-mini와 결합되어 큰 성능 향상을 보여주었습니다.
- ***Performance Highlights***: 실험 결과 AOT는 다양한 이유 기반 작업에서 일관된 성능 향상을 보여줍니다. 예를 들어, 수학 문제 해결에 있어 AOT는 MATH 데이터셋에서 84.9%의 정확도를 기록하였고, 다중 홉 질문 응답에서는 HotpotQA에서 F1 점수 80.6%를 달성하며 기존 모델들을 능가했습니다. 이는 특히, 추론이 긴 문맥 시나리오에서 더욱 효과적임을 보여줍니다.

### [Crowd Comparative Reasoning: Unlocking Comprehensive Evaluations for LLM-as-a-Judge](https://arxiv.org/abs/2502.12501)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.12501.png)

Vote: 5

Authors: Yuxin Jiang, Chuhan Wu, Lifeng Shang, Qiyuan Zhang, Chen Ma, Yufei Wang, Liangyou Li, Fuyuan Lyu, Ruiming Tang, Xin Jiang, Yasheng Wang

- ***What's New***: Crowd Comparative Reasoning (CCR)는 LLM-as-a-Judge의 한계를 극복하기 위해 제안된 새로운 방법입니다. 이 접근법은 기존의 다수결 투표나 기준 확장 방식과 달리, 비교 평가를 통해 더 깊고 포괄적인 판단을 이끌어냅니다.
- ***Technical Details***: CCR은 Crowd Response와 Crowd Judgment를 생성하고 이를 선택, 처리하여 Context-augmented Inference를 수행하는 세 가지 핵심 단계로 구성됩니다. 여러 LLMs을 활용해 다양한 응답을 생성하고, 각 후보 응답을 이들과 비교하여 다양한 판단을 도출합니다. 최종 판단은 이러한 Crowd Judgment를 바탕으로 이루어집니다.
- ***Performance Highlights***: CCR은 5개의 평가 기준에서 평균 6.7%의 정확도 향상을 보이며, 기존 방법들에 비해 뛰어난 성능을 입증했습니다. RewardBench, HelpSteer2, MTBench Human, JudgeBench, EvalBias 등의 벤치마크에서 일관된 성능 향상을 나타내어 더 높은 품질의 CoT를 생성합니다.

### [Flow-of-Options: Diversified and Improved LLM Reasoning by Thinking Through Options](https://arxiv.org/abs/2502.12929)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.12929.png)

Vote: 4

Authors: Ian Trase, Lakshmi Nair, Mark Kim

- ***What's New***: 이 논문에서는 대규모 언어 모델(LLMs)의 내재적 편향을 해결하기 위해 설계된 새로운 추론 접근법인 Flow-of-Options(FoO)을 제시합니다. FoO는 다양한 가능성을 체계적으로 탐색하여 훈련을 통해 LLM의 편향을 강화하지 않고, 대안적 접근을 제시함으로써 LLM의 추론 능력을 강화합니다.
- ***Technical Details***: Flow-of-Options(FoO)는 작업의 각 단계를 실행하기 전에 실행 가능한 옵션들을 네트워크 데이터 구조로 열거하여, 다양한 옵션을 탐색하도록 합니다. 이를 통해 AutoML(자동화된 머신 러닝) 시스템 내에서 응집된 가상 에이전트 시스템을 구축하였으며, 이를 통해 분류, 회귀를 넘어서 강화학습, 이미지 생성 등의 다양한 작업에도 적용 가능함을 보였습니다.
- ***Performance Highlights***: 제안된 FoO 기반 프레임워크는 기존의 최첨단 기준을 능가하는 성능을 보이며, 일반적인 데이터 과학 작업에서 38.2%에서 69.2%의 향상, 치료 화학 작업에서는 37.4%에서 47.9%의 성능 향상을 달성했습니다. 또한, 작업당 전체 실행 비용이 1달러 이하로 비용 절감 효과를 확인할 수 있습니다.

### [Injecting Domain-Specific Knowledge into Large Language Models: A Comprehensive Survey](https://arxiv.org/abs/2502.10708)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.10708.png)

Vote: 3

Authors: Rui Yan, Miao Fang, Xiuying Chen, Bin Yan, Zirui Song, Mingzhe Li, Yuhan Liu

- ***What's New***: 이 논문은 도메인 특화 지식을 대형 언어 모델(Large Language Models; LLMs)에 주입하는 다양한 방법론을 종합적으로 검토한 서베이입니다. 특히, 동적 지식 주입(Dynamic Knowledge Injection), 정적 지식 임베딩(Static Knowledge Embedding), 모듈형 어댑터(Modular Adapters), 프롬프트 최적화(Prompt Optimization) 등 네 가지 주요 접근 방식을 정리하여 LLMs에서 도메인 전문 지식을 통합하는 방법을 제시합니다.
- ***Technical Details***: 논문에서는 각 접근 방식을 설명하고, 이들이 어떻게 LLMs를 특화된 작업에 적합하게 만드는지 논의합니다. 예를 들어, 동적 지식 주입은 실행 시 외부 지식베이스에서 정보를 검색해 결합하고, 정적 지식 임베딩은 전체 또는 부분 미세조정을 통해 모델의 파라미터에 도메인 지식을 포함시킵니다. 모듈형 어댑터는 원래 모델의 파라미터를 유지하면서 외부 지식을 저장하는 소형 모듈을 사용하며, 프롬프트 최적화는 모델 아키텍처 변경 없이 조심스럽게 설계된 프롬프트로 기존 지식을 활용합니다.
- ***Performance Highlights***: 분야별 도메인 특화 LLMs가 일반 LLMs에 비해 특수 작업에서 뛰어난 성과를 나타냅니다. 예를 들어, PMC-LLaMA와 같은 모델은 MedQA 데이터셋에서 LLaMA2 모델보다 10점 이상 높은 성능을 보이며, 이는 도메인 지식 주입이 특정 작업에서의 성능을 크게 향상시킨다는 것을 보여줍니다. 이러한 결과는 도메인 특화 LLMs의 중요성을 강조하며, 이들이 전문 분야에서의 과제를 해결하는 데 필요한 성능을 제공할 수 있음을 제시합니다.

### [Multilingual Encoder Knows more than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages](https://arxiv.org/abs/2502.10852)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.10852.png)

Vote: 2

Authors: XU Han, Zeli Su, Jianing Liu, Guixian Xu, Ziyin Zhang, Ting Zhang, Yushuang Dong

- ***What's New***: 이번 연구는 XLM-SWCM이라는 새로운 모델을 제안하며, 이는 특히 자원이 극히 제한적인 언어들, 예를 들어 티베트어, 위구르어, 카자흐어, 몽골어에 대해 다중언어 인코더를 텍스트 생성에 효율적으로 적응시키기 위한 새로운 프레임워크를 소개합니다. 인코더와 디코더간의 가중치(Shared Weights)를 공유함으로써, 모델이 인코더의 학습된 의미 공간을 활용하여 저자원 언어에서 효율적 학습 및 일반화 능력을 갖추도록 합니다.
- ***Technical Details***: XLM-SWCM 모델은 XLM-R 기반의 기존 인코더(CINO)를 개선하여 중국 소수 언어 전용 연속 사전학습 모델로 사용하며, 이 인코더의 가중치를 디코더 층의 초기화에 활용합니다. 디코더는 두 가지 유형인 NormalDecoderLayer와 CustomDecoderLayer로 구성되며, CustomDecoderLayer는 인코더의 사전학습 가중치를 상속하여 생성 작업에 효율적으로 적응합니다. 또한, 데이터 학습 불균형 문제를 해결하기 위해 균형 잡힌 샘플링 전략을 적용합니다.
- ***Performance Highlights***: 실험 결과 XLM-SWCM 모델은 mBART와 같은 기존 베이스라인을 능가하며, 특히 텍스트 요약에서 198.8%, 독해 영역에서 107.6% 성능 개선을 보여줍니다. 또한 훨씬 더 큰 규모의 MC2-LLaMA-13B 모델보다 우수한 성과를 기록하며, 극히 제한적인 환경에서도 탁월한 성능을 발휘합니다. 크로스링구얼 전이 실험에서 XLM-SWCM은 ECM-LLaMA-13B를 포함한 대부분의 베이스라인들보다 뛰어난 적응력을 보여줍니다.

### [Perovskite-LLM: Knowledge-Enhanced Large Language Models for Perovskite Solar Cell Research](https://arxiv.org/abs/2502.12669)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.12669.png)

Vote: 2

Authors: Chang Yan, Longhan Zhang, Huajie You, Yongqi Zhang, Penglei Sun, Peijie Dong, Xiang Liu, Tong-yi Zhang, Shuyan Chen, Xiaowen Chu

- ***What's New***: Perovskite-LLM은 페로브스카이트 태양 전지 연구를 위한 특화된 대형 언어 모델(Large Language Models; LLMs) 시스템으로, 도메인 지식을 체계화하고 연구자에게 지능형 도움을 제공합니다. 특히, 추가된 Perovskite-KG, Perovskite-Chat-LLM, Perovskite-Reasoning-LLM은 기존 모델에 비해 도메인 내 지식 검색과 과학적 추론 작업에서 뛰어난 성능을 발휘합니다.
- ***Technical Details***: Perovskite-KG는 1,517개의 연구 논문으로부터 23,789개의 엔티티와 22,272개의 관계를 포함한 도메인 특정 지식 그래프(Knowledge Graph; KG)로 구성됩니다. 또한, 다중 에이전트 프레임워크를 통한 고품질의 교육 데이터를 생성하여 Perovskite-Chat 데이터셋에는 55,101개 문항의 질문-답변 쌍이 포함되어 있으며, Perovskite-Reasoning 데이터셋에는 2,217개의 문제들이 포괄됩니다. Perovskite-Chat-LLM은 도메인 특정 지식 지원을, Perovskite-Reasoning-LLM은 과학적 추론을 다루기 위한 모델로 설계되었습니다.
- ***Performance Highlights***: Perovskite-LLM 시스템은 도메인 내 지식 검색과 과학적 문제 해결에서 기존 모델들과 비교하여 유리한 성능 향상을 보여줍니다. Perovskite-Chat-LLM은 특히 도메인 특정 작업에서 state-of-the-art 성능을 기록하였으며, Perovskite-Reasoning-LLM은 상당히 적은 훈련 예제만을 사용하고도 과학적 추론 벤치마크에서 경쟁력 있는 성과를 보여 줍니다.

### [Pre-training Auto-regressive Robotic Models with 4D Representations](https://arxiv.org/abs/2502.13142)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.13142.png)

Vote: 2

Authors: Roei Herzig, Giscard Biamby, Junyi Zhang, Trevor Darrell, Dantong Niu, Haoru Xue, Yuvan Sharma, Ziteng Ji

- ***What's New***: 이 논문은 모니큘러 깊이 추정을 통해 2D 표현을 3D 공간으로 끌어올리고 이 3D 포인트를 시간을 기준으로 추적하여 인간 비디오 데이터로부터 학습된 새로운 4D 표현(4D Representations)을 활용한 자율 회귀 로봇 모델(ARM4R)을 소개합니다. 이러한 4D 표현은 로봇 상태 표현과 공유된 기하학적 구조를 유지하며, 인간 비디오 데이터로부터 로봇 제어로의 효율적인 전이 학습을 가능하게 합니다.
- ***Technical Details***: ARM4R은 3개의 트레이닝 단계를 거쳐 학습됩니다. 첫 번째 단계는 인간 비디오 데이터로부터 3D 포인트 추적을 통해 일반화된 저수준 표현을 학습하며, 두 번째 단계는 보다 적은 양의 로봇 데이터로 3D 포인트 추적을 미세 조정합니다. 마지막으로 세 번째 단계에서는 로봇 제어를 위한 미세 조정을 통해 실제 로봇 제어를 수행할 수 있도록 합니다. 이 모델의 아키텍처는 주의 풀링(Attention Pooling) 레이어와 인과 변환기(Causal Transformer)를 통해 입력 텍스트, 이미지, 포인트, 로봇 상태를 동일한 잠재 공간으로 변환하여 다음 상태를 예측합니다.
- ***Performance Highlights***: ARM4R은 시뮬레이션 및 실제 로봇 환경에서 다양한 로봇 작업에 대해 기존 방법보다 일관되게 우수한 성능을 보여줍니다. RLBench 시뮬레이션 환경의 12개의 작업과 7 DoF Kinova Gen3 및 Franka Emika Panda 로봇을 사용한 실제 실험에서 ATM, OpenVLA 등의 기존 방법보다 높은 성공률을 기록하였습니다. 이는 로봇 제어에 있어서 저수준 4D 표현의 강점을 시사합니다.

### [FinMTEB: Finance Massive Text Embedding Benchmark](https://arxiv.org/abs/2502.10990)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.10990.png)

Vote: 1

Authors: Yixuan Tang, Yi Yang

- ***What's New***: 이 논문에서는 금융 도메인에 특화된 임베딩 모델을 평가하기 위해 설계된 새로운 평가 프레임워크인 FinMTEB를 소개합니다. FinMTEB는 7개의 서로 다른 작업에 걸쳐 64개의 금융 도메인 특정 임베딩 데이터셋을 포함하고 있으며, 이러한 데이터셋은 중국어와 영어로 된 금융 뉴스, 기업 연차 보고서, ESG 보고서, 규제 문서 및 수익 전화 회의 전사본 등 다양한 텍스트 유형을 다룹니다.
- ***Technical Details***: FinMTEB는 e5-Mistral-7B-Instruct 모델을 금융에 적합하게 조정한 Fin-E5 모델을 개발하여 훈련했습니다. 이 모델은 다양한 금융 임베딩 작업을 위해 훈련된 페르소나 기반 데이터를 사용하며, 15개의 임베딩 모델을 광범위하게 평가하여 금융 도메인의 작업에 대한 성능을 분석했습니다. 특히 Bag-of-Words (BoW) 접근법이 복잡한 금융 텍스트의 의미론적 유사성(semantic similarity) 작업에서 밀집 임베딩(dense embeddings)을 초과 성능을 보였고, 이는 현재 밀집 임베딩 기술의 한계를 시사합니다.
- ***Performance Highlights***: 실험 결과 대역 목적의 벤치마크에서의 성능은 금융 도메인 작업과의 상관관계가 제한적이며, 도메인에 적응된 모델이 대역 목적의 모델을 일관되게 능가했습니다. 더욱이 간단한 Bag-of-Words 접근법이 금융 의미론적 유사성(STS) 작업에서 복잡한 밀집 임베딩을 능가하는 것으로 나타나, 현재 밀집 임베딩 기법에 한계가 있음을 보여줍니다.

### [YOLOv12: Attention-Centric Real-Time Object Detectors](https://arxiv.org/abs/2502.12524)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.12524.png)

Vote: 1

Authors: Qixiang Ye, David Doermann, Yunjie Tian

- ***What's New***: 이번 논문은 YOLOv12를 소개합니다. 이 모델은 전통적으로 실시간 요구사항에 비효율적이라고 여겨졌던 attention 중심의 설계를 YOLO 프레임워크에 성공적으로 도입하여 최신의 지연-정확도(accuracy) 균형을 달성했습니다. 주목할만한 것은 이 모델이 추가적인 사전 학습 없이 높은 탐지 정확도와 빠른 추론 속도를 이룩했다는 것입니다.
- ***Technical Details***: YOLOv12는 새로운 영역 주의 메커니즘(Area Attention)을 도입하여 계산 복잡도를 줄이고, R-ELAN(Residual Efficient Layer Aggregation Networks)을 활용하여 피처 집계(feature aggregation)를 향상시킵니다. 또한, YOLO 시스템의 실시간 제약에 더 잘 맞도록 기본 주의 메커니즘을 개선했습니다. 이 모델은 5가지 스케일로 개발되었으며: YOLOv12-N, S, M, L, X가 포함되어 있습니다. 이들은 각각의 스케일에서 널리 사용되는 YOLO 및 RT-DETR 모델을 성능에서 능가합니다.
- ***Performance Highlights***: YOLOv12-N은 6.5G FLOPs와 2.6M 파라미터로 40.6% mAP를 달성하며 YOLOv11-N을 초과합니다. 또한 YOLOv12-S는 RT-DETR-R18보다 1.5%~0.1% mAP 향상된 성능을 보이며, 처리 속도에서는 42% 더 빠르게 작동합니다. 전체적으로 YOLOv12는 기존 YOLO 및 RT-DETR 시리즈보다 정확도와 효율성에서 뛰어난 성능을 보입니다. 특히 YOLOv12-X는 모든 모델 중 최고 성능을 기록하며, 60.2% mAP를 달성했습니다.

### [Harnessing Vision Models for Time Series Analysis: A Survey](https://arxiv.org/abs/2502.08869)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.08869.png)

Vote: 1

Authors: Dongsheng Luo, Dongjin Song, Wei Cheng, Ziming Zhao, ChengAo Shen, Jingchao Ni, Hanghang Tong, Haifeng Chen

- ***What's New***: 이 논문은 시계열 데이터 분석에 비전 모델(Vision Models; LVMs 및 VLMs)을 활용한 방법들을 최초로 체계적으로 조사하였습니다. 이 연구는 시계열 데이터를 이미지로 변환하여 비전 모델을 적용하는 방법을 중점으로 다루며, 시계열 분석에 대한 새로운 시각적 접근을 제시합니다.
- ***Technical Details***: 연구는 시계열을 이미지로 변환하는 다양한 기법(Line Plot, Heatmap, Spectrogram 등)과 그 이미지를 통해 시계열 분석에 적용하는 전통적 비전 모델부터 최신 LVMs, LMMs까지의 방법론을 포괄적으로 검토합니다. 각 방법론에 대해 자세한 분류학을 제시하며, 변환된 이미지를 LVM과 LMM을 통해 분류, 예측, 이상 탐지 등의 시계열 작업에 활용하는 방법을 설명합니다.
- ***Performance Highlights***: 비전 모델이 시계열 이미지를 통해 고효율적인 시계열 패턴 인식을 가능하게 하며, 특히 사전 학습된 LVMs가 시계열 분석에 뛰어난 상호 디멘션 전환 능력을 통해 성능을 향상시킬 수 있음을 발견했습니다. 또한, 여러 변환 방법의 조합이 단일 방법보다 이미지에 대한 강건성을 증가시켜 분류 작업에서의 성능을 향상시켰습니다.

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
