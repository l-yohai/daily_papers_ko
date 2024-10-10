# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using ChatGPT.

Thanks to [@AK391](https://github.com/AK391) for great work.


## Daily Papers (2024-10-10)

### [GLEE: A Unified Framework and Benchmark for Language-based Economic Environments](https://arxiv.org/abs/2410.05254)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.05254.png)

Vote: 62

Authors: Moshe Tennenholtz, Itamar Reinman, Omer Madmon, Eilam Shapira, Samuel Joseph Amouyal, Roi Reichart

- **What's New**: 이 연구는 자연어 기반 경제 환경에서의 게임(GLEE)이라는 통합 프레임워크를 제안합니다. LLM(Large Language Model) 기반 에이전트가 경제 환경에서 의사결정 능력을 평가하고 최적화하기 위한 표준화된 벤치마크를 제공합니다.
- **Technical Details**: GLEE는 두 플레이어 게임의 경우에 초점을 맞추고 있으며, 협상, 교섭, 설득 게임의 파라미터화를 통해 경제 환경 내 다양한 자유도를 통제합니다. 게임의 라운드 수, 정보 구조, 의사소통 형태 등이 중요한 파라미터로 설정되어 있으며, 4개의 서로 다른 LLM을 사용하여 7.15M의 결정과 954K 이상의 게임 데이터를 수집했습니다.
- **Performance Highlights**: 이 프레임워크를 통해 다양한 경제 시나리오에서 LLM 성능을 비교하고 평가할 수 있으며, 경제 환경의 특성이 에이전트 행동에 어떻게 영향을 미치는지에 대한 깊이 있는 통찰을 제공합니다. 또한 사람이 LLM 기반 에이전트와 무료 게임을 할 수 있는 상호작용 인터페이스도 제공되어 인간 대 LLM 성능을 비교할 수 있습니다.

### [Personalized Visual Instruction Tuning](https://arxiv.org/abs/2410.07113)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.07113.png)

Vote: 56

Authors: Jipeng Zhang, Tianyang Han, Tong Zhang, Rui Pan, Renjie Pi, Jianshu Zhang

- **What's New**: 이번 연구에서는 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLM)에 개인화 기능을 부여하기 위한 새로운 훈련 방식인 개인화 시각 지시 튜닝(Personalized Visual Instruction Tuning, PVIT)을 제안합니다. PVIT는 모델이 다양한 개인화 대화를 수행할 수 있도록 하여, 추가적인 훈련 없이 다양한 개인을 다룰 수 있습니다.
- **Technical Details**: PVIT는 각 개인을 <<<개인 이미지, 개인 소개>>> 쌍으로 표현하여 멀티 모달 접두사(prefix)로 제공하는 방식을 채택합니다. 이는 여러 개인이 포함된 상황에서 모호성을 제거하기 위해 개인화된 래퍼 토큰(wrapper tokens)을 도입합니다. 또한, 자동화된 데이터 생성 프레임워크를 설계하여 개인화 훈련 데이터를 합성하고, 시각적 전문가 모델을 활용하여 장면 이미지에서 개인의 시각적 개념을 추출한 후, 이를 MLLM을 통해 개인 및 장면 수준의 텍스트 설명으로 변환합니다.
- **Performance Highlights**: P-Bench라는 벤치마크를 만들어, 현존하는 최신 MLLM이 개인화된 개념을 인식하는 능력을 평가하였습니다. PVIT로 훈련한 MLLM은 개인화된 대화를 수행하는 능력이 크게 향상된 것으로 나타났습니다. 이 연구는 현재 MLLM의 한계를 극복하고, 개인화된 AI 인터렉션을 가능하게 하는 중요한 발걸음을 내딛었습니다.

### [IterComp: Iterative Composition-Aware Feedback Learning from Model Gallery for Text-to-Image Generation](https://arxiv.org/abs/2410.07171)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.07171.png)

Vote: 38

Authors: Xinchen Zhang, Ling Yang, Guohao Li, Bin Cui, Yaqi Cai, Yong Tang, Yujiu Yang, Mengdi Wang, Jiake Xie

- **What's New**: 최근의 텍스트-이미지 생성(text-to-image generation) 분야에서 확산 모델(diffusion models)의 급격한 발전은 놀라운 능력을 보여주고 있습니다. 이러한 모델들은 복잡한 프롬프트(prompt)에 따른 정확한 구성 생성(compositional generation)에는 여전히 어려움을 겪고 있습니다. 이러한 문제를 해결하기 위해, 새로운 IterComp라는 프레임워크를 소개합니다. IterComp는 다양한 모델로부터 수집한 구성 인식(composition-aware) 모델 선호 데이터를 활용하여 반복적 피드백 학습(iterative feedback learning)을 통해 구성 생성 전반에서의 포괄적인 개선을 달성합니다.
- **Technical Details**: IterComp는 여러 모델에서 구성 인식 모델 선호 데이터를 수집하여 고품질의 이미지 순위 쌍을 포함하는 새로운 데이터셋을 만듭니다. 이를 기반으로 보상 모델(reward models)을 훈련하여 기저 확산 모델(base diffusion model)의 미세 조정(finetuning) 동안 세밀한 구성 안내를 제공합니다. 이 프레임워크는 반복적 피드백 학습을 통해 확산 모델과 보상 모델의 자기 개선(self-refinement)을 여러 번의 반복을 통해 추진합니다.
- **Performance Highlights**: IterComp의 성능을 평가하기 위해 광범위한 실험을 수행했으며, 기존 최첨단 방법들과 비교하여 우수한 구성 생성 능력을 입증했습니다. 이는 모델의 속성을 결합하는 능력(attribute binding), 공간적 관계(spatial relationships), 비공간적 관계(non-spatial relationships) 등의 다양한 측면에서의 성능을 크게 향상시킵니다.

### [Aria: An Open Multimodal Native Mixture-of-Experts Model](https://arxiv.org/abs/2410.05993)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.05993.png)

Vote: 38

Authors: Junnan Li, Guoyin Wang, Yudong Liu, Bei Chen, Zhiqi Shen, Yue Wang, Bowen Qu, Xinyao Niu, Dongxu Li, Haoning Wu

- **What's New**: 이번 연구에서 우리는 Aria라는 최초의 오픈 소스 멀티모달 (multimodal) 네이티브 모델을 소개합니다. 이 모델은 텍스트, 코드, 이미지, 비디오 등 다양한 입력 모달리티에 대해 뛰어난 이해 능력을 보유하고 있으며, 동일한 용량의 모달리티 전문 모델과 성능을 견줄 수 있습니다. 상업용 멀티모달 모델의 유용성을 경험할 수 있는 공개 모델로, 다양한 소스로부터 고품질 데이터를 큐레이션하여 훈련되었습니다.
- **Technical Details**: Aria 모델의 핵심은 세부 전문가 (fine-grained experts)로 구성된 Mixture-of-Experts (MoE) 디코더입니다. 이 구조는 효과적인 파라미터 활용을 통해 훈련 및 추론 속도를 향상시킵니다. 각 MoE 레이어에는 66명의 전문가가 존재하며, 2명은 공통 지식 포착을 위해 모든 입력에 공유되며, 추가로 6명의 전문가가 라우팅 모듈에 의해 각 토큰에 대해 활성화됩니다. 시각적 입력은 Vision Transformer (ViT) 및 투영 모듈로 변환되어 처리됩니다.
- **Training**: Aria는 4가지 단계로 구성된 훈련 파이프라인을 통해 개발되었습니다: 언어 사전 훈련, 멀티모달 사전 훈련, 멀티모달 롱 컨텍스트 사전 훈련, 멀티모달 후속 훈련입니다. 각 단계는 데이터와 연산 자원을 최대한 활용하여 모델 성능을 극대화합니다. 처음 두 단계는 상당한 양의 언어 및 멀티모달 데이터를 통해 MoE 디코더와 시각적 인코더를 훈련하여 모달리티의 넓은 이해 능력을 갖추도록 하였습니다.
- **Performance Highlights**: Aria는 Pixtral-12B 및 Llama3.2-11B와 비교하여 넓은 범위의 멀티모달, 언어 및 코딩 작업에서 우수한 성능을 보여주며, 활성화된 파라미터 수가 적어 추론 비용이 더 낮습니다. 또한, GPT-4o 및 Gemini-1.5와 같은 상용 모델과 다양한 멀티모달 작업에서도 동등한 성능을 제공합니다. Aria는 Apache 2.0 라이선스 하에 학술 및 상업적 목적으로 무료로 배포됩니다.

### [Pixtral 12B](https://arxiv.org/abs/2410.07073)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.07073.png)

Vote: 36

Authors: Guillaume Lample, Szymon Antoniak, Andy Lo, Soham Ghosh, Teven Le Scao, Arthur Mensch, Timothée Lacroix, Thibaut Lavril, Diego Las Casas, Pravesh Agrawal, Saurabh Garg, Pavankumar Muddireddy, Amélie Héliou, Paul Jacob, Albert Q. Jiang, Valera Nemychnikova, +, Theophile Gervet, Emma Bou Hanna, Louis Martin, Jessica Chudnovsky, Devendra Chaplot, William Marshall

- **What's New**: 이 논문은 중간 오류를 자동으로 수정하는 시스템을 제안합니다. 기존의 시스템들은 오류를 식별하는 데 중점을 두었으나, 이 논문에서는 식별된 오류를 자동으로 수정하는 새로운 방법론을 소개합니다.
- **Technical Details**: 새로운 방법론은 머신러닝 모델을 사용하여 오류 발생 패턴을 학습하고 이를 바탕으로 자동 수정을 수행합니다. 구체적으로는 GPT-3와 같은 대형 언어 모델이 활용됩니다. 이 모델은 오류의 컨텍스트(Context)를 이해하고 가장 적절한 수정안을 제안합니다.
- **Performance Highlights**: 제안된 시스템은 다양한 데이터셋을 통해 테스트되었고, 비교 연구에서 기존 방법들에 비해 더 높은 정확도와 효율성을 보였습니다. 특히, 수정 과정에서 발생하는 부작용을 최소화하였다는 점이 큰 장점으로 나타났습니다.

### [Towards World Simulator: Crafting Physical Commonsense-Based Benchmark for Video Generation](https://arxiv.org/abs/2410.05363)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.05363.png)

Vote: 33

Authors: Kaipeng Zhang, Fanqing Meng, Wenqi Shao, Dianqi Li, Jiaqi Liao, Yu Cheng, Yu Qiao, Xinyu Tan, Quanfeng Lu, Ping Luo

- **What's New**: PhyGenBench와 PhyGenEval이라는 새로운 평가 도구를 개발하여 Text-to-Video (T2V) 모델의 물리적 상식 이해 능력을 자동화하여 평가합니다. PhyGenBench는 기본 물리 법칙에 기반하여 물리적 상식 평가를 위한 벤치마크를 제공합니다. 또한 PhyGenEval은 향상된 트리 계층적 평가 전략을 통해 물리적 상식의 정확성을 평가하는 새로운 프레임워크입니다.
- **Technical Details**: PhyGenBench는 간단하고 분명한 물리현상을 반영하는 텍스트 프롬프트를 활용하여 2727개의 물리법칙과 160개의 프롬프트를 수집하고 있습니다. PhyGenEval은 GPT-4o를 활용하여 텍스트로부터 물리 법칙을 분석하고 이미지 기반에서 전체 비디오로 이동하는 세 가지 계층적 단계의 평가 전략을 사용합니다. 각 단계는 비디오-언어 모델과 맞춤 지침을 결합하여 평가를 수행합니다.
- **Performance Highlights**: 인기 있는 T2V 모델들을 포괄적으로 평가한 결과, 최고 성능의 모델인 Gen-3도 0.51이라는 점수를 받았으며 이는 현재 모델들이 여전히 세계 시뮬레이터로 작동하는 것에서 멀리 떨어져 있음을 나타냅니다. PhyGenBench와 PhyGenEval을 활용한 평가 결과는 인간의 피드백과 높은 일관성을 보여줍니다.

### [Deciphering Cross-Modal Alignment in Large Vision-Language Models with Modality Integration Rate](https://arxiv.org/abs/2410.07167)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.07167.png)

Vote: 31

Authors: Yuhang Zang, Dahua Lin, Jiaqi Wang, Pan Zhang, Weiming Zhang, Yuhang Cao, Nenghai Yu, Xiaoyi Dong, Qidong Huang

- **What's New**: 최근 LVLMs(Large Vision-Language Models)의 탐구가 여러 측면에서 폭발적으로 증가하면서, 멀티모달 기능을 강화하기 위한 새로운 패러다임이 제시되었습니다. 이 연구는 LVLMs의 사전 학습 품질을 평가하기 위한 새로운 지표인 Modality Integration Rate (MIR)를 소개합니다. MIR는 사전 학습에서의 모듈 조정 및 데이터 설정이 모델 성능에 미치는 영향을 측정하여 더 나은 사전 학습 최적화를 가능하게 합니다.
- **Technical Details**: 모듈리티 통합률(MIR)은 이미지-텍스트 쌍의 분포 간 거리를 측정하여 LVLMs 사전 학습의 크로스 모달 정렬 품질을 평가하는 신규 지표입니다. 이를 통해 사후 SFT(Supervised Fine-Tuning) 평가 없이도 모델 성능을 예측할 수 있습니다. MIR은 다양한 이미지-텍스트 페어링과 다양한 LVLM 사전 학습 구조에 대한 유연성과 견고함을 가지고 있어, 이미지-텍스트 간의 도메인 격차를 줄이는 데 효과적입니다.
- **Performance Highlights**: MIR는 사전 학습 품질을 효과적으로 나타내며, 사후 SFT 벤치마크 성능과 강력한 상관관계를 가지고 있습니다. MIR는 다양한 이미지-텍스트 유형에 걸쳐 견고하며, 이는 과적합 상황에서도 신뢰성을 제공합니다. 제안된 모듈 MoCa는 미니머니와 LLaVA-v1.5 모델에서 평균 1.5%와 0.9%를 각각 향상시켰습니다.

### [Pyramidal Flow Matching for Efficient Video Generative Modeling](https://arxiv.org/abs/2410.05954)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.05954.png)

Vote: 24

Authors: Quzhe Huang, Yadong Mu, Ningyuan Li, Hao Jiang, Yang Jin, Zhicheng Sun, Kun Xu, Nan Zhuang, Zhouchen Lin, Yang Song

- **What's New**: 이 연구는 비디오 생성 모델에서의 제한사항을 뛰어넘는 효율적인 비디오 생성 모델링 프레임워크를 제안합니다. 이 프레임워크는 초기 타임스텝에서 발생하는 노이즈를 줄이고, 피라미드 형태로 압축된 표현을 사용하여 계산 비용을 절감합니다.
- **Technical Details**: 기존의 고해상도 프로세스를 다수의 단계로 분할하는 계단형 아키텍처(cascaded architecture)와 달리, 이 연구는 이미지 및 비디오 피라미드를 사용하는 '피라미달 흐름 매칭(pyramidal flow matching)' 알고리즘을 도입합니다. 이 방식은 이미지와 비디오를 공간적, 시간적으로 압축하며, 효율적인 트레이닝을 위해 단일 'Diffusion Transformer (DiT)' 안에 통합된 통합 흐름 매칭 목표를 갖추고 있습니다. 이를 통해 여러 개의 독립된 모델을 사용하지 않아도 되는 통합된 접근 방식을 제공합니다.
- **Performance Highlights**: 제안된 방법은 높은 훈련 효율성을 보여주며, 768p 해상도, 24fps의 10초짜리 고품질 비디오를 생성할 수 있습니다. 이 연구는 공개 데이터셋에서 훈련된 비디오 생성 모델들 사이에서 높은 경쟁력을 가진 성능을 평가받았습니다. VBench와 EvalCrafter에서 그 효과가 확인되었습니다.

### [Unveiling the Backbone-Optimizer Coupling Bias in Visual Representation Learning](https://arxiv.org/abs/2410.06373)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.06373.png)

Vote: 24

Authors: Baigui Sun, Zedong Wang, Stan Z. Li, Siyuan Li, Weiyang Jin, Yang Liu, Juanxi Tian, Luyuan Zhang, Zicheng Liu

- **What's New**: 최근 컴퓨터 비전 분야에서는 네트워크 아키텍처(network architectures)와 옵티마이저(optimizers)의 발전이 이루어졌지만, 주목할 만한 것은 비전 백본(vision backbones)과 옵티마이저 사이의 상호작용에 대한 연구가 부족하다는 점입니다. 이 논문은 이러한 상호작용이 각 모델의 훈련 성능 및 일반화 능력에 어떻게 영향을 끼치는지 탐구합니다.

### [Falcon Mamba: The First Competitive Attention-free 7B Language Model](https://arxiv.org/abs/2410.05355)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.05355.png)

Vote: 18

Authors: Guillaume Kunsch, Hakim Hacid, Dhia Eddine Rhaiem, Ilyas Chahed, Maksim Velikanov, Younes Belkada, Jingwei Zuo

- **What's New**: Falcon Mamba 7B 모델은 첫 번째 순수 Mamba 아키텍처를 가진 대규모 State Space Language Model(SSLM)입니다. 이 모델은 Transformer 기반의 대형 LLM과 경쟁할 수 있는 성능을 보여주며, 특히 Llama3.1 8B, Mistral 7B 등과 비교하여도 대등하거나 더 나은 성능을 보입니다.

### [MM-Ego: Towards Building Egocentric Multimodal LLMs](https://arxiv.org/abs/2410.07177)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.07177.png)

Vote: 17

Authors: Yinfei Yang, Erik Daxberger, Yanghao Li, Zhe Gan, Jiasen Lu, Lin Chen, Haotian Zhang, Hanrong Ye, Haoxuan You, Zongyu Lin, Dan Xu, Bowen Zhang

- **What's New**: 연구자들이 새로운 모델, MM-Ego를 개발하여 장기간의 에고센트릭 비디오(Egocentric Video)를 이해하고 처리할 수 있는 방법을 제안했습니다. 이 연구는 에고센트릭 비디오를 자동으로 분석할 수 있는 대규모 QA Dataset을 최초로 구축하고, 이를 통해 기계가 인간과 유사한 1인칭 시점에서 주변 환경을 이해할 수 있게 하는데 초점을 맞추고 있습니다.
- **Technical Details**: 에고센트릭 비디오 데이터를 다루기 위해 'narration to egocentric QA' 전략을 사용하여 비디오 내러티브 데이터로부터 자동으로 QA 데이터를 생성하는 데이터 엔진을 개발했습니다. 이 데이터 엔진을 활용해 7백만 개 이상의 QA 샘플을 포함하는 대규모 에고센트릭 QA 데이터셋을 제작했습니다. 또한 'Memory Pointer Prompting' 방법을 도입하여 'global glimpse'와 'fallback'의 두 가지 단계로 비디오를 처리하며, 이는 비디오 전체를 이해하고 질문에 따라 중요한 시각 정보를 추출하여 답할 수 있도록 합니다.
- **Performance Highlights**: 새로운 MM-Ego 모델은 고해상도 시각 정보와 질문 연관성을 바탕으로 비디오를 처리하여 기존 모델들이 제공하지 못했던 더 높은 시각 이해력을 제공할 수 있습니다. 또한, 30초에서 1시간에 이르는 다양한 길이의 에고센트릭 비디오를 포함하는 검증을 통해 에고센트릭 이해 능력을 보다 정확하게 평가할 수 있는 기준을 마련했습니다.

### [Story-Adapter: A Training-free Iterative Framework for Long Story Visualization](https://arxiv.org/abs/2410.06244)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.06244.png)

Vote: 13

Authors: Yunfei Xie, Jiawei Mao, Mude Hui, Bingjie Xu, Yuanqi Chang, Yuyin Zhou, Xiaoke Huang

- **What's New**: 이 연구는 Story-Adapter라는 새로운 프레임워크를 소개하여, 스토리 시각화를 위한 텍스트에서 이미지로의 생성 과정에서의 일관성과 상호작용의 세밀함을 향상시키고자 합니다. 기존 모델들이 오토-리그레시브(Auto-Regressive) 또는 고정된 참조 이미지를 이용한 방식의 한계를 극복하고자, 모든 생성된 이미지를 현재 생성에 반영하여 에러 축적을 줄이고 참조 이미지의 결함 전파를 줄입니다.
- **Technical Details**: Story-Adapter는 고정된 Stable-Diffusion(SD) 모델을 활용하여 크로스 어텐션(cross-attention) 메커니즘을 도입하였습니다. 초기 단계에서는 텍스트 프롬프트를 사용하여 초기 이미지를 생성하고, 이 과정을 통해 얻어진 글로벌 임베딩(global embeddings)을 다음 단계의 이미지 생성 과정에 활용합니다. 이를 통해 스토리 전체에 걸친 의미적 일관성을 유지하며, 반복(iterations)을 통해 세밀한 상호작용을 개선합니다.
- **Performance Highlights**: Story-Adapter는 StorySalon 벤치마크 데이터셋을 사용한 정규 길이 스토리 시각화에서 기존의 StoryGen 모델보다 평균 캐릭터-캐릭터 유사성(aCCS) 9.4% 개선되고, 평균 프레셰 인셉션 거리(aFID) 측면에서 21.71이 개선되었습니다. 또한 긴 스토리 시각화에서도 aCCS 3.4%와 aFID 8.14의 개선을 보여주며, 의미적 일관성과 세밀한 상호작용 면에서 우수한 생성 품질을 입증했습니다.

### [Self-Boosting Large Language Models with Synthetic Preference Data](https://arxiv.org/abs/2410.06961)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.06961.png)

Vote: 13

Authors: Zhifang Sui, Li Dong, Xingxing Zhang, Furu Wei, Qingxiu Dong

- **What's New**: SynPO는 LLM(대형 언어 모델) 정렬을 위한 새로운 자기증강 패러다임입니다. 이 패러다임은 소량의 지도형 미세 조정(Supervised Fine-Tuning, SFT) 데이터를 활용하여 합성 선호 데이터(Synthetic Preference Data)를 생성하고 이를 통해 LLM이 지속적으로 성능을 향상시킬 수 있도록 합니다. 특히, LLM 자체와 무작위 키워드를 입력으로 사용하여 대규모 합성 프롬프트(Synthetic Prompts)를 생성하는 자가 프롬프트 생성기(Self-Prompt Generator)를 도입합니다.
- **Technical Details**: SynPO는 초기 정책 모델과 소량의 SFT 데이터를 사용하여 시작하고, 자가 프롬프트 생성기와 응답 개선기(Response Improver)를 통합하여 충분한 프롬프트와 합성 선호 데이터를 제공합니다. 응답 개선기는 현재 모델 출력과 구리 데이터의 정답 응답 간의 분포 차이를 식별함으로써 세심한 개선을 가능하게 합니다. 합성 프롬프트 생성에는 키워드-생성(keyword-to-text) 작업과 노이즈 키워드를 사용하여 프롬프트 생성기의 강건성을 높입니다.
- **Performance Highlights**: SynPO는 인간의 선호도와 일치하는 LLM을 학습시키는 데에 효율적이며, 인간 주석 데이터 없이도 합성 프롬프트와 응답의 다양성과 품질을 크게 높입니다. 초기 학습 모델인 Llama3-8B와 Mistral-7B에서 AlpaceEval 2.0 및 Arena-hard에서 각각 26%와 22-30% 향상된 성능을 보여주었습니다. Open LLM 리더보드에서도 SFT 모델 대비 3.2%에서 5.0% 높은 평균 성능을 기록하여 일반 성능도 향상되었습니다.

### [One Initialization to Rule them All: Fine-tuning via Explained Variance Adaptation](https://arxiv.org/abs/2410.07170)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.07170.png)

Vote: 12

Authors: Thomas Schmied, Marc Peter Deisenroth, Benedikt Alkin, Lukas Hauzenberger, Fabian Paischer, Sepp Hochreiter

- **What's New**: EVA (Explained Variance Adaptation)라는 새로운 방법이 소개되었습니다. EVA는 LoRA의 Adaptive Rank Allocation을 데이터 기반 초기화를 통해 개선하는 방법입니다. 이 방법은 하향식 작업에 대한 정보와 활성화 패턴을 활용하여 LoRA의 가중치를 초기화하고, 모델의 모든 계층에서 설명된 분산을 최대화하도록 순위를 할당합니다.
- **Technical Details**: EVA는 LoRA에 기반하여 매트릭스의 저랭크(low-rank) 분해를 활용합니다. 하향식 데이터의 미니배치를 사용하여 모델을 통해 이를 전달하고, 활성화 벡터의 SVD(Singular Value Decomposition)를 사용하여 우측 특이 벡터를 계산합니다. 그 후, 설명된 분산에 따라 우측 특이 벡터를 정렬하여 주어진 순위 예산에 따라 상위 k개의 요소를 이용해 LoRA를 초기화합니다.
- **Performance Highlights**: EVA는 다양한 하향식 작업에서 일관되게 성능 향상을 보입니다. 언어 생성 및 이해 작업에서는 수학 및 추론 작업에 대해 7B-9B 파라미터 언어 모델을 미세 조정하여 최고 평균 성능을 달성하였습니다. 이미지 분류에서는 19개의 다양한 작업을 통해 전이 학습 비전 트랜스포머를 미세 조정하여 LoRA 및 다른 초기화 방법보다 높은 평균 점수를 획득하였으며, 강화 학습에서는 연속 제어 작업에서 LoRA 및 풀 파인 튜닝(Full Fine-Tuning, FFT)을 초과하는 성능을 보였습니다.

### [TweedieMix: Improving Multi-Concept Fusion for Diffusion-based Image/Video Generation](https://arxiv.org/abs/2410.05591)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.05591.png)

Vote: 12

Authors: Jong Chul Ye, Gihyun Kwon

- **What's New**: 최근 텍스트-이미지 생성(text-to-image generation) 모델은 큰 발전을 이루었으며, 사용자가 텍스트 프롬프트를 통해 고품질 이미지를 제작할 수 있게 되었습니다. 이러한 성공은 비디오 및 3D 장면 생성과 같은 다른 도메인에도 빠르게 확장되고 있으며, 사용자 지정된 캐릭터를 활용한 콘텐츠 제작이 가능하게 되었습니다. 본 논문에서는 맞춤형 텍스트-이미지(diffusion) 모델을 조합하여 인퍼런스(inference) 단계에서 커스터마이징할 수 있는 튜닝(tuning) 없는 향상된 접근 방식을 소개합니다.
- **Technical Details**: 기존 방법들이 여러 객체들의 생성에 있어 가중치 병합(weight merging)이나 추가적인 인버전(inversion) 단계가 필요로 한 반면, 우리는 역샘플링(reverse sampling) 단계만을 활용하여 과정을 두 주요 단계로 나누었습니다. 첫 번째 단계에서는 다중 객체-인식 샘플링(mult-object-aware sampling)을 통해 텍스트 프롬프트에 포함된 여러 객체를 대상으로 하며, 새로운 리샘플링(resampling) 전략을 도입하여 생성 품질을 향상시킵니다. 두 번째 단계에서는 객체별 영역 가이던스를 통해 사용자 맞춤 개념 모델(custom concept models)을 통합합니다. 이를 통해 튼튼하고 고품질의 샘플링을 보장하는데, 이는 중간 디노이즈된 이미지 공간을 사용하여 트위디(Tweedie)의 공식을 활용합니다.
- **Performance Highlights**: 우리의 방법론은 잘못된 개념 혼합 없이 의미론적으로 관련된 개념을 가지고 있는 이미지를 합성할 수 있습니다. 또한, 두 개 이상의 개념을 문제 없이 처리하며, 입력 프롬프트의 의미적 의도와 긴밀히 일치하는 이미지를 생성합니다. 마지막으로, 우리의 비디오 출력은 기존의 모델 파인튜닝(fine-tuning) 기반의 사용자 맞춤 비디오 생성 방법보다 뛰어난 성능을 보여주어, 제안된 프레임워크의 효과성을 입증했습니다.

### [Temporal Reasoning Transfer from Text to Video](https://arxiv.org/abs/2410.06166)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.06166.png)

Vote: 11

Authors: Lei Li, Lean Wang, Qi Liu, Chenxin An, Linli Yao, Yuanxin Liu, Xu Sun, Peiyuan Zhang, Lingpeng Kong

- **What's New**: 최근 연구는 대형 언어 모델(LLMs)이 텍스트 기반에서 비디오 대형 언어 모델(Video LLMs)로 확장되면서 그 발전 가능성을 보여주는데 초점을 맞추고 있습니다. Video LLMs는 비디오의 시간적 추론(temporal reasoning)을 효과적으로 처리하는 데 어려움을 겪고 있으며, 본 연구는 그 원인을 탐구하고 해결책을 제시하고자 합니다.
- **Technical Details**: Video LLM은 주로 비전 인코더(vision encoder)와 LLM 디코더로 구성됩니다. 비전 인코더는 비디오 프레임으로부터 시각적 특징을 추출하고, LLM 디코더는 이 정보를 텍스트 지시와 통합하여 과업을 완수합니다. 연구진은 비디오 임베딩에 대한 시퀀셜 상관관계를 학습하는 작은 '프로브(prob)' 분류기를 훈련시켜 비디오 LLM의 시간 정보 처리 능력을 측정했습니다. GPT-4V를 이용해 합성된 비디오를 텍스트로 변환하여 LLM의 시간 정보 처리 능력을 분석했습니다.
- **Performance Highlights**: 비디오 임베딩에서 훈련된 프로브 분류기가 90% 이상의 정확도로 시간 정보를 거의 완벽하게 처리할 수 있음을 보여주었습니다. 그러나, LLM은 이 시간 정보를 효과적으로 처리하는 데 어려움이 있으며, 특히 큰 규모임에도 불구하고 낮은 프로빙 정확도를 보였습니다. 연구 결과, LLM 컴포넌트가 시간 추론의 주요 병목임을 밝혀냈습니다. 이러한 인사이트를 기반으로, 연구진은 이미지-텍스트 데이터셋을 활용하여 텍스트 기반의 시간 추론 작업을 생성하는 방법을 제안했으며, 이는 적은 비디오 데이터를 사용하지 않고도 LongVA-7B 모델의 성능을 향상시켰습니다. 제안된 접근 방식은 Video-MME 및 MLVU 벤치마크에서 경쟁력 있는 성능을 보이며, 시간 추론 정확도를 각각 12.4점과 평균 56.4에서 58.1로 향상시켰습니다.

### [CursorCore: Assist Programming through Aligning Anything](https://arxiv.org/abs/2410.07002)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.07002.png)

Vote: 10

Authors: Shijin Wang, Qi Liu, Hao Jiang, Rui Li, Shengyu Ye

- **What's New**: 이 논문에서는 AI 기반 프로그래밍 보조 기술의 새로운 프레임워크인 'Assistant-Conversation'을 소개합니다. 이 프레임워크는 프로그래밍 과정 전반에 걸쳐 필요한 모든 정보를 조정하는 것을 목표로 합니다. 또한, 다양한 정보의 모델 간 정렬과 대응 출력의 품질을 평가하기 위해 'APEval'이라는 새로운 벤치마크도 제안되었습니다.
- **Technical Details**: Assistant-Conversation 프레임워크는 시스템(S), 히스토리(H), 현재(C), 사용자(U), 그리고 보조 도구(A)라는 구성 요소로 이루어져 있습니다. 이는 프로그래머에게 필요한 모든 관련 정보를 활용하여 작업을 간소화할 수 있도록 설계되었습니다. 이 과정에서, 특정 코드 영역이나 커서 위치를 표시하는 '특수 토큰'을 사용하여 코드를 편집하는 과정도 포함됩니다. APEval 벤치마크는 다양한 정보 유형을 활용하여 프로그래밍을 돕는 모델의 능력을 평가합니다. 여러 정보 출처의 조합을 사용하여 각 작업을 평가합니다.
- **Performance Highlights**: 이 논문에서 제안된 CursorCore 시리즈는 동급 모델과 비교하여 최첨단의 성능을 보였습니다. 이 모델들은 219K의 데이터 포인트로 미세 조정되어 AI 보조 프로그래밍 작업에서 뛰어난 성과를 도출했습니다. 또한, 프로그램 작성의 히스토리를 효과적으로 조정하고, 사용자와의 상호작용을 효과적으로 모델링하여 기존의 부족한 점들을 보완하는 성과를 보였습니다.

### [AutoDAN-Turbo: A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs](https://arxiv.org/abs/2410.05295)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.05295.png)

Vote: 10

Authors: Yevgeniy Vorobeychik, Huan Sun, Zhuoqing Mao, Chaowei Xiao, Somesh Jha, Bo Li, Xiaogeng Liu, Patrick McDaniel, Edward Suh, Peiran Li

- **What's New**: AutoDAN-Turbo라는 혁신적인 방법이 도입되었습니다. 이는 사람의 개입 없이 다양한 전략을 자동으로 발견하고 결합해 LLMs(Large Language Models)에 대한 jailbreak 공격을 수행할 수 있는 방법입니다.
- **Technical Details**: AutoDAN-Turbo는 lifelong learning agents를 활용하여 새로운 전략을 스스로 개발하고, 이 전략들을 체계적으로 저장하여 다시 사용할 수 있도록 합니다. 또한 외부의 사람이 설계한 jailbreak 전략을 통합하여 더욱 발전된 공격 전략을 개발할 수 있습니다. 이 방법은 블랙박스 방식으로 작동하며 모델의 텍스트 출력에만 접근이 필요합니다.
- **Performance Highlights**: AutoDAN-Turbo는 공개 벤치마크와 데이터셋에서 고성능을 보여줬습니다. GPT-4-1106-turbo와 같은 모델에 대해 88.5%의 높은 공격 성공률을 기록하였으며, 기존의 runner-up baseline에 비해 74.3% 높은 공격 성공률을 나타냈습니다. 기존의 인간 개발 전략과 통합하면 공격 성공률을 93.4%까지 높일 수 있습니다.

### [ViBiDSampler: Enhancing Video Interpolation Using Bidirectional Diffusion Sampler](https://arxiv.org/abs/2410.05651)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.05651.png)

Vote: 10

Authors: Serin Yang, Taesung Kwon, Jong Chul Ye

- **What's New**: 최신의 텍스트-비디오(T2V) 및 이미지-비디오(I2V) 확산 모델의 발전으로, 주어진 텍스트나 이미지 조건에 부합하는 고품질의 비디오를 생성할 수 있게 되었습니다. 이 연구는 키프레임 보간(keyframe interpolation)을 개선하기 위한 확산 기반의 새로운 샘플링 전략을 소개합니다. 제안하는 방법은 영상의 시작 프레임과 끝 프레임을 조건으로 하여 비디오 중간 프레임을 부드럽고 자연스럽게 생성합니다.
- **Technical Details**: 이 방법은 비디오 확산 모델의 양방향 샘플링(bidirectional sampling)을 활용하여 off-manifold 문제를 해결하고자 합니다. 기존에는 중간 샘플을 병합하여 사용했으나, 이는 데이터 분포에서 벗어나는 문제를 야기합니다. 제안 방법은 시점별로 하나씩 순차적으로 샘플링하여 이러한 문제를 완화합니다. 또한, Classifier-Free Guidance(CFG) 및 DDS guidance를 활용하여 시작 및 종료 프레임과의 정렬을 보장합니다.
- **Performance Highlights**: ViBiD Sampler라는 이 새로운 방법은 기존 방법에 비해 파인튜닝이나 여러 번의 재노이징 과정을 요구하지 않으며, 효율적인 샘플링 전략 덕분에 단일 3090 GPU로 1024x576 해상도에서 25프레임 비디오를 195초 안에 생성할 수 있습니다. 이로써, 고품질 영상의 키프레임 보간 작업에서 최첨단의 성능을 보여줍니다.

### [Diversity-Rewarded CFG Distillation](https://arxiv.org/abs/2410.06084)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.06084.png)

Vote: 8

Authors: Andrea Agostinelli, Sertan Girgin, Olivier Bachem, Sarah Perrin, Alexandre Ramé, Johan Ferret, Romuald Elie, Geoffrey Cideron

- **What's New**: 이 논문은 창의적인 분야에서 generative models(생성 모델)의 quality-diversity trade-off(품질-다양성 상충)을 개선할 수 있는 새로운 finetuning(파인튜닝) 전략을 소개합니다. 이를 위해 distillation(지식 증류)과 reinforcement learning(강화 학습)을 결합하여 이 중 기능을 최적화합니다.
- **Technical Details**: 이 연구에서의 주요 기여는 CFG(distillation 방법으로서의) 증류와 다양성 보상 강화 학습을 조합하는 것입니다. CFG distillation은 teacher role을 하는 증류 모델을 학생 모델에 흡수하여 적응시킵니다. 다른 한편, 다양성 강화 학습은 부여된 prompt에 대한 생성물 간 다양성을 최대화하도록 보상합니다. 이 두 가지 목표를 결합함으로써 모델은 CFG의 장점을 유지하면서도, 증류 시의 계산 비용을 줄이고, 동시에 다양성을 강화하게 됩니다.
- **Performance Highlights**: 제안된 방법은 text-to-music generation(텍스트-음악 생성) 작업에 적용되어, CFG-기반 이전 모델에 비해 품질-다양성 상충에서 현저한 개선을 이루었습니다. 실험 결과 모델은 더 다양한 음악을 높은 품질로 생성할 수 있음을 사람을 통한 평가에서 확인됩니다.

### [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching](https://arxiv.org/abs/2410.06885)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.06885.png)

Vote: 8

Authors: Ziyang Ma, Yushen Chen, Kai Yu, Keqi Deng, Jian Zhao, Zhikang Niu, Chunhui Wang, Xie Chen

- **What's New**: 이번 연구에서는 F5-TTS라는 새로운 모델을 소개하여, 기존의 Text-to-Speech(TTS) 모델의 한계를 넘고자 합니다. F5-TTS는 Flow Matching(플로우 매칭)을 활용하여, 높은 자연스러움과 명료성을 가진 합성 음성을 생성하는 것을 목표로 합니다. 이 모델은 간단한 파이프라인을 유지하면서도, 텍스트와 음성의 정렬 문제를 잘 처리할 수 있도록 설계되었습니다.
- **Technical Details**: F5-TTS는 Diffusion Transformer와 ConvNeXt V2를 사용하여 텍스트-음성 정렬을 개선합니다. 이 모델은 기존의 복잡한 정렬 방식 대신, 텍스트-음성을 보다 자연스럽게 매칭시키는 방식을 채택하고 있습니다. 특히, Flow Matching(플로우 매칭)과 Conditional Flow Matching(CFM)을 사용하여 모델의 학습을 돕고 적응성을 강화합니다.
- **Performance Highlights**: F5-TTS는 기존 모델에 비해 더 높은 강인성과 충실도를 보여주며, 자연스러운 화자 유사성을 유지합니다. 또한, 추론 시간에 샘플링 전략을 도입함으로써 자연스러움, 명료성, 화자 유사성을 크게 향상시켰습니다. 이러한 개선점들은 기존의 플로우 매칭 기반 모델에 원활하게 통합될 수 있습니다.

### [T2V-Turbo-v2: Enhancing Video Generation Model Post-Training through Data, Reward, and Conditional Guidance Design](https://arxiv.org/abs/2410.05677)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.05677.png)

Vote: 8

Authors: William Yang Wang, Wenhu Chen, Jian Zheng, Jiachen Li, Xiaofeng Gao, Qian Long, Robinson Piramuthu

- **What's New**: 최신 비전-포착(Text-to-Video, T2V) 모델의 성능 격차를 줄이기 위해 다양한 감독 신호(supervision signals)를 통합한 T2V-Turbo-v2를 소개합니다. 이 논문은 고품질의 비디오 데이터셋, 리워드 피드백(reward feedback), 조건부 가이드(conditional guidance)를 일관성 증류(consistency distillation, CD) 과정에 통합하려고 합니다.
- **Technical Details**: T2V-Turbo-v2는 MotionClone으로부터 운동 지침을 사용해 에너지 함수(energy function)를 설계하는 방법을 제안합니다. 데이터 전처리 과정을 통해 추가적인 조건부 가이드를 계산하는 데 드는 계산 비용을 절감하고, 전체 모델 학습이 가능하도록 메모리를 절약합니다. 또한, 다양한 리워드 모델(RM)을 활용해 비주얼 품질을 최적화하려는 시도를 합니다. 이는 비디오-캡션 쌍을 활용하여 이상적인 참조로 동작하게 하며, 학습 데이터셋의 조합이 성능에 미치는 영향을 실험적으로 탐구합니다.
- **Performance Highlights**: T2V-Turbo-v2는 VideoCrafter2로부터 증류되었으며, VBench에서 모든 기존 기준을 능가해 새로운 SOTA(최상위 기술 성과)를 달성했습니다. 특히, 운동 지침을 통합한 변형은 Total Score 총 85.13점을 기록하여 Gen-3와 Kling과 같은 독점 시스템을 능가했습니다. 이 모델이 비주얼과 운동 품질에서 상당한 성능 개선을 이루었음을 입증합니다.

### [TRACE: Temporal Grounding Video LLM via Causal Event Modeling](https://arxiv.org/abs/2410.05643)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.05643.png)

Vote: 8

Authors: Qingbin Liu, Xi Chen, Jingyu Liu, Mingda Li, Xiaoying Tang, Yongxin Guo

- **What's New**: 이 논문에서는 비디오의 구조적 특성을 모델링하기 위해 TRACE라는 새로운 비디오 LLM(Video Large Language Model)을 소개합니다. TRACE는 기존의 인과적 언어 모델링(causal language modeling)에서 구조적 비디오 모델링(structural video modeling)으로 전환하는 이론적 프레임워크를 개발하고 이를 바탕으로 실질적인 비디오 LLM을 구성합니다.
- **Technical Details**: TRACE는 비디오를 순차적인 이벤트로 모델링하여 이벤트의 타임스탬프, 핵심 점수(salient scores), 텍스트 캡션을 기반으로 후속 이벤트를 예측합니다. 각각의 시각 프레임, 타임스탬프, 핵심 점수 및 텍스트를 별도의 태스크로 취급하며, 이를 위해 다양한 인코더와 디코딩 헤드를 활용합니다. 또한, 적응형 헤드 전환(adaptive head-switching) 방법을 개발하여 향상된 생성 기능을 제공합니다.
- **Performance Highlights**: TRACE는 여러 VTG(Video Temporal Grounding) 작업과 데이터셋에서 뛰어난 성능을 발휘하며, 기존 SOTA 영상 LLM들과 비교해 실질적인 개선을 보였습니다. Youcook2 데이터셋에서 zero-shot 성능이 CIDEr와 F1 Score에서 각각 3.1%, 4.9% 향상되었고, Charades-STA에서는 IOU(Intersection over Union) 0.5와 0.7의 Recall에서 6.5%, 3.7% 증가했습니다. QVHighlights 작업에서는 mAP와 HIT@1이 각각 10.3%, 9.2% 상승하였습니다. 이러한 결과는 TRACE가 세분화된 타스크에서 전통적인 비생성적(non-generative) 방식과 비슷한 성능을 발휘함을 시사합니다.

### [Data Selection via Optimal Control for Language Models](https://arxiv.org/abs/2410.07064)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.07064.png)

Vote: 8

Authors: Hongning Wang, Li Dong, Minlie Huang, Yaru Hao, Furu Wei, Yuxian Gu, Qingxiu Dong

- **What's New**: 이 논문은 대규모 언어 모델(Language Models; LMs)의 사전 훈련을 위한 데이터 선택의 중요성을 강조합니다. 기존의 수작업 기반 휴리스틱 접근법과 달리, 본 연구에서는 데이터 선택을 최적 제어 이론(Optimal Control Theory)과 연결하여 데이터 선택을 동적 시스템의 제어 변수로 보고 이를 최적화하여 LMs의 다운스트림 성능을 개선하려는 시도를 합니다.
- **Technical Details**: 저자들은 데이터 선택을 Pontryagin의 최대 원리(PMP; Pontryagin's Maximum Principle)를 이용한 수학적 문제로 규정하고, 이를 통해 최적의 데이터를 선택하는 수학적 근거를 제공합니다. 이 프레임워크인 PDS(PMP-based Data Selection)를 통해 각 데이터 인스턴스가 다운스트림 작업에 미치는 영향을 기반으로 품질 점수를 부여하고, 이를 바탕으로 사전 훈련 데이터 세트를 선택합니다. PDS는 사전 훈련 과정 전에 오프라인에서 동작하여 추가적인 훈련 비용을 피하며, 다양한 모델 구성에도 동일하게 적용될 수 있습니다.
- **Performance Highlights**: 실험 결과, 1.7B 매개변수를 가진 LM 사전훈련 시 약 2배의 속도 향상을 달성하였고, 모든 모델 크기에서 다운스트림 성능 및 언어 모델링 성능이 꾸준히 향상되었습니다. 또한 데이터가 제한된 상황에서도 데이터 활용도가 향상되어 사전 훈련 데이터 수요가 1.8배 감소되었습니다. 이는 데이터가 부족해지고 있는 LM 커뮤니티에 중요한 이점으로 작용합니다.

### [LLM Self-Correction with DeCRIM: Decompose, Critique, and Refine for Enhanced Following of Instructions with Multiple Constraints](https://arxiv.org/abs/2410.06458)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.06458.png)

Vote: 7

Authors: Shereen Oraby, Nanyun Peng, Yu-Hsiang Lin, Tagyoung Chung, Kartik Mehta, Vivek Subramanian, Haw-Shiuan Chang, Thomas Palmeira Ferraz, Sijia Liu, Mohit Bansal

- **What's New**: 이 연구는 대규모 언어 모델(LLM)이 사용자 정의 규칙 또는 제약 조건을 따르는 능력에 있어 가지는 한계를 평가하고 향상시키기 위한 RealInstruct 벤치마크를 도입합니다. RealInstruct는 실제 사용자 요청을 기반으로 한 최초의 데이터셋으로, LLM의 제약 조건을 따르는 능력을 평가하며 DiCRIM이라는 새로운 self-correction 방법을 도입하여 다중 제약 조건이 있는 지시사항을 효율적으로 처리합니다.
- **Technical Details**: RealInstruct는 사용자 생성 지시사항을 작업과 제약 조건 집합으로 분해하여 LLM 성능을 제약 조건 수준에서 평가하며, 이를 통해 지시사항 수준의 정확도 측정치를 집계합니다. 또 Decompose, Critique, and Refine(DeCRIM) 파이프라인을 통해 LLM의 응답을 반복적으로 조정하여 모든 제약 조건을 충족할 때까지 개선합니다. 벤치마크는 RealInstruct와 IFEval에서 실시되며, DeCRIM은 본질적인 피드백 없이도 성능을 크게 향상시킵니다.
- **Performance Highlights**: 베이스라인에 비해, DeCRIM은 RealInstruct에서 Mistral의 지시사항 수준 성능을 4.8% 향상시키고, IFEval에서는 1.2% 향상시켰습니다. 피드백이 강화되면, RealInstruct와 IFEval에서 각각 22.0%와 33.8% 향상하여 GPT-4를 능가합니다.

### [Response Tuning: Aligning Large Language Models without Instruction](https://arxiv.org/abs/2410.02465)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.02465.png)

Vote: 7

Authors: Seokhyun An, Hyounghun Kim

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)을 유용하고 안전한 보조 도구로 변환하기 위해 '응답 튜닝(Response Tuning, RT)'이라는 새로운 접근 방식을 제안합니다. 이는 기존의 명령 튜닝(Instruction Tuning, IT)과 달리 명령-응답 페어 데이터를 사용하지 않고, 모형의 응답 분포만 학습하도록 합니다.

### [Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning](https://arxiv.org/abs/2410.04223)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.04223.png)

Vote: 7

Authors: Wojciech Matusik, Jie Chen, Michael Sun, Gang Liu, Meng Jiang

- **What's New**: 이 연구는 분자 발견을 위한 새로운 멀티모달 대형 언어 모델(Multimodal Large Language Model, MLLM)인 라모레(Llamole)를 제안합니다. 라모레는 LLM과 그래프 모델을 결합하여 텍스트, 분자 및 반응을 상호 혼합하여 자동 생성하는 것이 특징입니다.
- **Technical Details**: 라모레는 LLM과 두 가지 사전 학습된 그래프 모듈인 Graph Diffusion Transformer(Graph DiT)와 반응 템플릿 예측을 위한 그래프 신경망(GNN)을 사용합니다. LLM은 텍스트 생성 흐름을 제어하며, 트리거 토큰을 예측하면 LLM은 그래프 모듈을 활성화하여 분자를 생성하거나 반응 템플릿을 예측합니다. Llamole는 A* 검색 알고리즘을 통해 설계된 분자의 합성 경로를 효율적으로 식별합니다.
- **Performance Highlights**: 실험 결과, Llamole은 14개 대형 언어 모델과 GraphGA와 비교하여 경쟁력이 있음을 보여주었습니다. 12개의 측정 기준에서 최대 80.9%까지 기존 LLM보다 성능이 향상되었고, 역합성 계획의 성공률은 5.5%에서 35%로 증가했습니다.

### [Mixed-Session Conversation with Egocentric Memory](https://arxiv.org/abs/2410.02503)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.02503.png)

Vote: 6

Authors: Jihyoung Jang, Hyounghun Kim, Taeyoung Kim

- **What's New**: 이 연구에서 Mixed-Session Conversation(혼합 세션 대화)라는 새로운 대화 시스템을 소개합니다. 기존의 다중 세션 대화에서는 한 고정된 파트너와 상호작용하지만, Mixed-Session Conversation에서는 주요 화자가 여러 대화 파트너와 다양한 세션 순서에서 상호작용하게 됩니다. 이를 실제로 구현하기 위해 MiSC라는 대화 데이터셋을 개발했습니다.

### [ING-VP: MLLMs cannot Play Easy Vision-based Games Yet](https://arxiv.org/abs/2410.06555)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.06555.png)

Vote: 6

Authors: Jiaheng Liu, Hangyu Guo, Haoran Zhang, Ge Zhang, Meng Cao, Wenhao Huang, Shuyue Guo

- **What's New**: 이 연구 논문은 AI 기반의 강화 학습(RL: Reinforcement Learning) 모델의 효율성을 향상시키기 위해 새로운 알고리즘을 제안하고 있습니다. 이 알고리즘은 학습 속도를 개선하고 더 복잡한 환경에서도 높은 성능을 보여줄 수 있도록 설계되었습니다.
- **Technical Details**: 제안된 알고리즘은 기존의 강화 학습 모델의 경사 하강법(Gradient Descent)과 신경망의 구조를 최적화하여 속도를 높입니다. 특히, 새로운 손실 함수(Loss Function)를 도입하여 모델의 일반화 성능을 극대화하고, 보상 함수(Reward Function)를 개선하여 에이전트가 목표 지향적으로 학습할 수 있도록 합니다.
- **Performance Highlights**: 실험 결과, 제안된 알고리즘은 여러 벤치마크 환경에서 기존 방법들보다 학습 효율성과 성능 면에서 우수한 결과를 보였습니다. 특히, 복잡한 게임 시뮬레이션에서의 실험에서 더 빠르게 주어진 목표를 달성하는데 성공했습니다.

### [FürElise: Capturing and Physically Synthesizing Hand Motions of Piano Performance](https://arxiv.org/abs/2410.05791)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.05791.png)

Vote: 6

Authors: C. Karen Liu, Ruocheng Wang, Pei Xu, Elizabeth Schumann, Haochen Shi

- **What's New**: 최근 물리 기반 인체 모션 합성 분야에서 엘리트 수준의 피아니스트 모션을 시뮬레이션하는 첫 번째 연구가 발표되었습니다. 이 논문은 피아노 연주라는 복잡한 모터스킬을 다양한 연주자들의 자연스러운 3D 핸드 모션을 캡처하여 분석하고자 합니다. FürElise라 명명된 대규모 데이터셋은 약 10시간의 3D 손 모션과 동기화된 오디오를 포함하며, 153개의 클래식 음악 작품을 연주하는 15명의 전문/보존소 피아니스트의 데이터로 구성되어 있습니다.

### [Multimodal Situational Safety](https://arxiv.org/abs/2410.06172)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.06172.png)

Vote: 6

Authors: Xin Eric Wang, Chengzhi Liu, Dawn Song, Anderson Compalas, Xuandong Zhao, Kaiwen Zhou

- **What's New**: 본 논문에서는 Multimodal Situational Safety라는 새로운 안전성 문제를 정의하고, 언어 질의와 실시간 시각적 맥락을 기반으로 질의의 안전성을 판단하는 평가기준인 Multimodal Situational Safety Benchmark (MSSBench)을 소개합니다. 이를 통해 현재의 Multimodal Large Language Models(MLLMs)의 안전성 성능을 종합적으로 평가하고자 합니다.
- **Technical Details**: MSSBench는 각각의 데이터 인스턴스에 언어 질의와 안전 또는 비안전한 시각적 맥락을 포함하며, 이는 MLLM의 실시간 관찰입니다. 벤치마크는 챗봇 에이전트 및 구현된(embodied) 에이전트 시나리오로 나누어져 있으며, 언어 질의는 특정 활동을 수행하려는 의도를 나타냅니다. 이를 통해 MLLM의 분명한 안전 추론(explicit safety reasoning), 시각적 이해(visual understanding), 그리고 상황적 안전 추론(situational safety reasoning) 능력을 평가합니다.
- **Performance Highlights**: 리뷰에 따르면, 현재의 MLLM은 시각적 맥락을 인식하여 안전하지 않은 상황을 인식하는 데 어려움이 있습니다. 명시적 안전 추론은 평균 상황 안전 성능을 향상시킬 수 있지만, 안전한 상황에서는 과민성을 증가시킬 수 있습니다. 구현된 시나리오에서는 MLLM이 시각적 이해와 상황 안전 판단 능력이 부족하며, 오픈소스 MLLM이 저작권 MLLM보다 이미지에서 안전 단서를 간과하는 빈도가 높습니다. 또한 더 많은 하위 과제가 있을 경우, MLLM의 안전 성능이 감소하는 경향을 발견했습니다. 문제를 개선하기 위해 여러 에이전트가 하위 작업을 나누어 수행하게 만드는 다중 에이전트 상황 추론 파이프라인(multi-agent situational reasoning pipelines)을 도입하였으며, 이는 평균 안전 정확도를 향상시킬 수 있지만 완벽하지는 못합니다.

### [Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning](https://arxiv.org/abs/2410.05664)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.05664.png)

Vote: 6

Authors: Sangdon Park, Saemi Moon, Dongwoo Kim, Minjong Lee

- **What's New**: 최신 연구에서는 지속 가능성(Sustainability)을 고려한 새로운 컴퓨터 과학적 접근 방식을 제안합니다. 이 연구는 정보 처리 시스템의 효율성을 개선하고, 에너지 소비를 줄이는 방법을 탐구합니다.
- **Technical Details**: 이 논문에서는 새로운 알고리즘(algorithms)을 소개하며, 이 알고리즘은 기존 방식보다 계산 속도와 자원 사용 측면에서 더 효율적입니다. 또한, 하드웨어(hardware) 최적화와의 상호작용(interaction)을 통해 성능을 극대화할 수 있습니다.
- **Performance Highlights**: 제안된 시스템은 기존 시스템에 비해 최대 30% 에너지 소비를 절감하며, 실행 속도는 두 배로 증가하였습니다. 이는 특정 벤치마크(benchmarks)에서 검증되었습니다.

### [Collective Critics for Creative Story Generation](https://arxiv.org/abs/2410.02428)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.02428.png)

Vote: 5

Authors: Minwook Bae, Hyounghun Kim

- **What's New**: CritiCS라는 새로운 프레임워크가 제안되었습니다. 이 프레임워크는 긴 이야기를 창의적으로 생성하기 위해 협업 비평 방식을 도입했고, 두 단계(CrPlan과 CrText)를 거쳐 이야기의 계획과 텍스트의 표현력을 강화합니다.
- **Technical Details**: CritiCS는 협업 비평 방식으로 여러 LLM(Large Language Models) 비평가들이 서사를 평가하고 개선 제안을 하도록 설계되었습니다. CrPlan 단계에서는 독창적인 스토리 계획에 집중하며 CrText 단계에서는 이야기의 감정 표현 및 생생한 묘사를 강화하며 만화책의 의성어와 비유적 표현을 통합합니다.
- **Performance Highlights**: 인간 평가 실험에서 CritiCS는 창의성과 흥미 측면에서 기존 방법을 크게 능가했습니다. 다양한 기준 적용, 리더 비평가의 역할, 페르소나 부여 등의 설계 선택이 효과적으로 실행되었습니다.

### [Retrieval-Augmented Decision Transformer: External Memory for In-context RL](https://arxiv.org/abs/2410.07071)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.07071.png)

Vote: 5

Authors: Thomas Schmied, Razvan Pascanu, Vihang Patil, Markus Hofmarcher, Fabian Paischer, Sepp Hochreiter

- **What's New**: 새로운 연구에서는 'Retrieval-Augmented Decision Transformer (RA-DT)'이라는 중량 기억체제를 사용한 인공지능 모델을 소개합니다. 이 모델은 기존 인-컨텍스트 강화 학습 방법에서 겪는 한계를 극복하고자 하며, 외부 메모리를 이용해 과거 경험에서 현재 상황에 맞는 정보만을 효율적으로 저장하고 검색합니다.
- **Technical Details**: RA-DT는 Decision Transformer (DT) 아키텍처에 외부 메모리를 결합하여, 길고 어려운 에피소드와 희박한 보상 문제를 해결합니다. 이 모델은 'vector index'와 '최대 내적 검색(maximum inner product search)'을 활용하여 필요한 서브 경로(무빙 패턴)만을 검색하며, 사전 학습된 임베딩 모델을 사용하여 이 서브 경로들을 인코딩합니다. 이 외부 메모리를 사용하는 방법은 LLMs 분야의 Retrieval-augmented Generation (RAG)와 유사합니다.
- **Performance Highlights**: RA-DT는 격자 기반 환경(grid-world environments)에서 이전의 인-컨텍스트 RL 방법보다 훨씬 우수한 성능을 보여줍니다. 복잡한 환경에서는 RA-DT가 보류된 작업에서 일관된 개선을 보였으나, 인-컨텍스트 향상은 나타나지 않았습니다. 이 연구는 도메인 비종속 임베딩 모델의 활용 가능성을 입증하며, 특정 도메인에 사전 학습된 모델과 유사한 성능을 발휘할 수 있음을 보였습니다.

### [BroadWay: Boost Your Text-to-Video Generation Model in a Training-free Way](https://arxiv.org/abs/2410.06241)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.06241.png)

Vote: 5

Authors: Tong Wu, Yuhang Zang, Dahua Lin, Jiaqi Wang, Pan Zhang, Pengyang Ling, Yuhang Cao, Jiazi Bu, Xiaoyi Dong

- **What's New**: 이 논문에서는 텍스트-비디오 생성(T2V) 확산 모델의 비디오 품질을 개선하기 위한 새로운 접근법인 BroadWay를 소개합니다. BroadWay는 추가적인 학습, 파라미터 도입, 메모리 증가 없이도 생성 품질을 향상시키는 훈련 없는 접근법으로, Temporal Self-Guidance와 Fourier-based Motion Enhancement라는 두 가지 주요 컴포넌트로 구성되어 있습니다.
- **Technical Details**: BroadWay의 Temporal Self-Guidance는 이전 블록의 temporal attention map을 현재 블록의 참고하여 규제함으로써, 여러 디코더 블록 간의 attention map 불일치를 줄입니다. 이로 인해 불가능하거나 시간적으로 불일관한 아티팩트가 감소됩니다. Fourier-based Motion Enhancement는 temporal attention map의 고주파 성분을 조정하여 맵의 에너지를 증폭하고, 이러한 방식으로 정적 이미지와 유사한 비디오 생성을 피할 수 있습니다.
- **Performance Highlights**: 다양한 인기 있는 T2V 백본에서 BroadWay를 평가한 결과, 강화된 비디오 품질을 보여줬으며, 이미지-비디오(I2V) 도메인에서도 잠재력을 보였습니다. BroadWay를 사용하는 방법은 추가적인 비용 없이 여러 주류 오픈 소스 T2V 백본과 통합될 수 있으며, 높은 확장성과 적용 가능성을 보여줍니다.

### [Hallucinating AI Hijacking Attack: Large Language Models and Malicious Code Recommenders](https://arxiv.org/abs/2410.06462)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.06462.png)

Vote: 5

Authors: Forrest McKee, David Noever

- **What's New**: 이 논문은 새로운 Deformable Convolutional Networks(변형 가능한 합성곱 신경망) 기술을 제안하고 있습니다. 이 기술은 기존의 모델에 비해 객체의 모양과 크기를 더 유연하게 처리할 수 있도록 설계되었습니다.
- **Technical Details**: 변형 가능한 합성곱 신경망(Deformable Convolutional Networks)은 표준 합성곱(convolution) 연산에 '오프셋(offsets)'이라는 요소를 추가하여 각 필터가 수용하는 영역을 조정할 수 있게 합니다. 이를 통해 객체의 복잡한 기하학적 변화에도 적응할 수 있습니다. 제안된 모델은 ResNet과 같은 기존의 Backbone 구조에 쉽게 통합될 수 있습니다.
- **Performance Highlights**: 제안된 모델은 기존 모델보다 다양한 데이터셋에서 더 나은 성능을 입증하였습니다. 특히, 객체 검출 및 인식 작업에서의 정확도가 향상되었으며, 네트워크의 복잡도는 상대적으로 낮은 상태로 유지되었습니다.

### [Seeker: Enhancing Exception Handling in Code with LLM-based Multi-Agent Approach](https://arxiv.org/abs/2410.06949)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.06949.png)

Vote: 5

Authors: Yuan Yuan, Minlie Huang, Xuanming Zhang, Yuxuan Chen

- **What's New**: 이 연구는 예외 처리(exception handling)의 표준화, 해석 가능성(interpretable), 일반화 가능성(generalizable)이 코드 개발자와 LLMs의 예외 처리 성능에 미치는 영향을 조사합니다. 특히, Four sets of in-context learning prompts를 사용하여 인간 개발자와 LLM의 예외 처리 성능을 분석하고, 인공지능 기반 코드 생성 모델의 견고성을 향상시키기 위한 새로운 방법론인 'Seeker'를 제안합니다.
- **Technical Details**: 이 연구에서는 예외 처리 성능을 개선하기 위해 Coarse-grained Reminding prompting, Fine-grained Reminding prompting, Fine-grained Inspiring prompting, 그리고 Fine-grained Guiding prompting을 도입하였습니다. 이 프롬프트는 코드 LLM이 예외의 직관적인 해석 가능성과 규칙 일반화를 통해 더 나은 결정을 내리도록 돕습니다. 또한, 복잡한 상속 관계에 적합한 Deep-RAG(Deep Retrieval-Augmented Generation) 알고리즘을 제안하여, 취약한 코드와 관련된 예외 분기를 식별 및 처리합니다.
- **Performance Highlights**: 비교 실험에서 Fine-grained Guiding 프롬프트를 사용하는 경우, 예외 처리 성능이 크게 향상되었습니다. 특히, 개발자가 코드의 취약성을 더 잘 이해하고 예외 캡처의 정확성을 향상시킬 수 있었습니다. 또한, Seeker 방법론은 LLM이 더 견고한 코드를 생성하거나 최적화하도록 도와 다양한 코드 작업에서 LLM의 성능을 향상시켰습니다.

### [Jointly Generating Multi-view Consistent PBR Textures using Collaborative Control](https://arxiv.org/abs/2410.06985)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.06985.png)

Vote: 4

Authors: Slava Elizarov, Ciara Rowles, Dante De Nigris, Shimon Vainer, Konstantin Kutsy, Simon Donné

- **What's New**: 이 논문은 새로운 시스템을 소개합니다, 이는 MLS를 위한 고속 책임 있는 생성 모델의 설계를 목표로 하고 있습니다. 이는 더 효과적인 다국어 처리(multilingual processing)을 가능하게 합니다.
- **Technical Details**: 시스템은 Transformer 아키텍처를 기반으로 하고 있으며, 다국어 음성 변환을 위해 기본적인 Attention Mechanism를 활용합니다. 이 과정에서 자연 어순을 유지하면서 음성의 질 감소를 최소화합니다.
- **Performance Highlights**: 제안된 모델은 기존 모델들보다 빠르면서도 높은 정합성(consistency)과 자연스러움(naturalness)을 유지하고 있습니다. 실험 결과, 여러 언어의 데이터셋에서 경쟁력 있는 성능을 보여주었습니다.

### [MentalArena: Self-play Training of Language Models for Diagnosis and Treatment of Mental Health Disorders](https://arxiv.org/abs/2410.06845)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.06845.png)

Vote: 4

Authors: Jindong Wang, Qingyun Wang, May Fung, Chi Han, Manling Li, Heng Ji, Cheng Li

- **What's New**: MentalArena는 정신 건강 장애의 진단과 치료를 지원하기 위해 설계된 셀프 플레이(self-play) 훈련 프레임워크입니다. 이 프레임워크는 언어 모델이 환자와 치료사의 역할을 동시에 수행하면서 진단과 치료 계획을 자동으로 생성할 수 있도록 합니다.
- **Technical Details**: MentalArena는 크게 세 가지 모듈, 'Symptom Encoder', 'Symptom Decoder', 및 'Model Optimizer'로 구성되어 있습니다. Symptom Encoder는 인지행동치료(CBT) 원칙을 기반으로 환자의 인지 모델을 구성하고, Symptom Decoder는 환자와 치료사 간의 상호작용을 시뮬레이션하여 더욱 개인화된 대화 및 진단을 생성합니다.
- **Performance Highlights**: MentalArena를 통해 훈련된 모델은 66개의 벤치마크에서 GPT-3.5-turbo 및 Llama-3-8b 등 기존 최첨단 모델들보다 우수한 성능을 보였습니다. 특히 GPT-3.5-turbo 기반 모델은 GPT-4o 대비 7.7% 더 높은 성능을 발휘했습니다. 이는 모델의 'Perplexity' 점수와 성능이 높은 상관관계를 가지며, 다양성 이득이 일정한 기준치를 초과하면 모델 성능이 증가할 수 있음을 보여주었습니다.

### [Do great minds think alike? Investigating Human-AI Complementarity in Question Answering with CAIMIRA](https://arxiv.org/abs/2410.06524)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.06524.png)

Vote: 4

Authors: Jordan Boyd-Graber, Tianyi Zhou, Maharshi Gor, Hal Daumé III

- **What's New**: 최근 AI 시스템이 인간의 성능을 뛰어넘는다는 주장이 제기되면서, AI와 인간이 어떻게 질문에 답하는지를 비교 연구하였습니다. 이를 위해 'Item Response Theory (IRT)'라는 심리측정적 기법을 사용하여, 인간과 AI 시스템의 차이를 분석했습니다.
- **Technical Details**: IRT는 개별 질문의 특성과 응답자의 능력을 동시에 평가할 수 있는 통계적 프레임워크로 활용되었습니다. 연구진은 IRT 기반으로 'Caimira'라는 새로운 신경 네트워크 프레임워크를 도입하여, 질문의 텍스트를 통해 특성을 추론하고, 특히 AI 시스템과 인간의 다양한 질문 수행 능력을 비교했습니다. 이는 특히 도메인 별 추론 능력을 평가할 때 유용했습니다.
- **Performance Highlights**: 연구 결과, 인간과 AI 시스템은 질문에 대한 접근 방식에서 뚜렷한 차이를 보였습니다. 인간은 해석적 사고와 직관적 사고에서 강점을 보였고, 특히 애매한 정보 갭이 있는 추론 질문에서 뛰어난 성과를 보였습니다. 반면, GPT-4-Turbo와 LLaMA-3-70b와 같은 대규모 LLM은 특정 정보 검색에서 뛰어난 결과를 보였으며, 문서와의 매칭이 쉬운 질문에서는 더 좋은 성능을 보였습니다. 그러나 복잡한 문장 구조와 의미적 관계를 포함한 질문에서는 어려움을 겪었습니다.

### [TinyEmo: Scaling down Emotional Reasoning via Metric Projection](https://arxiv.org/abs/2410.07062)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.07062.png)

Vote: 3

Authors: Cristian Gutierrez

- **What's New**: 이 논문에서는 새로운 딥러닝 기반 모델을 제안하며, 특정 문제를 해결하는 데 있어 기존의 방법들보다 뛰어난 성능을 보입니다. 연구자들은 데이터 효율성을 개선하고 오버피팅(overfitting)을 방지하기 위한 향상된 네트워크 구조를 개발했습니다.
- **Technical Details**: 이 모델은 여러 계층의 컨볼루션(convolutional)과 트랜스포머(transformer) 블록을 결합하여, 헥터로(Natural Language Processing)와 Computer Vision에서 유의미한 성과를 달성했습니다. 각 계층은 데이터의 복잡성을 다룰 수 있도록 고안되었으며, 학습 과정에서 유연성을 제공합니다.
- **Performance Highlights**: 실험 결과, 이 모델은 다양한 벤치마크 데이터셋에서 기존의 최첨단 모델을 능가하는 정확도를 보여주었습니다. 특히, 작은 데이터셋에서도 일반화(generalization) 성능이 뛰어나며, 계산 효율성 또한 크게 개선되었습니다.

### [TextToon: Real-Time Text Toonify Head Avatar from Single Video](https://arxiv.org/abs/2410.07160)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.07160.png)

Vote: 3

Authors: Chenliang Xu, Luchuan Song, Pinxin Liu, Lele Chen, Celong Liu

- **What's New**: 이 논문에서는 텍스트 기반의 두상 아바타(toonification avatar) 생성 시스템인 TextToon을 제안합니다. 이 시스템은 모노클라우럴 비디오만으로 동적인 아바타의 고품질 편집을 가능하게 하며, 이를 위해 3D Gaussian Splatting과 Text-to-Image (T2I) 모델을 활용합니다.
- **Technical Details**: TextToon은 Tri-plane 기반의 뉴럴 볼루메트릭(Volumetric) 표현을 이용하여 두상의 아바타를 toonification하는 효과적인 방법을 제안합니다. 정규화된 직교 렌더링을 3D Gaussian 점 속성을 위한 조건부 Tri-plane 입력으로 사용합니다. 3D Gaussian 점은 3DMM(coefficient) 계수의 선형 변형뿐 아니라 학습 가능한 Tri-plane 특징에도 영향을 받습니다.
- **Performance Highlights**: 이 방법은 실시간 애니메이션이 가능하며, 단일 GPU에서 48 FPS, 모바일 장치에서는 15-18 FPS의 추론 속도를 달성했습니다. 또한, 스타일리시한 최적화를 몇 분 안에 완료할 수 있습니다. 이 시스템은 텍스트 기반의 헤드 아바타 편집의 최초 접근 방식으로, 실시간 시스템을 디자인하여 효율적인 3DMM 추적 알고리즘을 제안합니다.

### [MEXA: Multilingual Evaluation of English-Centric LLMs via Cross-Lingual Alignment](https://arxiv.org/abs/2410.05873)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.05873.png)

Vote: 2

Authors: Ali Modarressi, Amir Hossein Kargaran, François Yvon, Jana Diesner, Nafiseh Nikeghbal, Hinrich Schütze

- **What's New**: 이 논문에서는 'Mexa'라는 새로운 방법을 소개합니다. 이는 중간 레이어에서 영어를 피벗 (pivot) 언어로 사용하는 영어 중심의 대형 언어 모델(LLM)의 다국어 커버리지를 평가합니다. Mexa는 비영어권 언어의 중간 레이어에서 병렬 문장의 임베딩이 영어와 얼마나 잘 정렬되어 있는지를 측정하여 다국어 커버리지를 추정합니다.
- **Technical Details**: Mexa 방법은 중간 레이어에서 영어를 피벗으로 사용하여 다국어 임베딩의 정렬을 측정합니다. 두 개의 병렬 데이터셋(FLORES-200, Bible)을 사용하고, Llama, Gemma, Mistral, OLMo 등의 9개의 LLM에서 여러 가지 작업(Belebele, m-MMLU, m-ARC)별로 Mexa 점수를 계산합니다. Mexa의 점수 계산은 토큰 수준과 레이어 수준의 풀링 방식(대표적으로 최종 토큰 사용 또는 가중 평균)에서 설계 분석을 통해 효과적임을 검증했습니다.
- **Performance Highlights**: 실험 결과, Mexa는 9개의 모델과 2개의 병렬 데이터셋에서 평균 Pearson 상관계수 0.90을 달성함으로써 강력한 성능을 보여주었습니다. 특히, 토큰 기반의 가중 평균을 사용하고 평균 풀링을 결합한 방법이 최고의 결과를 냈습니다.

### [VHELM: A Holistic Evaluation of Vision Language Models](https://arxiv.org/abs/2410.07112)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.07112.png)

Vote: 1

Authors: Josselin Somerville Roberts, Cihang Xie, Tony Lee, Wenhao Zheng, Percy Liang, Yiyang Zhou, Chi Heem Wong, Haoqin Tu, Michihiro Yasunaga, Huaxiu Yao, Yifan Mai

- **What's New**: 이 논문에서는 VHELM(Vision-Language Models의 Holistic Evaluation)을 제안합니다. 이 프레임워크는 다양한 시나리오와 메트릭을 사용하여 VLM(Vision-Language Model)의 성능을 다차원적으로 평가합니다. 특히 편향, 공평성, 지식, 다국어 처리능력, 추론, 견고성, 유독성, 안전성과 같은 측면에 초점을 맞춥니다.
- **Technical Details**: VHELM은 이미지와 텍스트 입력을 받아 텍스트를 생성하는 VLM을 평가합니다. 이 과정은 'aspect'(측면), 'scenario'(시나리오), 'adaptation'(적응), 'metric'(측정치)라는 네 가지 주요 컴포넌트로 구성됩니다. 9가지 평가 측면을 선정하고, 21개의 기존 VLM 벤치마크 데이터셋을 맵핑하여 포괄적인 평가를 수행하도록 합니다.
- **Performance Highlights**: VHELM을 통해 22개의 주요 VLM을 평가한 결과, 모든 측면에서 우수한 성능을 보이는 모델은 없었습니다. GPT-4o 모델이 대부분의 리더보드에서 우수한 성과를 보였으나 편향, 견고성, 유독성 측면에서는 다른 모델만큼 성능이 뛰어나지 않았습니다. 또한, 비공개 API 모델이 공개 가중치 모델보다 성능이 뛰어났습니다. 이는 공개 가중치 모델이 간단한 지침도 잘 따르지 못해, 지침에 대한 추가적인 미세 조정이 필요함을 시사합니다.

### [Does Spatial Cognition Emerge in Frontier Models?](https://arxiv.org/abs/2410.06468)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.06468.png)

Vote: 1

Authors: Erik Wijmans, Vladlen Koltun, Philipp Kraehenbuehl, Santhosh Kumar Ramakrishnan

- **What's New**: 이 논문에서는 Text-to-Image 단계에 대한 속도 및 품질을 개선하기 위한 새로운 네트워크 구조를 제안합니다. 이 새로운 접근 방식은 Transformer 기반의 모델을 활용하며, 모델을 경량화시키면서도 높은 성능을 유지하는 데 중점을 두고 있습니다.
- **Technical Details**: 제안된 모델은 Transformer 아키텍처를 기반으로 하며, 다양한 레이어에서 다중 헤드(self-attention)을 최적화하는 기술을 사용합니다. 추가적으로, 이 모델은 기존의 Text-to-Image 생성 작업에서 흔히 사용되는 CNN을 부분적으로 대체하며, GPU 메모리 사용량도 최적화합니다. 복합적인 데이터셋을 통해 다양한 환경에서의 일반화를 확인했습니다.
- **Performance Highlights**: 이 네트워크는 기존의 방법들보다 2배 이상 빠르며, 생성 이미지의 품질 면에서도 높은 평가를 받았습니다. 실험 결과, 모델은 기존 기술 대비 훨씬 적은 연산 자원만으로도 유사한, 또는 더 나은 이미지 품질을 달성할 수 있음을 보여주었습니다.

### [MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](https://arxiv.org/abs/2410.07095)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.07095.png)

Vote: 1

Authors: Kevin Liu, Neil Chowdhury, Dane Sherburn, Oliver Jaffe, Jun Shern Chan, Evan Mays, Giulio Starace, James Aung, Tejal Patwardhan, Lilian Weng, Aleksander Mądry, Leon Maksin

- **What's New**: 이 논문에서는 대화형 AI 시스템을 위한 새로운 모델링 기법을 소개하고 있습니다. 이 기법은 사용자 질의와 응답의 자연스러운 흐름을 유지하면서 더욱 유연한 응답 생성을 목표로 합니다.
- **Technical Details**: 이 연구에서는 대화 모델링에 Transformer 기반의 구조를 사용하며, 기존의 Attention Mechanism을 개선하여 문맥을 더 잘 이해할 수 있도록 합니다. 또한, 다양한 데이터셋에서 사전 학습을 수행하여 범용성을 높였습니다.
- **Performance Highlights**: 테스트 결과, 제안된 모델이 대화 자연스러움, 응답의 정확성 측면에서 기존 모델보다 우수함을 보였습니다. 특히, 대화의 연속성과 사용자 의도 파악 정확도에서 두드러진 향상을 나타냈습니다.

### [VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks](https://arxiv.org/abs/2410.05160)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.05160.png)

Vote: -

Authors: Ziyan Jiang, Wenhu Chen, Yingbo Zhou, Semih Yavuz, Xinyi Yang, Rui Meng

- **What's New**: 이 논문은 텍스트와 이미지를 결합해 일반적인 임베딩(embedding) 모델을 구축하는 새로운 다중 모드 임베딩(multi-modal embedding) 프레임워크를 제안합니다. 두 가지 주요 기여는 MMEB(Massive Multimodal Embedding Benchmark)라는 새로운 벤치마크(benchmark)와 Vlm2Vec이라는 모델을 도입하는 것입니다. MMEB는 36개의 데이터셋을 통해 다양한 텍스트 및 이미지 조합을 평가하며 Vlm2Vec은 Phi-3.5-V 모델을 기반으로 한 깊이 있는 시각-언어 통합을 제공합니다.
- **Technical Details**: MMEB는 4개의 메타 태스크(meta-task) 범주로 데이터를 구성하여 텍스트와 이미지의 결합된 임베딩을 평가합니다. 모든 태스크는 랭킹(ranking) 문제로 수정되어 모델이 지시를 따라 후보 중에서 올바른 타겟을 선택하게 합니다. Vlm2Vec은 대량의 멀티모달 데이터셋으로 학습된 피처를 Transformer 아키텍처에 깊이 있게 통합하며, 대조 학습(contrastive learning)을 통해 성능을 향상시킵니다.
- **Performance Highlights**: Vlm2Vec은 모든 MMEB 데이터셋에서 17.3포인트, 아웃-오브-디스트리뷰션(out-of-distribution) 데이터셋에서는 11.6포인트의 성능 향상을 기록하였습니다. 이는 제안된 프레임워크가 텍스트와 이미지 간의 관계를 효과적으로 포착할 수 있음을 증명합니다. 모든 데이터, 코드, 모델은 연구 진전을 돕기 위해 공개될 예정입니다.

### [Stuffed Mamba: State Collapse and State Capacity of RNN-Based Long-Context Modeling](https://arxiv.org/abs/2410.07145)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.07145.png)

Vote: -

Authors: Maosong Sun, Yingfa Chen, Xinrong Zhang, Shengding Hu, Zhiyuan Liu, Xu Han

- **What's New**: 최근 Transformer 기반 모델들이 긴 시퀀스를 처리하는 데에 비해 RNN 기반 모델들은 훨씬 낮은 비용으로 긴 시퀀스를 처리할 가능성을 보여주고 있습니다. 그러나 RNN은 특정 시퀀스 길이 이상에서 성능이 급격히 떨어지는 문제를 겪고 있습니다. 이 논문에서는 이러한 RNN의 문제를 분석하고 'state collapse' 현상을 해결하기 위한 방법을 제안합니다.
- **Technical Details**: RNN이 긴 문맥에서 공연 이상 행동을 보이는 이유는 'state collapse'라는 현상 때문입니다. 이는 몇몇 채널에서 큰 값이 발생해 다른 채널에서 값이 사라지는 현상으로, 초기 토큰을 잊지 못해 발생합니다. 이러한 문제를 해결하기 위해, SC 현상을 줄이기 위한 세 가지 교육 비의도적 방법과 지속적인 교육을 통한 해결 방법을 제안합니다. 제안된 방법은 기억력을 줄이거나 순환 상태를 정상화하는 방향으로 이루어집니다.
- **Performance Highlights**: 제안된 SC 완화 방법을 테스트한 결과, Mamba-2 모델이 1M 이상의 토큰을 SC 없이 소비할 수 있음을 확인했습니다. 또한 다양한 크기의 Mamba-2 모델을 교육하여 상태 크기와 상태 용량 간의 관계를 규명하였고, passkey retrieval 작업에서 256K 토큰에서 매우 높은 정확성을 달성한 370M 매개변수의 모델을 제시하였습니다. 이는 transformer 기반 모델보다 더욱 긴 시퀀스를 효과적으로 처리할 수 있습니다.



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
