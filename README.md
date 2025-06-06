# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using GPT-4o.

Thanks to [@AK391](https://github.com/AK391) for great work.

## Daily Papers (2025-05-09)

### [Perception, Reason, Think, and Plan: A Survey on Large Multimodal Reasoning Models](https://arxiv.org/abs/2505.04921)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2505.04921.png)

Vote: 75

Authors: Lanqing Hong, Zheng Zhang, Zhuotao Tian, Borui Jiang, Wenhan Luo, Zitao Li, Xinping Zhao, Shenyuan Jiang, Shouzheng Huang, Xintong Wang, Min Zhang, Weihua Luo, Baotian Hu, Jifang Wang, Haoyuan Shi, Xinyu Chen, Longyue Wang, Yunxin Li, Xuanyu Zhang, Baoxing Huai, Zhenyu Liu, Zhenran Xu

- ***What's New***: 이 논문은 대규모 멀티모달 추론 모델(Large Multimodal Reasoning Models; LMRMs)의 발전을 종합적으로 검토하며 새로운 개념으로 내부적으로 멀티모달 이해와 에이전트 추론, 제네리티브 추론을 목표로 하는 네이티브 멀티모달 추론 모델(Native Large Multimodal Reasoning Models; N-LMRMs)을 제안합니다. 이는 향후 멀티모달 AI 시스템의 설계 방향을 명확히 제시합니다.
- ***Technical Details***: 논문은 네 가지 단계의 발전 로드맵을 바탕으로 멀티모달 추론 연구를 체계적으로 정리했습니다. 초기에 태스크별 모듈을 활용한 추론에서 시작해, 최근의 멀티모달 LLMs(대형 언어 모델)와 멀티모달 체인 오브 쏘트(Multimodal Chain-of-Thought; MCoT), 멀티모달 강화 학습 등의 접근 방식으로 발전하였습니다. 나아가 복잡한 실세계 환경에서 확장 가능하고 적응 가능한 추론과 계획을 지원하는 N-LMRMs의 개념적 방향을 논의합니다.
- ***Performance Highlights***: 멀티모달 추론 모델들이 다룰 수 있는 복잡한 태스크를 해결할 잠재력을 보여주지만, 실제 적용에 있어서는 여러 도전 과제를 여전히 가지고 있습니다. 논문에서는 LMRMs의 미래 기술적 전망을 소개하며, 주요 방향으로 멀티모달 에이전트 추론 및 옴니-모달(Omni-Modal) 이해와 생성 추론의 향상을 제안합니다.

### [On Path to Multimodal Generalist: General-Level and General-Bench](https://arxiv.org/abs/2505.04620)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2505.04620.png)

Vote: 54

Authors: Shuicheng Yan, Siliang Tang, Qingyu Shi, Haobo Yuan, Shengqiong Wu, Tianjie Ju, Bobo Li, Meng Luo, Tao Zhang, Tat-Seng Chua, Wentao Hu, Liangtao Shi, Minghe Gao, Junbao Zhou, Daoan Zhang, Liyu Jia, Zhiyuan Zhou, Jiahao Meng, Xiangtai Li, Hanwang Zhang, Weiming Wu, Kaihang Pan, Jiebo Luo, Yaoting Wang, Shilin Xu, Zixiang Meng, Zhiqi Ge, Yuan Zhou, Qingshan Xu, Yaobo Ye, Hao Fei, Juncheng Li

- ***What's New***: 본 연구에서는 다중 모달 일반가(Multimodal Generalist)를 평가하기 위한 새로운 프레임워크인 General-Level과 대규모 벤치마크 데이터셋 General-Bench를 소개합니다. General-Level은 MLLM(Multimodal Large Language Model)의 성능을 5단계로 분류하여, AGI(Artificial General Intelligence)를 위한 다중 모달 일반가의 진화 과정을 정교하게 평가합니다. General-Bench는 다양한 모달리티, 태스크 및 도메인을 아우르는 700개 이상의 태스크와 325,800개의 인스턴스를 포함합니다.
- ***Technical Details***: General-Level 프레임워크는 MLLM의 능력을 시너지(Synergy)라고 불리는 평가 기준을 통해 평가하며, 시너지 능력에 따라 능력을 세 가지 영역으로 나눕니다: '태스크-태스크', '이해-생성', 그리고 '모달리티-모달리티'입니다. General-Bench는 이미지를 비롯한 비디오, 오디오, 3D와 같은 다양한 모달리티를 다루며 각 태스크의 원래 형식에 맞춰 자유형식(Free-Form)으로 평가됩니다.
- ***Performance Highlights***: 100개 이상의 MLLM을 대상으로 평가한 결과, 대부분의 모델이 고급 수준의 MLLM으로 평가되기에는 부족한 시너지 효과를 보여주었습니다. GPT-4V 및 GPT-4o를 비롯한 상용 모델조차도 상위 랭킹에 도달하지 못했으며, MLLM의 성장은 단일 모달리티에서 상호 모달리티로 확장되지 않았음을 강조했습니다. 오픈 소스 모델 중 Qwen2-VL-72B는 이미지 이해 영역에서 SOTA(최첨단) 모델을 몇 가지 태스크에서 넘어섰지만, 전반적인 언어 지능은 비언어 모달리티를 통해 개선되지 않았습니다.

### [Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2505.05470.png)

Vote: 34

Authors: Di Zhang, Gongye Liu, Xintao Wang, Jie Liu, Wanli Ouyang, Pengfei Wan, Jiajun Liang, Yangguang Li, Jiaheng Liu

- ***What's New***: Flow-GRPO는 온라인 강화 학습(Online RL)을 플로우 매칭 모델(Flow Matching Models)에 통합한 최초의 방법입니다. 이 방법은 (1) 결정론적 보통 미분 방정식(ODE)을 동등한 확률 미분 방정식(SDE)으로 변환하여 원래 모델의 주변 분포를 모든 시점에서 맞추고 통계 샘플링을 가능하게 하여 RL 탐색을 가능하게 하고, (2) 디노이징 감소 전략(Denoising Reduction)을 통해 훈련 디노이징 단계를 줄이면서도 원래 추론 시점 수를 유지하여 샘플링 효율성을 크게 향상시킵니다.
- ***Technical Details***: Flow-GRPO는 GRPO를 플로우 매칭 모델에 도입하여 ODE 기반의 플로우를 동등한 SDE 프레임워크로 변환함으로써 난수성을 도입하는 ODE-to-SDE 전략을 사용합니다. 또한 온라인 RL에서 샘플링 효율성을 개선하기 위해 Denoising Reduction 전략을 적용하며, 훈련 시 샘플 생성 동안 디노이징 단계를 줄이면서도 테스트 시에는 완전한 디노이징 단계를 유지하여 성능을 유지합니다.
- ***Performance Highlights***: Flow-GRPO는 여러 텍스트 이미지 생성 과제에서 효율적이며, 복잡한 구성 요소에서는 RL 조정된 SD3.5가 거의 완벽한 객체 수, 공간 관계 및 미세한 속성을 생성하여 GenEval 정확도를 63%에서 95%로 높입니다. 시각적 텍스트 렌더링에서는 정확도가 59%에서 92%로 개선되었으며, 인류 선호도 동조에 있어서도 상당한 향상을 달성하였습니다. 실험에서 보상 해킹(reward hacking)은 거의 발생하지 않으며, 이미지 품질과 다양성의 유지는 실험에서 안정적으로 유지되었습니다.

### [Sentient Agent as a Judge: Evaluating Higher-Order Social Cognition in Large Language Models](https://arxiv.org/abs/2505.02847)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2505.02847.png)

Vote: 16

Authors: Xiaolong Li, Bang Zhang, Zheng Xie, Peisong Wang, Jiaqi Chen, Jian Li, Yifan Yang, Zhaopeng Tu, Ruotian Ma, Xingyu Chen, Yue Wang, Fanghua Ye, Qingxuan Jiang

- ***What's New***: Sentient Agent as a Judge(SAGE)는 대형 언어 모델(LLM)의 고차원 사회적 인지를 평가하는 자동화된 프레임워크입니다. SAGE는 사람의 감정 변화와 내부 생각을 시뮬레이션하여 대화 중 모델의 능력을 보다 현실감 있게 평가합니다.
- ***Technical Details***: SAGE 프레임워크는 인간과 유사한 감정과 내부 모놀로그를 시뮬레이션 하는 Sentient Agent를 포함합니다. 이는 다중턴 대화에서 LLM의 감정 상태를 평가하기 위해 두 가지 멀티홉(reasoning chain)을 실행하여 감정 궤적과 해석 가능한 내부 생각을 생성합니다.
- ***Performance Highlights***: 실험 결과, SAGE는 Barrett–Lennard 관계 재고(BLRI) 평점 및 발언 수준의 공감 지표와 높은 상관관계를 나타냈습니다. GPT-4o-Latest가 79.9점으로 가장 높은 Sentient 점수를 기록하며 사회적 추론 능력이 언어적 능력과 독립적으로 발전할 수 있음을 보여주었습니다.

### [Scalable Chain of Thoughts via Elastic Reasoning](https://arxiv.org/abs/2505.05315)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2505.05315.png)

Vote: 16

Authors: Lei Wang, Doyen Sahoo, Yuhui Xu, Hanze Dong, Caiming Xiong, Junnan Li

- ***What's New***: Elastic Reasoning은 대형 추론 모델(Large Reasoning Models; LRMs)이 제약된 자원 상황에서도 효율적으로 결과를 생성할 수 있도록 돕는 새로운 프레임워크입니다. 이 방법은 추론을 생각 단계와 해결 단계로 명확히 분리하여 각 단계마다 독립적인 토큰 예산을 배정함으로써 해결의 완전성을 우선시합니다.
- ***Technical Details***: Elastic Reasoning은 두 개의 주요 구성 요소로 이루어져 있습니다. 먼저, GRPO 훈련과 함께 경량의 예산 제약 롤아웃(budget-constrained rollout) 전략을 활용하여 불완전한 추론에서도 효과적으로 해결책을 생성합니다. 다음으로, 두 가지 단계에 대해 독립적인 예산 할당을 통해, <think>와 <solution> 구간을 나누어 모델이 충분한 해결책을 반환하도록 유도합니다. 이 접근법은 추론과 해결 모두에 할당된 총 예산을 충족시키면서도 모델 출력의 신뢰성을 높입니다.
- ***Performance Highlights***: 재정적 자원 제약 하에서도 Elastic Reasoning은 기존 방법들보다 적은 훈련 비용으로도 높은 성능을 발휘합니다. 예를 들어, AIME2024 데이터셋에서 E1-Math-1.5B 모델은 35.0%의 정확도를 달성하며, 이는 L1-Max의 27.1%와 유사한 성능이면서도 훈련의 효율성 면에서 우위를 보였습니다. 또한, E1-Code-14B는 Codeforces의 등급에 있어서 1987 점으로 높은 퍼센타일에 위치합니다.

### [FG-CLIP: Fine-Grained Visual and Textual Alignment](https://arxiv.org/abs/2505.05071)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2505.05071.png)

Vote: 10

Authors: Jincheng Li, Gengshen Zhang, Dawei Leng, Chunyu Xie, Bin Wang, Yuhui Yin, Fanjing Kong, Dawei Liang

- ***What's New***: FG-CLIP는 세 가지 주요 혁신을 통해 CLIP의 미세한 이해 능력을 향상시킨 새로운 접근법입니다. 첫째, 최신 대형 멀티모달 모델(Large Multimodal Models; LMMs)을 활용하여 16억 장의 긴 설명 이미지 쌍을 생성하고, 이를 통해 세계 수준의 의미 세부사항을 포착합니다. 둘째, 1,200만 개의 이미지와 4천만 개의 특정 영역을 포함하여 고품질 데이터셋을 구축하고, 세부적인 설명과 정밀하고 문맥이 풍부한 표현을 통해 이미지와 텍스트의 미세한 정렬을 향상시킵니다. 셋째, 모델의 미묘한 의미적 차이를 구분하는 능력을 향상시키기 위해 1천만 개의 어려운 미세한 부정 샘플을 도입합니다.
- ***Technical Details***: FG-CLIP은 두 단계로 훈련됩니다. 첫 번째 단계는 세계 수준의 의미 정렬을 위해 긴 설명을 사용하는 글로벌 대조 학습(Global Contrastive Learning)을 포함하고 있습니다. 두 번째 단계에서는 지역 적 대조 학습(Regional Contrastive Learning) 및 어려운 미세한 부정 샘플 학습(Hard Fine-Grained Negative Samples Learning)을 추가하여 지역 텍스트 데이터에 대한 이해를 더욱 정교하게 만듭니다.
- ***Performance Highlights***: FG-CLIP은 다양한 벤치마크 작업에서 CLIP 및 다른 최신 방법을 능가하는 성능을 보여줍니다. 높은 성능이 필요한 세부 수준의 이해, 오픈 단어 객체 감지, 이미지-텍스트 검색 및 일반적인 멀티모달 벤치마크에서 FG-CLIP의 우수한 성능이 강조되며, 특히 미세한 내용 및 세부 사항을 포착함으로써 전반적인 모델 성능이 향상됩니다.

### [3D Scene Generation: A Survey](https://arxiv.org/abs/2505.05474)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2505.05474.png)

Vote: 10

Authors: Ziwei Liu, Fangzhou Hong, Beichen Wen, Haozhe Xie, Zhaoxi Chen

- ***What's New***: 이 설문조사는 3D 장면 생성(3D Scene Generation)의 최신 기술을 포괄적으로 다루며, 이는 몰입형 미디어, 로봇공학, 자율주행, 내재된 AI(Embodied AI) 등 다양한 응용 분야에서 공간적으로 구조화되고 의미론적 의미가 있으며 포토리얼리스틱한 환경을 합성하는 것을 목표로 합니다. 최근의 주요 발전 사항으로는 딥 생성 모델(GANs, Diffusion Models)과 3D 표현(NeRF, 3D Gaussians)을 활용하여 현실 세계 장면 분포를 학습하면서 사실성, 다양성 및 시야 일관성을 개선한 것입니다.
- ***Technical Details***: 이 설문은 3D 장면 생성을 네 가지 패러다임으로 정리합니다: 절차적 생성(Procedural Generation), 뉴럴 3D 기반 생성(Neural 3D-based Generation), 이미지 기반 생성(Image-based Generation), 비디오 기반 생성(Video-based Generation). 각 접근법의 기술적 기초와 타협점들에 대한 분석을 제공하며, 자주 사용되는 데이터셋, 평가 프로토콜, 다운스트림 응용 프로그램을 검토합니다.
- ***Performance Highlights***: 현재의 3D 장면 생성 기술은 사실성, 3D 일관성 및 제어 가능성 간의 균형을 맞추는 도전을 안고 있습니다. 절차적 및 뉴럴 3D 기반 생성은 기하학적 일관성과 제어 가능성이 뛰어난 반면, 이미지 및 비디오 기반 생성 모델은 높은 시각적 사실성을 제공하지만 3D 일관성을 유지하는 데 어려움을 겪고 있습니다. 미국의 유명 기업과 협력한 최신 연구들은 높은 사실성과 물리적 상관성을 갖춘 통합된 인식-생성 모델의 개발이 장기적 과제임을 시사하고 있습니다.

### [ICon: In-Context Contribution for Automatic Data Selection](https://arxiv.org/abs/2505.05327)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2505.05327.png)

Vote: 9

Authors: Zhifang Sui, Yixin Yang, Linli Yao, Qingxiu Dong, Fangwei Zhu

- ***What's New***: ICon은 In-Context Learning(ICL)을 활용하여 직접적으로 각각의 샘플의 기여도를 측정하는 새로운 데이터 선택 방법입니다. 이 방법은 비용이 많이 드는 그래디언트 기반 방법이나 사람이 설계한지표 공학 없이 샘플 기여도를 점수화하여 자동 데이터 선택을 지원합니다.
- ***Technical Details***: ICON은 세 가지 구성 요소로 이루어져 있으며, ICL의 암시적인 미세 조정 성질을 이용해 샘플 기여도를 평가합니다. ICON 점수는 샘플의 기여도를 정량화하며, 긍정적인 점수는 더 큰 기여를 의미합니다. ICON에 기반한 선택 패러다임은 선형 복잡도의 추론 호출을 통해 데이터를 효율적으로 선택할 수 있도록 LoRA로 학습됩니다.
- ***Performance Highlights***: LLaMA3.1-8B 모델에서 ICON으로 선택된 15%의 데이터를 사용한 경우 전체 데이터셋을 사용한 것보다 5.42% 포인트 높은 성능을 보였으며, 다른 널리 사용되는 선택 방법보다 2.06% 포인트 더 높은 성능을 달성했습니다. 이 결과는 ICON이 적은 데이터로도 높은 수준의 성능을 제공함을 보여줍니다.

### [StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Assistant](https://arxiv.org/abs/2505.05467)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2505.05467.png)

Vote: 8

Authors: Haibo Wang, Afshin Dehghan, Bo Feng, Ping Huang, Zhengfeng Lai, Shiyu Li, Weifeng Ge, Mingze Xu, Meng Cao

- **What's New**: StreamBridge는 오프라인 Video-LLMs(Video-Large Language Models)를 스트리밍 대응 모델로 변환하는 새로운 프레임워크입니다. 기존의 오프라인 모델을 온라인 적용 시 관찰되는 두 가지 주요 문제, 즉 (1) 다회차 실시간 이해의 제한된 능력과 (2) 능동적 응답 메커니즘의 결여를 해결합니다.
- **Technical Details**: StreamBridge는 메모리 버퍼와 라운드 감쇠 압축 전략을 결합하여 긴 컨텍스트의 다회차 상호작용을 지원하며, 기존 Video-LLMs에 쉽게 통합할 수 있는 독립된 경량화된 활성화 모델을 도입해 지속적인 능동적 응답을 가능하게 합니다. 또한, StreamBridge를 지원하기 위해 스트리밍 비디오 이해를 위한 대규모 데이터셋 Stream-IT를 구축하였습니다.
- **Performance Highlights**: StreamBridge로 전환된 모델들은 OVO-Bench와 Streaming-Bench와 같은 스트리밍 벤치마크에서 뛰어난 성능을 보여줍니다. Qwen2-VL 모델의 경우 OVO-Bench 평균 점수가 55.98에서 63.35로, Streaming-Bench에서는 69.04에서 72.01로 향상되었습니다. Stream-IT에 맞춰 미세 조정한 결과, LLaVA-OV 모델은 전반적으로 더 나은 다회차 실시간 이해 성능을 달성했습니다.

### [X-Reasoner: Towards Generalizable Reasoning Across Modalities and Domains](https://arxiv.org/abs/2505.03981)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2505.03981.png)

Vote: 8

Authors: Timothy Ossowski, Sid Kiblawi, Yu Gu, Qianchu Liu, Tristan Naumann, Sam Preston, Guanghui Qin, Paul Vozila, Mu Wei, Hoifung Poon, Sheng Zhang, Ying Jin

- ***What's New***: X-REASONER는 다중 모달 및 도메인 전반에 걸쳐 추론 기능을 일반화할 수 있는지에 대한 탐구를 기반으로, 오직 일반 도메인 텍스트를 통해 비전-언어 모델(Vision-Language Model)을 고품질로 훈련할 수 있는 두 단계의 새로운 포스트 트레이닝(포스트 트레이닝) 방식을 제안합니다. 이 방법을 통해 X-REASONER는 기존의 최첨단 모델을 능가하는 성과를 보여주며, 특히 의료분야를 포함한 다중 모달과 도메인 전반에서 뛰어난 성능을 발휘합니다.
- ***Technical Details***: X-REASONER는 일반 도메인 텍스트로부터 고유한 추론 패턴을 학습하는 두 단계의 텍스트 기반 후처리 방식을 적용합니다. 첫 번째 단계는 일반 도메인 텍스트 데이터에 대한 장기 연결 추론 패턴을 활용한 감독 학습(Supervised Fine-Tuning; SFT)이며, 두 번째 단계는 증명 가능한 보상을 사용하는 강화학습(Reinforcement Learning with Verifiable Rewards; RLVR)입니다. 이것을 통해 비록 오직 텍스트만을 기반으로 훈련되었지만 복잡한 다중 모달 및 도메인 특화 작업에서 우수한 성능을 달성하였습니다.
- ***Performance Highlights***: X-REASONER는 이전에 다중 모달 데이터로 명시적으로 훈련된 최첨단 모델들과 비교하여 MMMU, MMMU-Pro, MathVista 등의 까다로운 일반 도메인 다중 모달 벤치마크에서 최고 성능을 기록하였습니다. 의료 분야에 특화된 X-REASONER-MED는 추가적인 의료 도메인 텍스트 후처리를 통해, 다양한 텍스트 및 다중 모달 의료 작업에서 새로운 기록을 세웠습니다. 이러한 성과는 일반 텍스트 기반 추론 훈련만으로도 다중 모달 추론의 학습에 충분하다는 놀라운 결과를 보여줍니다.

### [Generating Physically Stable and Buildable LEGO Designs from Text](https://arxiv.org/abs/2505.05469)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2505.05469.png)

Vote: 7

Authors: Jun-Yan Zhu, Kangle Deng, Changliu Liu, Ruixuan Liu, Ava Pun, Deva Ramanan

- ***What's New***: LEGOGPT는 텍스트 설명을 통해 물리적으로 안정적이고 조립 가능한 LEGO 모델을 생성하는 최초의 접근법입니다. 이 모델은 자율 회귀 대형 언어 모델을 활용하여 다음 브릭을 예측하며, 물리적 안정성을 보장하기 위해 물리 기반의 롤백 기법을 도입하였습니다.
- ***Technical Details***: LEGOGPT는 방대한 LEGO 디자인 데이터셋과 그에 상응하는 캡션으로 훈련된 자율 회귀 대형 언어 모델(Autoregressive Large Language Model)을 사용하여, 텍스트 기반 LEGO 디자인을 생성합니다. 안정성을 유지하기 위해, 유효성 검사와 물리적 인식을 통한 롤백을 활용하여, 물리 법칙과 조립 제약을 위반하지 않도록 합니다.
- ***Performance Highlights***: LEGOGPT는 기존의 사전 학습된 LLM과 3D 발전 기법보다 안정적이고, 시각적으로 매력적인 LEGO 디자인을 생성하는 데 뛰어난 성능을 보였습니다. 생성된 디자인은 인간이 조립하거나 로봇 팔을 통해 자동 조립할 수 있어 폭넓은 응용 가능성을 가집니다.

### [LiftFeat: 3D Geometry-Aware Local Feature Matching](https://arxiv.org/abs/2505.03422)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2505.03422.png)

Vote: 6

Authors: Yongchao Xu, Yuxuan Xiong, Zhou Zhao, Wenpeng Lai, Jun Cheng, Yepeng Liu, Jinchi Zhu

- ***What's New***: LiftFeat는 3D 기하학 정보를 로컬 피처 매칭(Local Feature Matching)에 혁신적으로 도입한 경량 네트워크입니다. 새로운 3D Geometry-aware Feature Lifting 모듈을 설계하여 2D 디스크립터와 3D 표면 노멀(3D Surface Normal)을 융합함으로써 극한 조건에서의 2D 디스크립터의 판별 능력을 크게 향상시켰습니다.
- ***Technical Details***: LiftFeat 네트워크는 표면 노멀 추정 헤드를 포함하여 3D 기하학적 지식을 학습하도록 설계되었습니다. Depth Anything v2 모델을 사용하여 예상된 깊이 맵에서 파생된 유사 표면 노멀 라벨을 사용해 추가적인 주석 비용을 절감합니다. 최종적으로, 3D-GFL 모듈을 통해 원시 2D 디스크립션과 3D 노멀 정보가 융합되어 극한 조건에서도 강력한 피처 판별 능력을 제공합니다.
- ***Performance Highlights***: LiftFeat는 상대 위치 추정, 호모그래피 예측, 시각적 로컬라이제이션 작업에서 최신 경량 방식들에 비해 우수한 성능을 보입니다. 본 연구에서는 MegaDepth-1500과 ScanNet 같은 복잡한 데이터셋에서의 테스트를 통해 AUC@5°, AUC@10°, AUC@20°에서 높은 정확도를 기록했으며, 더 복잡한 장치에서도 실시간 성능(7.4 ms의 추론 지연 시간)을 보여주었습니다.

### [WaterDrum: Watermarking for Data-centric Unlearning Metric](https://arxiv.org/abs/2505.05064)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2505.05064.png)

Vote: 5

Authors: Rachael Hwee Ling Sim, Xinyang Lu, Bui Thi Cam Nhung, Xinyuan Niu, Fanyu Wen, Bryan Kian Hsiang Low, See-Kiong Ng, Chuan-Sheng Foo, Gregory Kang Ruey Lau

- ***What's New***: WaterDrum은 대형 언어 모델(LLM) 학습에서 사적인 데이터 또는 유해한 데이터의 영향을 효과적으로 제거하기 위한 새로운 데이터 중심의 언러닝 메트릭입니다. 이 메트릭은 강력한 텍스트 워터마킹을 이용하여 기존의 유틸리티 중심 언러닝 메트릭의 한계를 극복합니다.
- ***Technical Details***: WaterDrum은 여러 데이터 소유자의 워터마크를 LLM의 출력에서 확인할 수 있는 강력한 텍스트 워터마킹 프레임워크인 Waterfall을 기반으로 개발되었습니다. 새롭게 제안된 벤치마크 데이터셋인 WaterDrum-Ax는 여러 유사한 데이터 포인트를 포함하여 언러닝 알고리즘을 엄격히 평가할 수 있도록 구성되어 있습니다.
- ***Performance Highlights***: WaterDrum 메트릭은 유사한 데이터가 존재하는 상황에서도 뛰어난 성능을 보이며, 기존 메트릭이 실패하는 경우에도 높은 분리성과 캘리브레이션을 유지합니다. 실험을 통해, 동일하거나 유사한 데이터를 포함한 시나리오에서도 강력한 성능을 입증했습니다.

### [Crosslingual Reasoning through Test-Time Scaling](https://arxiv.org/abs/2505.05408)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2505.05408.png)

Vote: 5

Authors: Carsten Eickhoff, Genta Indra Winata, Ruochen Zhang, Jonibek Mansurov, Julia Kreutzer, Zheng-Xin Yong, Niklas Muennighoff, M. Farid Adilazuarda, Stephen H. Bach, Alham Fikri Aji

- ***What's New***: 이 논문에서는 테스트 시간 확장을 통해 영어 중심 대형 언어 모델(RLMs)의 다국어 추론 능력을 향상시키는 연구를 소개합니다. English chain-of-thought(장기적 사고)로 영문 추론을 다국어로 확장할 수 있는 가능성을 실험적으로 입증했습니다.
- ***Technical Details***: 테스트 시간 확장은 RLMs의 추론 능력을 글로벌하게 확장하는 새로운 접근법입니다. Qwen2.5-Instruct 모델을 활용하며, 다양한 언어로 번역된 MGSM 평가 데이터셋과 함께 s1 모델을 사용하여 1.5B에서 32B까지의 다양한 파라미터 크기로 실험을 진행했습니다. 주어진 언어의 문제에 대해 영어 중심의 긴 장고(고찰) 추론을 적용하여 모델의 다국어 일반화 성능을 평가했습니다.
- ***Performance Highlights***: s1 모델은 크로스링구얼 테스트 시간 확장에서 상당한 성능 향상을 보였습니다. 14B 모델은 8k의 최대 추론 예산으로 9.4%의 정확도 향상을 보였으며, 이는 기본 Qwen 모델의 성능을 크게 초과합니다. 고자원 언어에서는 성능이 더 높았고, 저자원 언어에서는 다소 낮은 성능을 보였지만, 전반적으로 테스트 시간 확장이 다국어 수학적 이유 평가에서 유효함이 입증되었습니다.

### [PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes](https://arxiv.org/abs/2505.05288)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2505.05288.png)

Vote: 5

Authors: Abdelrahman Eldesokey, Zawar Qureshi, Guillermo Garcia-Hernando, Peter Wonka, Sara Vicente, Filippo Aleotti, Gabriel Brostow, Ahmed Abdelreheem, Jamie Watson

- ***What's New***: PlaceIt3D는 자연어 안내를 통해 실제 3D 장면에서 객체를 배치하는 새로운 과제를 도입했습니다. 이 작업은 텍스트 프롬프트에 따라 3D 자산이 배치되어야 할 위치를 유효하게 찾아야 하며, 기존의 3D 장면 이해 작업과는 다르게 여러 유효한 솔루션이 존재하는 모호성과 3D 기하학적 관계 및 자유 공간에 대한 추론이 필요합니다. 이 과제를 위해 새로운 벤치마크와 평가 프로토콜이 제안되었으며, 3D 자산을 훈련하고 평가하기 위한 데이터셋이 소개되었습니다.
- ***Technical Details***: PlaceIt3D 벤치마크는 ScanNet에서 파생된 142개의 실제 장면과 PartObjaverse-Tiny 데이터셋의 20개 자산, 그리고 각각의 언어 프롬프트로 구성된 3,300개의 평가 예제를 포함합니다. 각 예제는 물리적 가능성과 언어적 제약을 고려하여 3D 공간에서의 유효한 배치 위치와 방향을 찾는 것을 목표로 합니다. PlaceWizard는 공간 집계 기법과 자산 인코더, 그리고 회전 예측 등을 활용하여 이 배치 작업을 수행하는 최초의 방법으로 성능을 향상시켰습니다.
- ***Performance Highlights***: PlaceWizard는 OpenMask3D 기반의 룰 시스템과 비교하여 모든 평가 항목에서 더 높은 성능을 보였으며, 특히 공간 제약과 회전 제약 평가에서 향상된 정확도를 기록했습니다. 이러한 결과는 PlaceWizard가 자연어 기반 객체 배치 과제에서 더욱 효과적으로 작동함을 입증합니다.

### [BrowseComp-ZH: Benchmarking Web Browsing Ability of Large Language Models in Chinese](https://arxiv.org/abs/2504.19314)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2504.19314.png)

Vote: 4

Authors: Jian Chen, Qichen Ye, Peilin Zhou, Dading Chong, Chenxuan Xie, Sixin Hong, Jing Ren, Yining Hua, Yifan Shao, Xiang Ying, Bruce Leon, Chao Liu, Zhiling Jin, Yuxin Gu, Can Zhang, Meng Cao

- **What's New**: BrowseComp-ZH는 중국 웹 환경에서 대형 언어 모델(LLMs)의 웹 탐색 능력과 추론 능력을 평가하는 첫 번째 고난이도 벤치마크입니다. 기존의 BrowseComp는 주로 영어 웹에만 초점을 맞추고 있었으나, BrowseComp-ZH는 중국어 웹에서 복잡한 정보 검색과 논리적 추론을 필요로 하는 멀티홉 질문 289개로 구성되어 있습니다.
- **Technical Details**: BrowseComp-ZH 데이터셋은 영화, 기술, 법률 등 11개의 다양한 도메인을 커버하며, 각 질문은 쉽게 검증 가능한 답변에서부터 역설계되어 답변의 고유성을 보장하기 위해 두 단계의 품질 관리 프로토콜을 적용하였습니다. 또한, 세 개의 주요 검색 엔진에서 검색이 어려운 질문으로 구성되어 있으며, 여러 AI 에이전트를 활용하여 답변의 고유성과 질문의 난이도를 검증하였습니다.
- **Performance Highlights**: BrowseComp-ZH에서 평가된 대부분의 모델은 탐색 및 추론 작업에 큰 어려움을 겪으며, 정확도 10% 이하에 그친 경우가 많았습니다. OpenAI의 DeepResearch는 42.9%의 가장 높은 정확도를 기록한 반면, DeepSeek R1 모델은 23.2%에서 웹 검색 기능을 활성화할 경우 정확도가 7.6%로 급락했습니다. 이 실험 결과는 현재의 LLM들이 중국 웹 환경에서 멀티홉 검색과 논리적 추론에 상당한 도전 과제를 안고 있음을 시사합니다.

### [Putting the Value Back in RL: Better Test-Time Scaling by Unifying LLM Reasoners With Verifiers](https://arxiv.org/abs/2505.04842)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2505.04842.png)

Vote: 4

Authors: Morgane M Moss, Alessandro Sordoni, Kusha Sareen, Arian Hosseini, Rishabh Agarwal

- ***What's New***: 이 논문에서는 강화 학습(교정 보상 기반)을 통해 대규모 언어 모델(LLM)의 합리적 추론 능력을 향상시키기 위한 새로운 접근법인 RLV를 제안합니다. RLV는 기존 'value-free' 강화 학습 방식에 학습된 유도 검증기를 통합함으로써 테스트 단계에서의 계산 확장성을 높이고, MATH 데이터셋에서의 정확성을 20% 이상 개선하였습니다.
- ***Technical Details***: RLV는 RL 학습 동안 생성된 데이터를 활용하여 LLM을 추론자와 생성적 검증기로 동시에 학습시킵니다. 전통적인 value 함수 대신 생성적 검증 목표를 추가함으로써 LLM이 해결책을 생성하면서 자체의 올바름을 평가할 수 있는 점수를 제공합니다. 이 과정에서 문제 해결과 검증을 위한 공동 목표를 최적화하며, 통합된 학습 구도를 통해 발생하는 메모리 및 연산 비용을 최소화합니다.
- ***Performance Highlights***: 평행 샘플링을 활용한 RLV는 테스트 시간 계산 확장에서 8 ~ 32배의 효율성을 보이며, MATH500, GPQA 등의 데이터셋에서 기저 강화 학습 방법 대비 1.2 ~ 1.6배 향상된 성능을 달성했습니다. 이 방법은 어려운 문제 및 도메인 외 문제에 대한 강력한 일반화 능력을 보여주며, 테스트 시간에서 기존보다 훨씬 더 적은 연산으로 높은 정확성을 이끌어냅니다.

### [SIMPLEMIX: Frustratingly Simple Mixing of Off- and On-policy Data in Language Model Preference Learning](https://arxiv.org/abs/2505.02363)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2505.02363.png)

Vote: 3

Authors: Tianjian Li, Daniel Khashabi

- ***What's New***: SIMPLEMIX는 언어 모델(Language Model) 선호 학습에서 정책 내(on-policy)와 정책 외 자료(off-policy data)의 혼합을 통한 성능 향상을 제안합니다. 이 연구는 두 데이터 유형의 보완적 강점을 활용하는 방법을 탐색하며, SIMPLEMIX가 Alpaca Eval 2.0에서 기존 방법들보다 평균 6.03% 성능 개선을 달성했습니다.
- ***Technical Details***: SIMPLEMIX는 정책 내 자료가 수학 및 코딩과 같은 객관적 문제에서, 정책 외 자료가 창의적 글쓰기나 개인화된 추천처럼 열린 결과를 요구하는 작업에서 더 효과적임을 발견했습니다. 이를 바탕으로 두 데이터 출처를 단순히 혼합하여 사용합니다. 방법론은 다양한 태스크와 벤치마크를 통해 검증되었습니다.
- ***Performance Highlights***: SIMPLEMIX는 정책 내 단독 사용 대비 6.03%, 더 복잡한 정책 혼합 방법인 HyPO 및 DPO-Mix-P보다 평균 3.05% 향상된 결과를 보여줍니다. 이는 해당 방법의 단순함에도 불구하고 효과적임을 증명하며, 데이터 제한 환경에서도 데이터 출처를 혼합하는 것이 성과를 높이는 데 유리함을 보여줍니다.

### [Vision-Language-Action Models: Concepts, Progress, Applications and Challenges](https://arxiv.org/abs/2505.04769)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2505.04769.png)

Vote: 3

Authors: Ranjan Sapkota, Manoj Karkee, Konstantinos I. Roumeliotis, Yang Cao

- ***What's New***: VLA (Vision-Language-Action) 모델은 인공지능 분야의 획기적인 발전을 나타내며, 단일 계산 프레임워크 내에서 인지, 자연어 이해, 실행 기능을 통합하려는 시도를 하고 있습니다. 이 리뷰는 최근 몇 년간 VLA 모델의 주요 발전을 다섯 가지 주제 기반으로 체계적으로 정리하여 제공합니다.
- ***Technical Details***: VLA 시스템의 개념적 기초는 통합 비전-언어 모델(Vision-Language Models; VLMs), 액션 플래너, 계층적 컨트롤러와의 긴밀한 통합을 통해 추적됩니다. 이 논문은 VLA 모델의 80개 이상의 최근 논문을 검토하여 건축 혁신, 매개변수 효율적 학습 전략 및 실시간 추론 가속화와 같은 주요 발전 영역을 다룹니다.
- ***Performance Highlights***: VLA 모델은 실시간 제어, 다중 모달 행동 표현, 시스템 확장성, 보지 못한 작업에 대한 일반화 및 윤리적 배치 위험 등에서 주요 도전 과제에 직면했습니다. 하지만, 이 모델들은 농업, 의료 로봇, 증강 현실 네비게이션 등의 다양한 응용 분야에서 혁신적인 솔루션을 제시합니다.

### [Chain-of-Thought Tokens are Computer Program Variables](https://arxiv.org/abs/2505.04955)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2505.04955.png)

Vote: 1

Authors: Zhifang Sui, Fangwei Zhu, Peiyi Wang

- ***What's New***: 이 논문에서는 체인-오브-생각(Chain-of-Thought; CoT) 토큰이 실제로 컴퓨터 프로그램의 변수처럼 기능할 수 있는지를 탐구합니다. 저자들은 CoT가 다중 숫자 곱셈 및 동적 프로그래밍과 같은 문제에서 어떻게 중간 결과를 저장하고, 이들이 최종 출력을 제어할 수 있는지를 연구하고 있습니다.
- ***Technical Details***: 이 연구는 CoT 토큰이 중간 결과를 저장하는 데 중요한 역할을 한다는 가설을 세우고, 다중 숫자 곱셈 및 동적 프로그래밍을 포함한 두 가지 합성적 문제를 통해 이를 검증합니다. 실험적으로 CoT 단계가 없을 경우보다 이들을 통해 모델 수행 능력이 개선됨을 확인했으며, 중간 결과를 다른 형태로 저장하더라도 성능에는 영향을 미치지 않음을 발견했습니다.
- ***Performance Highlights***: 실험에 따르면 중간 결과를 저장하는 토큰만으로도 성능 저하 없이 CoT의 기능을 어느 정도 보존할 수 있습니다. 또한, 중간 결과가 최종 결과와 인과적으로 연결되어 있다는 것을 입증하기 위해 무작위로 값을 대체하여 실험하였고, 이로 인해 결과가 어떻게 변화하는지를 관찰했습니다. CoT 토큰은 실제로 프로그램 변수처럼 행동할 수 있으며, 간단한 문제에서는 모델이 더 간단한 '지름길'을 배우는 경향이 있습니다.

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
