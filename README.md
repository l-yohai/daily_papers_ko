# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using GPT-4o.

Thanks to [@AK391](https://github.com/AK391) for great work.

## Daily Papers (2024-10-21)

### [Minimum Tuning to Unlock Long Output from LLMs with High Quality Data as the Key](https://arxiv.org/abs/2410.10210)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.10210.png)

Vote: 2

Authors: ['Jintao Huang', 'Yuze Zhao', 'Yunlin Mao', 'Yingda Chen', 'Daoze Zhang', 'Xingjun Wang']

- ***What's New***: 이 연구는 고품질 데이터(High Quality Data)를 결합한 최소 튜닝(Minimum Tuning) 기법을 통해 LLM에서 긴 출력을 확보하는 방법을 제시합니다. 연구는 인간과의 정렬을 바탕으로 한 모델(Human-aligned model)을 출발점으로 사용하여 소수의 고품질 데이터 샘플로도 모델의 성능을 향상시킬 수 있음을 보여줍니다.
- ***Technical Details***: 연구진은 LongBench-Write 벤치마크를 사용하여 모델의 성능을 평가하고 Output Length Score 𝑆𝐿와 Output Quality Score 𝑆𝑄를 적용했습니다. LongWriter-6K 데이터셋의 666개 고품질 데이터 샘플을 활용하여 필터링하고, 인간과의 정렬 모델에서 시작해 작은 양의 고품질 데이터를 통해 효율적인 튜닝을 수행합니다.
- ***Performance Highlights***: 고품질 데이터를 사용하여 작은 양(약 3.74%)의 훈련 데이터로도 원래의 LongWriter 모델과 비슷한 성능을 달성했습니다. 튜닝된 모델은 요구된 길이의 텍스트 출력을 더 잘 따라가며, 이는 인간 정렬 모델에서 출발해 비용 효율적인 튜닝 접근법을 가능하게 합니다.

### [LoLDU: Low-Rank Adaptation via Lower-Diag-Upper Decomposition for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2410.13618)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13618.png)

Vote: 4

Authors: ['Yujia Wu', 'Ran Ran', 'Yiming Shi', 'Jiwei Wei', 'Chengwei Sun', 'Shiyuan He', 'Yang Yang']

- ***What's New***: LoLDU는 Lower-Diag-Upper(LDU) 분해를 활용한 새로운 파라미터 효율적인 미세 조정(Parameter-Efficient Fine-Tuning; PEFT) 방법입니다. 이 방법은 기존의 PEFT 방법들이 필요로 하는 것보다 훨씬 적은, 0.00025%에 불과한 파라미터 수로도 모델 성능을 유지할 수 있는 혁신적인 접근 방식을 제시합니다.
- ***Technical Details***: LoLDU는 LDU 분해를 통해 메트릭스를 초기화하여, 초점이 LDU 분해의 대각 행렬을 최적화하는 데 맞춰져 있습니다. 이 방법은 직교성을 유지해 사전 훈련된 지식을 보존하며 일반화를 강화하고, 각 변환의 확장 요소를 최적화하는 추론기법을 사용합니다.
- ***Performance Highlights***: LoLDU는 다양한 모델 아키텍처와 과제 유형에 대한 종합적인 실험을 통해 그 효과와 다목적성을 입증했습니다. 실험 결과, LoLDU는 RoBERTa, LLaMA2-7B 등 여러 모델과 데이터셋에서 매우 적은 수의 학습 가능한 파라미터로도 기존의 풀 파인튜닝 성능과 비슷한 결과를 보였습니다.

### [MuVi: Video-to-Music Generation with Semantic Alignment and Rhythmic Synchronization](https://arxiv.org/abs/2410.12957)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.12957.png)

Vote: 3

Authors: ['Zhou Zhao', 'Ziang Zhang', 'Xize Cheng', 'Ruiqi Li', 'Shengpeng Ji', 'Siqi Zheng']

- ***What's New***: MuVi는 비디오의 시각적 콘텐츠와 음악을 생성하는 혁신적인 프레임워크로서, 시각적 의미 정렬(semantic alignment)과 리듬 동기화(rhythmic synchronization) 문제를 효과적으로 해결합니다. 특히 콘트라스티브 뮤직-비주얼 사전학습(contrastive music-visual pre-training) 스킴을 도입하여 음악 구문의 주기성을 기반으로 시각적 동기화를 보장합니다.
- ***Technical Details***: MuVi는 비디오의 문맥적, 시간적 관련성을 추출하기 위해 시각 어댑터(visual adaptor)를 사용합니다. 이 특징들은 비디오와 잘 조화되는 음악을 생성하기 위해 사용되며, 무작위 음악-비디오 쌍을 사용하여 음 -비 동기화를 강조하고 패널티를 부여하는 콘트라스티브 사전학습을 적용하였습니다. 음악 생성기 자체는 흐름 매칭(flow-matching) 기반으로 설계되어 있으며, 비자연스런 방법(Non-autoregressive)을 사용하여 스타일과 장르를 제어할 수 있습니다.
- ***Performance Highlights***: 실험 결과 MuVi는 오디오 품질과 시간 동기화 면에서 뛰어난 성능을 보였으며, 특히 비트 히트 스코어(Beats Hit Score)와 시맨틱 동기화(SIM)에서 측정된 수치를 통해 음악의 리듬과 비디오의 동작이 잘 맞물려 있음을 확인했습니다. 또한 맞춤형 음악 스타일을 실시간으로 생성할 수 있는 인컨텍스트 학습(in-context learning) 능력도 입증되었습니다.

### [A Comparative Study on Reasoning Patterns of OpenAI's o1 Model](https://arxiv.org/abs/2410.13639)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13639.png)

Vote: 12

Authors: ['Wangchunshu Zhou', 'Tuney Zheng', 'Jialong Wu', 'Siwei Wu', 'Jian Yang', 'Xinrun Du', 'Minghao Liu', 'Zhaoxiang Zhang', 'Qunshu Lin', 'Junbo Zhao', 'Zhongyuan Peng', 'Chenghua Lin', 'Yizhi Li', 'Jiachen Ma', 'Ge Zhang', 'Wenhao Huang', 'J. H. Liu']

- ***What's New***: 이 논문은 OpenAI의 o1 모델이 보여주는 독특한 추론 패턴을 다양한 추론 벤치마크에서 비교하여 분석한 최초의 연구입니다. o1은 기존 추론 능력을 크게 향상시켰으며, 특히 수학 및 코딩 작업에서 월등한 성능을 보였습니다. 또한, o1 모델의 Test-time Compute 방법을 활용한 다양한 접근법을 기존 메소드들과 비교하여 연구하였습니다.
- ***Technical Details***: o1 모델은 Best-of-N, Step-wise BoN, Agent Workflow, Self-Refine과 같은 Test-time Compute 방법들과 비교되었으며, GPT-4o를 백본으로 사용하여 일반적인 추론 벤치마크에 대한 성능을 테스트했습니다. 벤치마크로는 HotpotQA, Collie, USACO, AIME가 사용되었으며, 이들은 LLMs가 잘 다루지 못하는 문제의 난이도를 반영하기 위해 데이터 필터링을 통해 강화되었습니다.
- ***Performance Highlights***: o1 모델은 대부분의 벤치마크에서 최고의 성능을 기록하였으며, 특히 수학 및 코드 작업에서 CoT(Chain of Thought) 기반 접근법을 활용하여 뛰어난 성과를 보였습니다. Agent Workflow 방법은 모든 작업에서 상당한 성능 향상을 보였으나, 특히 Collie와 같은 엄격한 형식 요구사항이 있는 작업에서는 BoN과 Step-wise BoN의 성능이 크게 제한되었습니다. o1의 중요한 추론 패턴으로는 시스템 분석(Systematic Analysis), 메소드 재사용(Method Reuse), 분할 정복(Divide and Conquer) 등이 있으며, 이는 o1의 성능 향상에 중요한 역할을 했습니다.

### [MedMobile: A mobile-sized language model with expert-level clinical capabilities](https://arxiv.org/abs/2410.09019)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.09019.png)

Vote: 7

Authors: ['Eric Karl Oermann', 'Daniel Alexander Alber', 'Krithik Vishwanath', 'Jaden Stryker', 'Anton Alaykin']

- ***What's New***: MedMobile은 모바일 디바이스에서도 실행 가능한 3.8억 파라미터 크기의 모바일 크기 언어 모델로, 전문가 수준의 클리니컬 기능을 제공합니다. 이 모델은 MedQA(USMLE)에서 75.7%의 점수를 기록하며, 이는 의사 합격 기준(약 60%)을 초과하고, 자체 크기보다 100배 큰 모델들의 성능에 근접합니다.
- ***Technical Details***: MedMobile은 phi-3-mini라는 오픈 소스 3.8B 파라미터 모델을 의료 분야 데이터로 파인튜닝하여 개발되었습니다. 인간 전문가 및 GPT-4와 교과서에서 생성한 합성 데이터를 사용해 모델의 성능을 높였으며, Chain-of-Thought(CoT), 앙상블(ensembling), 및 지도 학습 파인튜닝(SFT)이 성능 향상에 크게 기여하였습니다. 반면, Retrieval Augmented Generation(RAG)은 큰 개선을 보이지 않았습니다.
- ***Performance Highlights***: MedMobile은 MedQA에서 75.7%의 정확도를 기록하며, 10B 미만의 파라미터 공간 내에서 현존 최고 기록을 가진 UltraMedical 8B(76.1%)에 거의 근접한 성능을 발휘합니다. 총 9개의 평가 과제 중 6개에서 UltraMedical 8B를 능가하거나 동일한 성능을 보여 주며, 5B 미만 파라미터 공간에서 최초로 USMLE 스타일의 문제를 통과한 모델입니다.

### [MoH: Multi-Head Attention as Mixture-of-Head Attention](https://arxiv.org/abs/2410.11842)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.11842.png)

Vote: 18

Authors: ['Shuicheng Yan', 'Bo Zhu', 'Peng Jin', 'Li Yuan']

- ***What's New***: MoH는 Transformer 모델의 핵심인 다중 헤드 주의 메커니즘을 개선하여 효율성을 높이는 새로운 접근 방식입니다. MoH는 주의 헤드를 Mixture-of-Experts(MoE) 메커니즘 내의 '전문가'로 취급하여, 각 토큰이 적절한 주의 헤드를 선택함으로써 추론 효율성을 높입니다.
- ***Technical Details***: MoH는 다중 헤드 주의를 MoE 메커니즘과 통합합니다. MoH는 여러 주의 헤드와 각 토큰에 대해 최상위 K개의 헤드를 활성화하는 라우터로 구성됩니다. 표준 다중 헤드 주의의 합산을 가중 합산으로 대체하여 주의 메커니즘의 유연성을 증가시켰습니다.
- ***Performance Highlights***: MoH는 여러 모델 구조의 다양한 실험에서 다중 헤드 주의보다 50%~90%의 헤드만을 사용하여 더 나은 성능을 발휘했습니다. 예를 들어, MoH-ViT-B는 ImageNet-1K 분류 벤치마크에서 84.9%의 Top-1 정확도를 기록했으며, 이는 75%의 주의 헤드만 사용한 것입니다. 이는 MoH가 다중 헤드 주의의 유망한 대안임을 보여줍니다.

### [DreamVideo-2: Zero-Shot Subject-Driven Video Customization with Precise Motion Control](https://arxiv.org/abs/2410.13830)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13830.png)

Vote: 20

Authors: ['Haonan Qiu', 'Hangjie Yuan', 'Yingya Zhang', 'Rui Zhao', 'Zhizhong Huang', 'Feng Liu', 'Yutong Feng', 'Yujie Wei', 'Shiwei Zhang', 'Hongming Shan', 'Jiaxin Ye', 'Xiang Wang']

- ***What's New***: DreamVideo-2는 제로샷 기법을 활용하여 테스트 시간에 추가 미세 조정 없이 주제 및 모션 궤적을 사용자 지정하여 비디오를 생성할 수 있는 새로운 비디오 커스터마이제이션 프레임워크입니다. 이는 주제를 학습하기 위해 레퍼런스 어텐션(Reference Attention)을 도입하고, 경계 상자에서 파생된 박스 마스크를 사용하여 정확한 모션 제어를 달성하는 마스크 기반 모션 모듈을 설계했습니다.
- ***Technical Details***: DreamVideo-2는 주제 이미지와 경계 상자 시퀀스를 입력으로 하여 비디오를 생성합니다. 모션 제어를 위해 스파티오템포럴 인코더(Spatiotemporal Encoder)와 스페이셜 컨트롤넷(Spatial ControlNet)으로 구성된 마스크 기반 모션 모듈을 사용합니다. 주제 학습과 모션 제어의 균형을 맞추기 위해 혼합 마스크(Blended Masks)와 재조정된 확산 손실(Reweighted Diffusion Loss)을 도입하여 주제 학습과 모션 제어의 기여도를 차별화합니다.
- ***Performance Highlights***: DreamVideo-2는 텍스트 정렬, 주제 충실도 및 모션 제어 정밀도 측면에서 최신 방법보다 우수합니다. 주어진 시험 데이터 세트에서 mIoU 및 CD 메트릭에서 크게 우수한 성과를 보이며, 사용자 지정 비디오 생성의 실제 응용 가능성을 개선합니다.

### [PopAlign: Diversifying Contrasting Patterns for a More Comprehensive Alignment](https://arxiv.org/abs/2410.13785)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13785.png)

Vote: 17

Authors: ['Wangchunshu Zhou', 'Kang Zhu', 'Shawn Wang', 'Jiaheng Liu', 'Ke Xu', 'Jie Fu', 'Zekun Moore Wang', 'Wenhao Huang']

- ***What's New***: PopAlign은 대형 언어 모델(LLMs)의 정렬(Alignment)을 보다 포괄적으로 수행하기 위해 새로운 대조 패턴(Diversified Contrastive Patterns)을 도입합니다. 이는 모형의 반응을 인간의 선호도에 맞게 조절할 때, 기존의 한정된 대조 패턴에 비해 확장된 대조 패턴을 통해 보다 포괄적인 정렬을 제공합니다.
- ***Technical Details***: PopAlign 프레임워크는 프롬프트(Prompt), 모델(Model), 파이프라인(Pipeline) 수준에서 여섯 가지 대조 전략을 통합하여 별도의 피드백 레이블링 절차 없이 선호도 데이터를 증진시킵니다. 이 여섯 가지 전략은 프리픽스 대조(Prefix Contrast), 데몬 대조(Demon Contrast), 이리시브 대조(Elicitive Contrast), NParam 대조(Number-of-Parameter Contrast), 리더보드 대조(Leaderboard Contrast), 리파인 대조(Refine Contrast)를 포함합니다.
- ***Performance Highlights***: PopAlign은 기존 방법보다 다양한 작업에서 선호도 데이터의 포괄적인 정렬을 달성하며, 특히 이리시브 대조(Elicitive Contrast) 전략을 통한 성능 향상이 눈에 띕니다. 또한 다른 선호도 최적화 알고리즘이나 모델에 따라 서로 다른 효과를 나타내며, 안전성 또는 전반적인 정렬 효과 측면에서 다양한 강점을 보였습니다.

### [Do LLMs Have Political Correctness? Analyzing Ethical Biases and Jailbreak Vulnerabilities in AI Systems](https://arxiv.org/abs/2410.13334)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13334.png)

Vote: 9

Authors: ['Haebin Seong', 'Isack Lee']

- ***What's New***: 이 논문은 대형 언어 모델(Large Language Models; LLMs)의 안전 정렬(safety alignment)이 의도하거나 의도하지 않은 편향을 어떻게 유발하고 이러한 편향이 '탈옥(jailbreak)' 공격을 통해 악용될 수 있는지를 조사합니다. 특히, GPT-4o 모델에서 비이진 키워드와 시스젠더 키워드 간의 20%, 백인과 흑인 키워드 간의 16% 차이를 보이는 탈옥 성공률을 발표합니다.
- ***Technical Details***: PCJailbreak는 LLMs의 안전 기준에 맞춰 의도적으로 주입된 편향이 어떻게 고의적으로 악용되어 해로운 콘텐츠를 생성할 수 있는지를 분석합니다. 이 연구는 다양한 인구 통계 및 사회 경제적 그룹의 키워드를 사용하여 탈옥 성공률을 평가하고 이러한 편향을 완화시키는 방어 기법인 PCDefense를 제안합니다.
- ***Performance Highlights***: GPT-4o와 같은 최신 모델에서, 비특권 그룹에 대한 키워드가 특권 그룹에 비해 상당한 탈옥 성공률을 보이며, 이를 통해 의도적인 편향이 생성되었음을 알 수 있습니다. PCDefense는 추가 추론 단계 없이 방어 프롬프트를 활용하여 성능 개선을 보여주며, LLAMA3 모델은 다른 모델에 비해 더 낮은 탈옥 성공률을 보여 안전성에 초점을 맞추고 있음을 나타냅니다.

### [MobA: A Two-Level Agent System for Efficient Mobile Task Automation](https://arxiv.org/abs/2410.13757)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13757.png)

Vote: 29

Authors: ['Yansi Li', 'Kai Yu', 'Yixiao Wang', 'Zichen Zhu', 'Liangtai Sun', 'Kunyao Lan', 'Yixuan Jiang', 'Lu Chen', 'Hao Tang', 'Hao Zhou', 'Situo Zhang']

- ***What's New***: MobA는 Mobile Task Automation을 위한 이중 수준 에이전트 시스템으로, 멀티모달 대형 언어 모델(Multimodal Large Language Models; MLLMs)을 이용하여 사용자 명령의 이해와 계획 기능을 강화합니다. 이 시스템은 고급 글로벌 에이전트(Global Agent; GA)가 사용자 명령을 해석하며, 로컬 에이전트(Local Agent; LA)는 화면 제어를 담당합니다.
- ***Technical Details***: MobA의 구조는 Global Agent와 Local Agent로 나뉩니다. Global Agent는 사용자의 명령을 해석하고, 기억 모듈(Memory Module)에서 관련 경험을 검색하여 계획 모듈(Plan Module)을 통해 할 일을 하위 작업으로 나눕니다. Local Agent는 하위 작업과 기억을 받아 적절한 작업을 선택하고 실행합니다. 이중 반영 메커니즘(Double-reflection mechanism)을 사용해 하위 목표의 완료 여부를 사전에 확인하고, 작업 후에도 성공을 검증합니다.
- ***Performance Highlights***: MobA는 여러 애플리케이션과 다양한 난이도의 50가지 실제 작업에서 테스트된 MobBench에서 Milestone Score Rate에서 66.2%를 기록하며, 기존의 모바일 에이전트 시스템을 앞서는 성과를 보였습니다. 이는 MLLM 기반 모바일 에이전트가 복잡한 작업을 효율적으로 관리할 수 있음을 입증합니다.

### [TransAgent: Transfer Vision-Language Foundation Models with Heterogeneous Agent Collaboration](https://arxiv.org/abs/2410.12183)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.12183.png)

Vote: 2

Authors: ['Kunchang Li', 'Yiwei Guo', 'Yu Qiao', 'Shaobin Zhuang', 'Yali Wang']

- ***What's New***: TransAgent는 이종 에이전트 협업을 통해 비전-언어 기반 모델(Vision-Language Foundation Models)의 전이를 가능하게 하는 새로운 프레임워크입니다. 이 프레임워크는 고유한 방식으로 고립된 에이전트의 지식을 통합하여, 멀티소스 지식 증류(Multi-Source Knowledge Distillation)를 통해 CLIP와 같은 모델의 일반화 능력을 향상시킵니다. 이는 학습 과정 중 추가적인 추론 비용 없이 이질적 에이전트 11개와 유연하게 협업합니다.
- ***Technical Details***: TransAgent는 비전, 언어, 멀티모달 연구에서 이질적 에이전트 11개를 활용하여 CLIP와 같은 모델에 지식을 전수합니다. 비전 에이전트는 MAE, DINO, SAM, ViTDet와 같이 세밀한 이미지 모델링에 중점을 두고 있으며, 언어 에이전트로는 GPT-3, Vicuna가 사용됩니다. 멀티모달 에이전트는 BLIP2, Shikra와 같은 텍스트-이미지, 이미지-텍스트 변환을 수행하여 비전-언어 정렬을 촉진합니다.
- ***Performance Highlights***: TransAgent는 11개의 시각 인식 데이터셋에서 SOTA(State-of-the-Art) 성능을 달성하며, CoOp를 평균 약 10%, 유로SAT에서 20% 이상 능가합니다. 이는 대규모 도메인 변화를 포함하는 데이터셋에서도 효과적으로 일반화 성능을 보입니다.

### [WorldCuisines: A Massive-Scale Benchmark for Multilingual and Multicultural Visual Question Answering on Global Cuisines](https://arxiv.org/abs/2410.12705)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.12705.png)

Vote: 21

Authors: ['Haryo Akbarianto Wibowo', 'Muhammad Farid Adilazuarda', 'En-Shiun Annie Lee', 'Garry Kuwanto', 'Jan Christian Blaise Cruz', 'Yutong Wang', 'Genta Indra Winata', 'Natasha Santosa', 'Adam Nohejl', 'Yinxuan Gui', 'Maria Angelica Riera Machin', 'Marina Zhukova', 'Ashmari Pramodya', 'Aulia Adila', 'David Ifeoluwa Adelani', 'Stephanie Yulia Salim', 'Patrick Amadeus Irawan', 'Taro Watanabe', 'Jan Wira Gotama Putra', 'Ubaidillah Ariq Prathama', 'Michael Anugraha', 'Holy Lovenia', 'Shi-Xiong Zhang', 'Anirban Das', 'Ching Lam Cheng', 'Derry Tanti Wijaya', 'Daud Abolade', 'Yi Zhou', 'Frederikus Hudi', 'Ayu Purwarianti', 'Afifa Amriani', 'Anar Rzayev', 'Rio Alexander Audino', 'Fariz Ikhwantri', 'Junho Myung', 'Nedjma Ousidhoum', 'Rifki Afina Putri', 'Candy Olivia Mawalim', 'Emmanuele Chersoni', 'Peerat Limkonchotiwat', 'Lucky Susanto', 'Bryan Wilie', 'Alice Oh', 'Raj Dabre', 'Samuel Cahyawijaya', 'Alham Fikri Aji', 'Chong-Wah Ngo', 'Shogo Okada', 'Enrico Santus', 'David Anugraha', 'Hanyang Zhao']

- ***What's New***: WORLDCUISINES는 다국어 및 다문화 시각 기반 언어 이해를 위한 대규모 벤치마크로, 30개의 언어와 방언에 걸쳐 1백만 개 이상의 텍스트-이미지 쌍을 포함하는 시각적 질문 답변(VQA) 데이터셋을 제공합니다. 이 벤치마크는 음식 이름과 그 기원을 예측하는 과제를 포함하며, 이는 현재까지 가장 큰 다문화 VQA 벤치마크입니다.
- ***Technical Details***: WORLDCUISINES는 VQA 데이터셋인 WC-VQA와 세계 요리에 대한 가용 지식 베이스(WC-KB)로 이루어져 있습니다. WC-KB는 전 세계의 2,414가지 요리를 다루고, 6,045장의 이미지와 메타데이터를 포함합니다. VQA 데이터 생성은 WC-KB를 기반으로 하고, 요리 이미지와 관련된 질문 및 맥락을 통해 이루어집니다. 질문은 30개의 언어로 번역되며, 트레이닝 세트와 두 가지 크기의 평가 세트가 준비되어 있습니다.
- ***Performance Highlights***: WC-VQA 데이터셋에 대한 성능 테스트 결과, 저작권 모델 GPT-4o는 여러 설정에서 최고의 성능을 보였으며, 반복적인 맥락이 포함된 경우 성능이 향상되었습니다. 그러나 관객을 혼란스럽게 하는 역방향 맥락에서 모델은 성능 저하를 겪었습니다. 실험 결과에 따르면, VLM은 정확한 맥락에서는 잘 작동하지만, 지역 특산 요리나 저자원 언어에는 어려움을 겪고 있다는 점을 보여주었습니다.

### [Roadmap towards Superhuman Speech Understanding using Large Language Models](https://arxiv.org/abs/2410.13268)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13268.png)

Vote: 32

Authors: ['Haizhou Li', 'Yuhao Zhang', 'Benyou Wang', 'Fan Bu', 'Qun Liu', 'Xidong Wang']

- ***What's New***: 이 논문은 대형 언어 모델(LLMs)를 활용하여 인간의 수준을 넘는 음성 이해(Speech Understanding)를 목표로 하는 로드맵(Roadmap)을 제시합니다. 이는 자동 음성 인식(ASR)부터 음향 지식과 비어문 정보의 통합을 통해 고급 과제를 수행할 수 있는 초인간 모델(Superhuman Models)까지 5단계로 나뉘어 있으며, 이를 평가하기 위한 SAGI 벤치마크도 소개합니다.
- ***Technical Details***: 제안된 로드맵은 음성 LLMs를 통해 비어문적 정보(Non-Semantic Information)와 세계 지식(World Knowledge)을 유지하는 것을 강조합니다. 음성 LLMs는 기본적으로 ASR 기능을 가지고 있으면서 점진적으로 비어문적 특징을 이해하고, 추상 음향 지식(Abstract Acoustic Knowledge)을 활용함으로써 다양한 복잡한 작업을 수행할 수 있도록 발전해야 합니다. 또한, 음성 이해 능력을 평가하기 위한 다양한 작업이 포함된 SAGI 벤치마크가 설계되었습니다.
- ***Performance Highlights***: 현재의 음성 LLMs는 일부 영역에서는 인간 성능을 초과할 수 있지만, 비어문적 정보의 인식과 이해에서 여전히 제한점을 보입니다. 특히, 추상 음향 지식의 결핍은 더 높은 수준의 작업 수행에서의 공통 병목 현상으로 확인되었습니다. GPT-4o는 음성 지침을 따르는 데 이점이 있지만, 여전히 개선의 여지가 많이 남아 있습니다.

### [Fluid: Scaling Autoregressive Text-to-image Generative Models with Continuous Tokens](https://arxiv.org/abs/2410.13863)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13863.png)

Vote: 27

Authors: ['Tianhong Li', 'Kaiming He', 'Yonglong Tian', 'Chen Sun', 'Lijie Fan', 'Siyang Qin', 'Michael Rubinstein', 'Deqing Sun', 'Yuanzhen Li']

- ***What's New***: Fluid는 새로운 랜덤 순서(autoregressive model) 텍스트-이미지 생성 모델로, 연속 토큰(continuous tokens)을 사용하여 이미지 품질을 크게 향상시킨다는 것을 보여주었습니다. Fluid 10.5B 모델은 MS-COCO 30K에서 새로운 제로샷 FID(Frechet Inception Distance) 6.16을 달성하였으며, GenEval 벤치마크에서 0.69의 전체 점수를 기록하며 최신 성능을 자랑합니다.
- ***Technical Details***: Fluid는 텍스트-이미지 생성에서 모델의 성능을 지속적으로 검토하여 두 가지 주요 요인을 탐구합니다: 모델이 연속 토큰을 사용하는지 여부와 토큰이 랜덤 순서로 생성되는지 여부입니다. Fluid는 연속 토큰을 사용하여 정보 손실을 최소화하고, BERT와 유사한 양방향(attention mechanism)을 통해 랜덤 순서로 토큰을 생성하여 글로벌 구조를 리얼타임으로 조정할 수 있는 이점을 가집니다.
- ***Performance Highlights***: Validation 손실이 모델 크기와 비례하여 로그스케일로 감소하며, Fluid 모델의 성능은 연속 토큰을 사용한 랜덤 순서 모델에서 가장 뛰어남을 보여주었습니다. GenEval에서 Fluid 모델은 3.1B 파라미터까지 보여지는 모든 평가 지표에서 지속적인 개선을 보여주며, 최신 모델들과 비교해 뛰어난 이미지-텍스트 정렬 및 생성 품질을 제공합니다.

### [γ-MoD: Exploring Mixture-of-Depth Adaptation for Multimodal Large Language Models](https://arxiv.org/abs/2410.13859)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13859.png)

Vote: 5

Authors: ['Jiayi Ji', 'Zhiqiang Shen', 'Yiyi Zhou', 'Xiaoshuai Sun', 'Rongrong Ji', 'Yaxin Luo', 'Gen Luo']

- ***What's New***: γ-MoD는 다중 모드 대형 언어 모델(Multimodal Large Language Models; MLLMs)의 계산 효율성을 높이기 위한 새로운 혼합 깊이(Mixture-of-Depths; MoDs) 적응 전략을 제안합니다. 이 전략은 많은 중복 레이어를 활성화 토큰(Activated Tokens) 관점에서 식별하여 대체하는 모습을 보여줍니다.
- ***Technical Details***: γ-MoD는 주의 지도 계층의 순위(ARank)를 도입하여 토큰의 중복성을 측정하고, 이를 통해 중복되는 레이어를 MoD 레이어로 대체합니다. 공유된 시각-언어 라우터(Shared Vision-Language Router)와 마스크된 경로 학습(Masked Routing Learning)을 통해 MoD의 효과를 극대화합니다. 이러한 설계를 통해 90% 이상의 밀집 레이어를 MoD 레이어로 변환할 수 있습니다.
- ***Performance Highlights***: γ-MoD는 기존의 MLLMs에서 평균 성능 저하가 1.5%에 불과한 상태로 트레이닝 시간은 31%, 추론 시간은 53.2%까지 감소시켰습니다. 이 접근 방식은 다양한 MLLM 구조와 파라미터 규모에 대한 일반화 능력을 보여줍니다.

### [JudgeBench: A Benchmark for Evaluating LLM-based Judges](https://arxiv.org/abs/2410.12784)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.12784.png)

Vote: 27

Authors: ['Kyle Montgomery', 'Chenguang Wang', 'Ion Stoica', 'Sijun Tan', 'Siyuan Zhuang', 'Raluca Ada Popa', 'William Y. Tang', 'Alejandro Cuadron']

- ***What's New***: JudgeBench는 LLM 기반 심사위원(LLM-based Judges)을 평가하기 위한 새로운 벤치마크입니다. 이 벤치마크는 지식, 추론, 수학, 코딩과 같은 다양한 어려운 반응 쌍에 대해 LLM 기반 심사위원의 능력을 평가하도록 설계되었습니다.
- ***Technical Details***: JudgeBench는 기존 데이터셋을 변환하여 객관적인 정확성을 반영하는 선호 레이블을 갖춘 어려운 반응 쌍(challenging response pairs)으로 만드는 새로운 파이프라인을 활용합니다. 이 데이터셋은 GPT-4o를 비롯한 강력한 모델을 사용하여 생성된 350개의 반응 쌍으로 구성되며, 각 쌍은 하나의 객관적으로 정확한 응답과 미묘한 오류를 포함한 하나의 잘못된 응답을 포함합니다. GPT-4o를 통해 생성된 반응에 대한 객관적인 평가를 지원합니다.
- ***Performance Highlights***: JudgeBench는 기존 벤치마크보다 훨씬 더 어려운 도전 과제를 제시하며, GPT-4o와 같은 강력한 모델도 무작위 추측보다 약간 우수하게 성능을 보였습니다. Skywork의 Fine-tuned 심사위원은 57.43%의 정확도를 기록한 반면, 최신 OpenAI 모형인 o1-preview는 75.43%의 정확도를 달성했습니다. 이러한 결과는 현재 LLM 기반 심사위원이 더욱 발전된 AI 모델을 평가할 때의 한계를 보여주며, 향후 연구 방향성을 제시합니다.

### [Can MLLMs Understand the Deep Implication Behind Chinese Images?](https://arxiv.org/abs/2410.13854)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13854.png)

Vote: 6

Authors: ['Feiteng Fang', 'Yuelin Bai', 'Shiwen Ni', 'Yifei Zhang', 'Guangzeng Han', 'Xi Feng', 'Yiming Liang', 'Bingli Wang', 'Jinchang Hou', 'Qinrui Li', 'Chenghua Lin', 'Wenhao Huang', 'Min Yang', 'Ge Zhang', 'Xingwei Qu', 'Ziqiang Liu', 'Chenhao Zhang', 'Jiaheng Liu', 'Xinrun Du', 'Qixuan Zhao', 'Kaixin Deng']

- ***What's New***: 이 연구에서는 다중 모드 대형 언어 모델(Multimodal Large Language Models; MLLMs)의 중국 이미지 해석 능력을 평가하기 위해 중국 이미지 암시 이해 벤치마크(CII-Bench)를 소개합니다. CII-Bench는 중국 전통 예술 및 인터넷에서 수집된 이미지를 포함하여 중국 문화에 대해 심도 있는 이해를 요구합니다.
- ***Technical Details***: CII-Bench는 698개의 이미지와 800개의 다중 선택 질문으로 구성되어 있으며, 삶, 예술, 사회, 정치, 환경, 중국 전통 문화 등 6가지 도메인에서 다양한 유형의 이미지를 포함합니다. 이 데이터셋은 30명의 학부생에 의해 수집 및 주석이 달렸으며, GPT-4o 모델을 사용한 평가도 포함되었습니다.
- ***Performance Highlights***: 시험 결과 인간의 평균 정확도는 78.2%에 달했지만, 가장 높은 성능의 MLLM은 64.4%만을 기록했습니다. 특히, 중국 전통 문화 이미지에서 MLLMs의 성능은 부족한 이해를 보여주었습니다. 모델들은 감정 힌트를 포함할 때 더 나은 성능을 보였으나, 여전히 감정 이해에 어려움을 겪고 있습니다.

### [Toward Guidance-Free AR Visual Generation via Condition Contrastive Alignment](https://arxiv.org/abs/2410.09347)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.09347.png)

Vote: 2

Authors: ['Jun Zhu', 'Huayu Chen', 'Hang Su', 'Peize Sun']

- ***What's New***: 이 연구에서는 새로운 기법인 Condition Contrastive Alignment (CCA)을 소개하여 AutoRegressive (AR) 모델의 시각적 생성 품질을 개선하고, 기존의 Classifier-Free Guidance (CFG)의 필요성을 제거합니다. CCA는 사전 학습된 모델을 미세 조정함으로서 동일한 표본 분포를 맞추고, CFG와 비교해 샘플링 비용을 절반으로 줄입니다.
- ***Technical Details***: Condition Contrastive Alignment (CCA)는 이미지를 위한 양성 조건과 음성 조건을 활용하여 대조 학습으로 조건 간 차이를 학습합니다. 이는 사전 훈련된 모델을 미세 조정하는 단순한 손실 함수를 사용하여 수행되며, 새로운 데이터셋이 필요하지 않습니다. 실험적으로 확인된 결과, CCA는 사전 학습된 데이터셋으로 단 1 에포크의 훈련만으로 CFG와 동등한 성능을 제공합니다.
- ***Performance Highlights***: CCA는 모든 검증된 모델에서 '샘플링 없는 상태(Without guidance)'에서도 높은 성능을 보였습니다. 예를 들어, LlamaGen-L 모델은 FID 점수가 19.07에서 3.41로 개선되었고, IS 점수는 64.3에서 288.2까지 상승했습니다. 이러한 성과는 샘플링 비용이 기존 방식의 절반으로 줄어든 상태에서 일어납니다.

### [A Unified View of Delta Parameter Editing in Post-Trained Large-Scale Models](https://arxiv.org/abs/2410.13841)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13841.png)

Vote: 14

Authors: ['Yaojie Lu', 'Xianpei Han', 'Qiaoyu Tang', 'Keming Lu', 'Hongyu Lin', 'Bowen Yu', 'Le Sun', 'Le Yu']

- ***What's New***: 이 논문은 대규모 사전 학습 모델의 후속 학습 단계에서 델타 매개변수(Delta Parameters)를 편집하는 방법에 대한 통합 관점을 제시합니다. 저자들은 리만 합(Riemann Sum) 근사를 기반으로 손실 함수의 변화를 분석하여 다양한 후 수정 작업이 모델 성능에 미치는 영향을 설명합니다.
- ***Technical Details***: 리만 합 근사(Riemann Sum Approximation)를 사용하여 델타 매개변수 편집의 손실 변화를 체계적으로 분석했습니다. DARE와 DELLA-Merging 같은 방법은 무작위 삭제와 리스케일을 통해 손실을 제로로 유지하여 경쟁력 있는 성능을 유지합니다. BitDelta, Twin-Merging 등은 양자화(Quantization) 및 저순위 근사(Low-rank Approximation)로 성능이 감소할 수 있습니다. EXPO는 델타 매개변수를 적절히 확장하여 성능을 향상시킵니다.
- ***Performance Highlights***: 실험 결과, VeriT와 LLaMA 3, Mistral 등의 시각 및 언어 모델에서 이론적 분석이 뒷받침되었습니다. EXPO와 같은 방법은 모델의 조정 데이터에서 손실을 줄여 더 잘 정렬된 모델을 생성합니다. DARE의 경우 무손실의 성능을 유지하면서 델타 매개변수를 99%까지 삭제할 수 있습니다.

### [SBI-RAG: Enhancing Math Word Problem Solving for Students through Schema-Based Instruction and Retrieval-Augmented Generation](https://arxiv.org/abs/2410.13293)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13293.png)

Vote: 3

Authors: ['Prakhar Dixit', 'Tim Oates']

- ***What's New***: SBI-RAG는 수학 문장형 문제(Math Word Problem; MWP)를 보다 효율적으로 해결하기 위해 스키마 기반 지시(Schema-Based Instruction; SBI)와 검색을 통한 생성(Retrieval-Augmented Generation)을 통합한 혁신적인 프레임워크입니다. 이 접근 방식은 큰 언어 모델(Large Language Model; LLM)을 활용하여 단계별 추론을 강조하고, 문제 해결 과정에서 스키마를 통해 해결 방식을 유도함으로써 학생들의 논리적 사고를 향상시킵니다.
- ***Technical Details***: SBI-RAG는 DistilBERT를 이용한 스키마 분류기를 통해 주어진 문제에 대한 적절한 스키마를 예측하며, Ollama Llama 3.1 LLM을 활용해 문맥과 스키마별 프롬프트를 생성합니다. 검색-강화 생성(RAG) 프레임워크를 활용하여 관련 문서를 검색, 문맥 내 정보를 바탕으로 구조화된 답변을 제공합니다. 이 시스템은 GSM8K 데이터셋을 통해 평가되며, 문제 해결의 논리성을 평가하기 위해 '추론 점수(reasoning score)'라는 새로운 지표를 도입합니다.
- ***Performance Highlights***: SBI-RAG 프레임워크는 GPT-4와 GPT-3.5 Turbo와 비교하여 응답의 추론 품질에서 더 높은 점수를 기록하였습니다. SBI-RAG의 추론 점수가 0.588인 반면, GPT-4는 0.491, GPT-3.5 Turbo는 0.290으로 나타났습니다. 통계적 검정 결과, SBI-RAG는 두 모델에 비해 유의미하게 높은 성능을 보였습니다. 이는 교육적 맥락에서 스키마 기반 추론이 LLM 단독으로 생성한 응답보다 더 나은 품질의 추론을 제공할 수 있음을 시사합니다.

### [BenTo: Benchmark Task Reduction with In-Context Transferability](https://arxiv.org/abs/2410.13804)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13804.png)

Vote: 19

Authors: ['Tianyi Zhou', 'Hongyu Zhao', 'Ming Li', 'Lichao Sun']

- ***What's New***: BenTo는 대형 언어 모델(LLMs)의 평가 비용을 줄이기 위해 벤치마크 작업을 효율적으로 축소하는 방법을 제안합니다. 특히, 문맥 학습(in-context learning; ICL)을 사용하여 두 작업 간의 전이 가능성을 추정하는 비용 효율적이고 훈련이 필요 없는 방식인 맥락 내 전이 가능성(In-Context Transferability; ICT)을 도입하여 벤치마크 축소를 가능하게 합니다.
- ***Technical Details***: BenTo는 벤치마크 작업의 대표적인 하위 집합을 선택하기 위해 시설 위치 함수(facility location function)를 최적화하는 방식으로 작업의 전이 가능성과 관련성을 평가합니다. ICT를 사용하여 두 작업 사이의 전이 가능성을 추정하고, 이를 통해 현대 LLM 벤치마크(e.g., MMLU, FLAN)를 5%로 축소하면서도 원래 벤치마크와 비교해 평가 오차를 4% 이하로 유지할 수 있습니다.
- ***Performance Highlights***: BENTO 방법은 57개의 MMLU 작업 중 단 3개의 작업 선택 시에도 평균 97%의 평가 정확도를 달성하며, GPT-4와 같은 기존 방법보다 낮은 비용에서 더 정확한 평가 결과를 제공합니다. 이는 LLM의 성능 평가에 있어 전이 가능성에서 얻은 정보가 어떻게 효율적으로 활용될 수 있는지를 보여줍니다.

### [MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models](https://arxiv.org/abs/2410.13085)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13085.png)

Vote: 19

Authors: ['Kangyu Zhu', 'Peng Xia', 'Huaxiu Yao', 'Linjun Zhang', 'Tianze Wang', 'James Zou', 'Sheng Wang', 'Weijia Shi', 'Haoran Li']

- ***What's New***: MMed-RAG는 의료 대형 시각-언어 모델(Med-LVLMs)의 사실성을 개선하기 위해 설계된 다목적 멀티모달 RAG 시스템입니다. 의료 영상의 다양한 도메인을 보다 효과적으로 처리하기 위해 도메인 인식 정보 검색 메커니즘과 적응형 검색 문맥 선택 방법을 소개하고, 입증된 RAG 기반의 선호 조정 전략을 통합하여 RAG 과정이 충분히 일반적이고 신뢰할 수 있도록 만들며, 검색 문맥을 도입할 때 정렬을 크게 개선합니다.
- ***Technical Details***: MMed-RAG는 의료 영상의 도메인에 따른 최적의 검색기를 선택하는 도메인 인식 검색 메커니즘을 도입합니다. 각 입력 의료 이미지를 적응형으로 분석하여 해당하는 검색 모델을 선택하고, 검색된 문맥의 수를 선택하는 적응형 보정 접근 방식을 포함합니다. 끝으로, 교차 모달 정렬 및 모델과 참사이의 정렬을 개선하기 위해 RAG 기반의 선호 조정 방법을 도입합니다. 이러한 선호 쌍은 모델이 입력된 의료 이미지를 활용하지 않고 응답을 생성하지 않도록 촉구하여 교차 모달 정렬을 개선하고, 검색된 문맥을 이해하도록 장려하여 전반적인 정렬을 개선하는 두 가지 목표를 달성하기 위해 설계되었습니다.
- ***Performance Highlights***: MMed-RAG는 다섯 가지 의료 데이터셋(영상의학, 안과, 병리학)에 걸친 의료 VQA 및 보고서 생성 작업에서 Med-LVLMs의 사실 정확성을 평균 43.8％ 향상시킬 수 있음을 실험적으로 보여줍니다. 이러한 실험 결과는 제안된 구성 요소의 효과를 더욱 입증하고, 미세 조정 문제를 해결하기 위한 이론적 분석을 지원합니다. 또한, MMed-RAG는 Med-LVLMs의 크로스 모달 정렬과 전반적인 사실 정렬을 크게 개선하여 의료 멀티모달 작업에서 확장성과 효과성을 보여줍니다.

### [Harnessing Webpage UIs for Text-Rich Visual Understanding](https://arxiv.org/abs/2410.13824)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13824.png)

Vote: 23

Authors: ['Yifan Song', 'Xiang Yue', 'Graham Neubig', 'Yuxiao Qu', 'Wenhu Chen', 'Wai Lam', 'Tianyue Ou', 'Junpeng Liu', 'Chenyan Xiong']

- ***What's New***: 웹페이지 UI를 활용한 텍스트가 풍부한 시각 이해(Harnessing Webpage UIs for Text-Rich Visual Understanding) 연구에서는 다양한 멀티모달 태스크와 UI 레이아웃을 아우르는 MultiUI 데이터셋을 소개합니다. 이 데이터셋은 1백만 개의 웹사이트에서 추출된 730만 개의 샘플로 구성되어 있으며, 웹 UI 데이터의 광범위한 적용 가능성을 강조합니다.
- ***Technical Details***: MultiUI 데이터셋은 웹페이지의 접근성 트리(Accessibility Tree)를 활용하여 일반 멀티모달 명령어를 생성하고, UI 스크린샷과 결합하여 멀티모달 모델을 학습시킵니다. 데이터 수집 파이프라인은 웹 사이트 스크래핑, 사이트 선별, 태스크 추출 및 명령어 생성의 4단계로 나뉩니다. 다양한 디바이스 타입과 윈도우 크기 변형을 소개하여 모델의 강건성을 높였습니다.
- ***Performance Highlights***: MultiUI로 학습된 모델이 VisualWebBench에서 최대 48% 성능 향상을 보였으며, Mind2Web 데이터셋에서 엘리먼트 정확도를 19.1% 향상시켰습니다. 이는 웹 UI 데이터가 시각적 이해를 향상시키고 비UI 도메인에서도 뛰어난 성능을 발휘할 수 있음을 보여줍니다.

### [Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2410.13848)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13848.png)

Vote: 24

Authors: ['Xingchao Liu', 'Zhiyu Wu', 'Wen Liu', 'Chong Ruan', 'Ping Luo', 'Xiaokang Chen', 'Yiyang Ma', 'Zhenda Xie', 'Chengyue Wu', 'Zizheng Pan', 'Xingkai Yu']

- ***What's New***: Janus는 멀티모달 이해 및 생성을 위한 통합적인 프레임워크를 제공합니다. 기존의 단일 비주얼 인코더에 의존하던 방법을 개선하여 이해와 생성을 위해 시각적 인코딩을 분리했습니다. 이로 인해 시각적 인코더의 다양한 역할 간의 충돌을 완화하고 프레임워크의 유연성을 높였습니다.
- ***Technical Details***: Janus는 멀티모달 이해와 생성을 위한 독립적인 두 개의 시각적 인코딩 경로를 도입했습니다. 동일한 트랜스포머 아키텍처에 의해 통합됩니다. 멀티모달 이해에서는 SigLIP( 시그니파이어 이미지-텍스트 학습) 인코더를 사용하여 고차원적 의미적 특징을 추출하고, 생성에서는 VQ 토크나이저를 사용하여 이미지를 이산 아이디로 변환합니다. 세 가지 단계로 훈련이 진행되며, 각 단계는 시각과 언어 요소의 개념적 연결을 강화, 통합 프리트레이닝, 및 지도학습 튜닝으로 구성됩니다.
- ***Performance Highlights***: Janus는 다양한 멀티모달 이해 및 생성 벤치마크에서 기존 모델보다 우수한 성능을 발휘했습니다. MMBench, SEED-Bench, POPE와 같은 이해 벤치마크에서 탑 성과를 기록했으며, MSCOCO-30K 및 GenEval에서 FID 8.53과 61%의 정확도를 달성하여 시각 생성 능력에서도 뛰어난 결과를 보여줍니다.

### [Long-LRM: Long-sequence Large Reconstruction Model for Wide-coverage Gaussian Splats](https://arxiv.org/abs/2410.12781)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.12781.png)

Vote: 4

Authors: ['Kai Zhang', 'Zexiang Xu', 'Fujun Luan', 'Sai Bi', 'Chen Ziwen', 'Yicong Hong', 'Li Fuxin', 'Hao Tan']

- ***What's New***: Long-LRM은 긴 사진 시퀀스에서 대형 장면을 재구성할 수 있는 일반화 가능한 3D 가우시안 복원 모델(Long-sequence Large Reconstruction Model; Long-LRM)을 제안합니다. 이는 기존의 3D 가우시안 스플래팅(3D Gaussian Splatting; 3D GS)보다 두 배 이상의 효율성을 가지며 32개의 입력 이미지를 1.3초 만에 처리합니다.
- ***Technical Details***: Long-LRM의 아키텍처는 최신 마마바2 블록(Mamba2 blocks)과 클래식 트랜스포머 블록(Transformer blocks)에 기반하여 설계되어 있습니다. 이는 보다 많은 토큰을 처리할 수 있게 해주며, 토큰 병합(token merging)과 가우시안 가지치기(Gaussian pruning) 단계로 효율성과 품질의 균형을 맞춥니다. 이 모델은 960×540 해상도의 32개의 소스 이미지로부터 데이터 손실을 방지하기 위해 패치화된 형식으로 작동하며, 변환을 통해 픽셀 정렬된 가우시안 프리미티브(Gaussian primitives)를 회귀합니다.
- ***Performance Highlights***: Long-LRM은 대규모 장면 데이터셋인 DL3DV-140과 Tanks and Temples에서 성능을 평가하였으며, 3D GS의 최적화 기반 접근법에 비해 두 배 이상의 효율성을 보여주었습니다. 960×540 해상도에서 높은 수준의 렌더링 품질을 유지하면서도 2차 복잡도보다 빠른 예측을 가능케 하여 1.3초 내에 전체 장면을 피드포워드 방식으로 재구성합니다.

### [Remember, Retrieve and Generate: Understanding Infinite Visual Concepts as Your Personalized Assistant](https://arxiv.org/abs/2410.13360)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13360.png)

Vote: 7

Authors: ['Changsheng Li', 'Jiaming Han', 'Haoran Hao', 'Yu-Feng Li', 'Xiangyu Yue']

- ***What's New***: 이 논문에서는 사용자의 개인 맞춤형 비주얼 개념을 이해하고 생성할 수 있는 다중 모달 LLM(Multimodal Large Language Model)의 새로운 개인화 프레임워크인 RAP(Retrieval Augmented Personalization)을 소개합니다. 일반 MLLM을 시작점으로 사용자의 데이터베이스를 통해 실시간 개념 편집이 가능하며 무한한 시각적 개념을 확장할 수 있는 개인 맞춤형 비서로 전환합니다.
- ***Technical Details***: RAP 프레임워크는 크게 세 가지 단계로 구성됩니다. (a) 기억하기(Remember): 사용자의 이름, 아바타 등 사용자와 관련된 정보를 저장하기 위한 키-값 데이터베이스를 설계합니다. (b) 검색하기(Retrieve): 대화가 시작되면 멀티모달 검색기를 통해 데이터베이스에서 관련 정보를 검색합니다. (c) 생성하기(Generate): 입력 질의와 검색된 개념의 정보를 MLLM에 삽입하여 개인 맞춤형 지식 증강 응답을 생성합니다. 추가로 대규모 데이터 수집 파이프라인과 개인화 훈련용 데이터를 제작하여 MLLM을 훈련했습니다.
- ***Performance Highlights***: RAP-MLLM은 개인 맞춤형 이미지 캡션 생성, 질문 응답 및 시각적 인식 등의 다양한 작업에서 뛰어난 유연성과 생성 품질을 발휘합니다. 본 모델들은 LLaVA(2023), Phi3-V(2024)와 같은 대규모 데이터셋을 미리 학습함으로써 추가 미세 조정 없이 무한한 시각적 개념으로 일반화할 수 있습니다.

### [MixEval-X: Any-to-Any Evaluations from Real-World Data Mixtures](https://arxiv.org/abs/2410.13754)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13754.png)

Vote: 67

Authors: ['Bo Li', 'Yifan Song', 'Kaichen Zhang', 'Michael Shieh', 'David Junhao Zhang', 'Kabir Jain', 'Deepanway Ghosal', 'Fuzhao Xue', 'Xiang Yue', 'Mahir Shah', 'Yang You', 'Jinjie Ni', 'Zian Zheng']

- **What's New**: MixEval-X는 실세계 데이터 혼합을 통해 다양한 입력과 출력 모달리티에 대한 평가를 표준화하고 최적화하기 위해 설계된 최초의 벤치마크입니다. 이 연구는 다중 모달 평가가 일관되지 않은 표준과 편향성을 극복하고, 실제 세계의 과제 분포에 맞추어 효과적으로 일반화할 수 있도록 하는 방법론을 제안합니다.
- **Technical Details**: MixEval-X는 여덟 가지의 입력-출력 모달리티 조합을 포함하며, 각 모달리티는 다중 모달 이해(MMU), 다중 모달 생성(MMG), 에이전트 작업으로 분류됩니다. 웹 사용자 질의 탐지 파이프라인을 통해 다양한 입력-출력 모달리티에 걸쳐 잘 분포된 실제 세계의 질의를 수집하고, 다중 모달 벤치마크 풀을 활용하여 이러한 분포에 맞추어 새로운 벤치마크 분포를 재구축합니다. 또한, 적응-수정 파이프라인을 통해 MMG와 에이전트 작업을 자동으로 생성합니다.
- **Performance Highlights**: MixEval-X의 메타평가 결과, 벤치마크 샘플이 실제 세계 과제 분포와 잘 일치하고, 크라우드 소싱 기반의 실제 세계 평가와 강한 상관관계를 나타내며 효율적임을 보여주었습니다. 특히, Image2Text 평가 결과는 사용자 주도 평가와 최대 0.98의 상관관계를 보여줍니다.

### [Open Materials 2024 (OMat24) Inorganic Materials Dataset and Models](https://arxiv.org/abs/2410.12771)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.12771.png)

Vote: 4

Authors: ['Misko Dzamba', 'Muhammed Shuaibi', 'Meng Gao', 'Luis Barroso-Luque', 'Brandon M. Wood', 'Xiang Fu', 'Ammar Rizvi', 'C. Lawrence Zitnick', 'Zachary W. Ulissi']

- ***What's New***: Open Materials 2024 (OMat24)는 Meta의 FAIR팀이 개발한 대규모 무기 재료 데이터셋으로, 1억 1,000만 건 이상의 밀도 함수 이론(DFT; Density Functional Theory) 연산을 포함하고 있습니다. 새로운 EquiformerV2 모델은 최첨단 성능을 보여주며, 이 데이터셋은 AI를 활용한 재료 과학 혁신을 가속화하는 데 중요한 자원이 될 것입니다.
- ***Technical Details***: OMat24 데이터셋은 다양한 무기 벌크 재료를 대상으로 하는 DFT 단일점 계산, 구조 최적화, 분자 동역학 궤적을 포함합니다. EquiformerV2 모델은 OMat24 데이터셋과 MPtraj, Alexandria 데이터셋을 사용하여 사전 훈련했으며, DeNS 프로토콜을 사용해 데이터 증강을 통해 모델의 예측 정확도를 높였습니다.
- ***Performance Highlights***: EquiformerV2 모델은 Matbench Discovery 벤치마크에서 F1 스코어 0.916 및 에너지 오차 20 meV/atom을 기록하며 가장 높은 성능을 보였습니다. 이는 기존의 독점 모델과 비교해도 경쟁력 있는 수준입니다. OMat24와 MPtrj 데이터셋의 사전 학습은 모델 성능을 크게 향상시켰습니다.

### [MagicTailor: Component-Controllable Personalization in Text-to-Image Diffusion Models](https://arxiv.org/abs/2410.13370)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13370.png)

Vote: 2

Authors: ['Guangyong Chen', 'Donghao Zhou', 'Jiaze Wang', 'Jiancheng Huang', 'Xiaowei Hu', 'Jinbin Bai', 'Hao Chen', 'Pheng-Ann Heng']

- ***What's New***: MagicTailor는 텍스트에서 이미지로 변환하는 확산 모델(Text-to-Image Diffusion Models)을 사용하여 개념의 특정 구성 요소를 재구성할 수 있는 새로운 작업, 구성 요소 제어 기반 개인화를 소개합니다. 이는 시맨틱 오염(Semantic Pollution)과 시맨틱 불균형(Semantic Imbalance)을 극복하기 위해 설계되었습니다.
- ***Technical Details***: MagicTailor는 Dynamic Masked Degradation(DM-Deg)과 Dual-Stream Balancing(DS-Bal)이라는 두 가지 핵심 기술을 사용합니다. DM-Deg는 원하지 않는 시각적 시맨틱스를 동적으로 혼란스럽게 만들어 시맨틱 오염을 방지하고, DS-Bal은 목표 개념 및 구성 요소의 시각적 시맨틱스를 균형 있게 학습할 수 있도록 보장합니다.
- ***Performance Highlights***: MagicTailor는 구성 요소 제어 기반 개인화 작업에서 최첨단(SOTA) 성능을 달성하며, 이는 정량적 및 정성적 실험을 통해 입증되었습니다. 사용자 연구에서도 MagicTailor가 텍스트 정렬, 아이덴티티 충실도, 생성 품질에서 높은 점수를 받은 것으로 나타났습니다.

### [VidPanos: Generative Panoramic Videos from Casual Panning Videos](https://arxiv.org/abs/2410.13832)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13832.png)

Vote: 9

Authors: ['Tali Dekel', 'Forrester Cole', 'Erika Lu', 'Shiran Zada', 'Jingwei Ma', 'Aleksander Holynski', 'Roni Paiss', 'Michael Rubinstein', 'Brian Curless']

- ***What's New***: VidPanos는 일상적으로 촬영된 패닝 비디오로부터 파노라마 비디오를 생성하는 혁신적인 시스템입니다. 이 방법은 일반적인 동적 장면의 비디오 파노라마를 만들어 냅니다.
- ***Technical Details***: VidPanos는 입력 비디오를 공간-시간 아웃페인팅(space-time outpainting) 문제로 변환하여 전체 파노라마 비디오를 완성합니다. 이를 위해 일반적으로 공간-시간적 맥락 창에 제한되는 생성 비디오 모델(generative video model)을 조정합니다. 이 과정에서 Lumiere와 Phenaki의 생성 비디오 모델을 사용하여 공간적 집합기법(spatial aggregation techniques)과 거친-to-세밀한(coarse-to-fine) 합성 접근 방식을 채택합니다.
- ***Performance Highlights***: Synthesized panoramas는 다양한 동적 장면에서 정지 장면 요소를 충실히 복원하며, 이동 물체를 상당히 그럴듯한 위치에 렌더링합니다. 예를 들어, 합성 여행 비디오 실험에서 이동하는 사람 또는 스케이트보더가 현실적인 동작을 보여주었으며, 이는 기존의 선형 보간이나 optick flow 기반 방법보다 월등한 성능을 입증합니다.

### [Diffusion Curriculum: Synthetic-to-Real Generative Curriculum Learning via Image-Guided Diffusion](https://arxiv.org/abs/2410.13674)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13674.png)

Vote: 1

Authors: ['Tianyi Zhou', 'Shweta Bhardwaj', 'Yijun Liang']

- ***What's New***: Diffusion Curriculum(DisCL)는 이미지 교차 확산(image-guided diffusion)을 통해 합성 데이터와 실제 데이터 간의 스펙트럼을 생성하여, 적응형 교육(Adaptive Learning)을 가능하게 하는 새로운 생성형 커리큘럼 학습 접근법입니다. DisCL은 합성 이미지의 이미지 가이던스 수준을 조정하여 학습 단계마다 적절한 난이도와 다양성의 데이터를 제공합니다.
- ***Technical Details***: DisCL은 이미지 가이던스의 강도를 조절하여 합성 데이터와 실제 데이터를 결합한 광범위한 스펙트럼을 생성합니다. 코드베이스에서는 Stable Diffusion 모델이 사용되며, 합성 데이터 생성 시마다 이미지 가이던스의 레벨을 다양하게 조절해 실제 데이터와의 시각적 유사성을 맞춥니다. 데이터 생성 단계에서는 고난이도 샘플을 식별하고, 이러한 샘플을 바탕으로 가장 적합한 합성 데이터 가이던스 수준을 설정하여 모든 학습 단계에서 모델 성능을 최적화합니다.
- ***Performance Highlights***: DisCL은 iWildCam 데이터셋에 대해 ID 및 OOD 매크로 정확도를 각각 2.7%와 2.1% 향상시켰습니다. ImageNet-LT의 경우, 기본 모델의 테일 클래스 정확도를 4.4%에서 23.64%로 개선시켜 전체 클래스 정확도는 4.02% 증가했습니다. 이는 DisCL이 기존까지의 어려운 데이터 학습 문제를 효과적으로 해결함을 보여줍니다.

### [Movie Gen: A Cast of Media Foundation Models](https://arxiv.org/abs/2410.13720)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13720.png)

Vote: 70

Authors: ['Xiaoliang Dai', 'Ali Thabet', 'Mannat Singh', 'Samyak Datta', 'Simone Parmeggiani', 'Haoyu Ma', 'Tara Fowler', 'Zecheng He', 'Siddharth Bhattacharya', 'Xi Yin', 'Jonas Kohler', 'Matthew Yu', 'Yue Zhao', 'Simran Motwani', 'Andros Tjandra', 'Fraylie Nord', 'Edgar Schonfeld', 'Matt Le', 'Zijian He', 'Sara K. Sampson', 'Kunpeng Li', 'Peizhao Zhang', 'Dhruv Choudhary', 'Breena Kerr', 'Bowen Shi', 'Luya Gao', 'Dimitry Vengertsev', 'Yaniv Taigman', 'Licheng Yu', 'Rashel Moritz', 'Carleigh Wood', 'Rohit Girdhar', 'Tao Xu', 'Yuval Kirstain', 'Baishan Guo', 'Arun Mallya', 'Luxin Zhang', 'Shikai Li', 'Tingbo Hou', 'Karthik Sivakumar', 'Ji Hou', 'Samaneh Azadi', 'Elliot Blanchard', 'Vladan Petrovic', 'Ann Lee', 'Ching-Yao Chuang', 'Quentin Duval', 'Yen-Cheng Liu', 'John Hoffman', 'Jeff Liang', 'Dingkang Wang', 'Chih-Yao Ma', 'Guan Pang', 'Animesh Sinha', 'Albert Pumarola', 'Amit Zohar', 'Wei-Ning Hsu', 'Jialiang Wang', 'Kaolin Fire', 'Tianhe Li', 'Yi-Chiao Wu', 'Peter Vajda', 'Sharadh Ramaswamy', 'Boris Araya', 'Mary Williamson', 'David Yan', 'Felix Juefei-Xu', 'Kiran Jagadeesh', 'Cen Peng', 'Markos Georgopoulos', 'Steve Fine', 'Sai Saketh Rambhatla', 'Shelly Sheynin', 'Adam Polyak', 'Roshan Sumbaly', 'Yaqiao Luo', 'Apoorv Vyas', 'Andrew Brown', 'Sam Tsai', 'Ishan Misra', 'Artsiom Sanakoyeu', 'Mitesh Kumar Singh', 'Sanyuan Chen', 'Lawrence Chen', 'Ce Liu', 'Geet Sethi', 'Sean Bell', 'Yuming Du']

- ***What's New***: Movie Gen은 고품질 1080p HD 비디오와 동기화된 오디오를 생성하는 미디어 기초 모델 세트를 선보입니다. 사용자 이미지에 기반한 개인화된 비디오 생성 및 정밀 지침 기반 비디오 편집을 포함한 추가 기능도 제공합니다. 이 모델은 텍스트 기반 비디오 합성, 비디오 개인화, 비디오 편집, 비디오-오디오 생성 및 텍스트-오디오 생성에서 최신 상태를 설정합니다.
- ***Technical Details***: Movie Gen Video 및 Audio 모델은 각각 30B 및 13B 매개변수 변환기로, 73K 비디오 토큰의 최대 컨텍스트 길이로 훈련됩니다. 모델은 아키텍처, 잠재 공간, 훈련 목표 및 레시피, 데이터 큐레이션, 평가 프로토콜, 병렬화 기술, 추론 최적화 등 여러 기술 혁신을 통해 대규모 미디어 생성 모델의 트레이닝에서 사전 훈련 데이터, 모델 크기 및 훈련 계산의 이점을 얻습니다.
- ***Performance Highlights***: Movie Gen은 Runway Gen3, LumaLabs, OpenAI Sora 등 기존 상업 시스템을 능가하며, 비디오 개인화 및 정밀 비디오 편집에서도 새롭게 능력을 제공하여 이전 작업들을 초과합니다. Movie Gen Audio는 PikaLabs 및 ElevenLabs와 같은 상업 시스템들을 사운드 효과 생성, 음악 생성 및 오디오 확장에서 능가합니다.

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
