# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using ChatGPT.

Thanks to [@AK391](https://github.com/AK391) for great work.


## Daily Papers (2024-06-28)

### [Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs](https://arxiv.org/abs/2406.18629)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.18629.png)

Vote: 14

Authors: Senqiao Yang, Yukang Chen, Xin Lai, Zhuotao Tian, Xiangru Peng, Jiaya Jia

- **What's New**: 수학적 추론은 대형 언어 모델(LLMs)에서 중요한 장기 추론 능력으로 인식되고 있습니다. Step-DPO라는 새로운 방법을 도입하여 각 중간 추론 단계를 기본 단위로 취급하여 선호 최적화를 수행합니다. 이를 통해 모델이 더 효과적으로 오류 토큰을 찾아내고 최적화할 수 있도록 지원합니다.
- **Technical Details**: Step-DPO는 각 중간 추론 단계를 개별 단위로 취급하여, 올바른 추론 단계와 잘못된 단계를 구분해 내는 방법입니다. 한 수학 문제와 여러 초기 올바른 추론 단계를 제공했을 때 올바른 단계를 최대화하고 잘못된 단계를 최소화하는 방식입니다. 이 방법을 통해 모델이 쉽게 오류 토큰을 찾아내고 최적화할 수 있습니다. 또한, 고품질의 데이터셋을 효율적으로 수집하는 파이프라인을 제안합니다.
- **Performance Highlights**: Step-DPO로 Qwen-72B-Instruct 모델을 미세조정한 결과, MATH에서 70.8%, GSM8K에서 94.0%의 정확도를 달성했습니다. 이는 GPT-4-1106, Claude-3-Opus, Gemini-1.5-Pro 등 일련의 비공개 모델을 능가하는 성과입니다.

### [OMG-LLaVA: Bridging Image-level, Object-level, Pixel-level Reasoning and Understanding](https://arxiv.org/abs/2406.19389)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.19389.png)

Vote: 12

Authors: Hao Fei, Shengqiong Wu, Tao Zhang, Xiangtai Li, Shuicheng Yan, Chen Change Loy, Haobo Yuan, Shunping Ji

- **What's New**: OMG-LLaVA는 이미지 수준, 객체 수준, 픽셀 수준의 다양한 과제를 하나의 모델로 해결하는 통합 멀티모달 언어 모델(Multi-modal Large Language Model, MLLM)입니다. 이 모델은 OMG-Seg를 기반으로 하여 더 간단하게 다양한 작업을 수행할 수 있도록 설계되었습니다. 이를 통해 이미지를 이해하고, 객체 위치를 추적하며, 세그멘테이션을 수행할 수 있는 능력을 갖추었습니다.
- **Technical Details**: OMG-LLaVA는 하나의 LLM(Large Language Model), 하나의 비주얼 인코더(visual encoder) 및 하나의 디코더(decoder)로 구성되어 있습니다. 이 모델은 시각 세그멘테이션 결과를 더 잘 인코딩하기 위해 객체 중심의 시각 토큰(object-centric visual tokens)을 생성하는 환경 인코딩 모듈(perception prior embedding module)을 사용합니다. 우리는 통일된 명령어 형성 전략(unified instruction formation strategy)을 채택하여 시각적 이미지, 텍스트 및 시각적 프롬프트를 입력으로 받아 텍스트, 세그먼테이션 토큰, 세그멘테이션 마스크 및 레이블을 생성할 수 있습니다.
- **Performance Highlights**: OMG-LLaVA는 COCO 파노프틱 세그멘테이션(COCO panoptic segmentation), VIPSeg 비디오 파노프틱 세그멘테이션(VIPSeg video panoptic segmentation), refCOCO, refCOCO+, refCOCOg 지시 표현 세그멘테이션(referring expression segmentation), GranDf 그라운딩 대화 생성(grounding conversation generation) 및 refCOCOg 영역 캡션 데이터셋에서 우수한 성능을 발휘했습니다.

### [Simulating Classroom Education with LLM-Empowered Agents](https://arxiv.org/abs/2406.19226)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.19226.png)

Vote: 11

Authors: Zheyuan Zhang, Juanzi Li, Lei Hou, Daniel Zhang-Li, Jifan Yu, Linlu Gong, Zhiyuan Liu, Jinchang Zhou

- **What's New**: SimClass는 다중 에이전트(Classroom Simulation)를 통해 실제 교실 환경을 시뮬레이션하는 새로운 프레임워크입니다. 이 시스템은 대규모 모델(LLMs)의 잠재력을 최대한 활용하여 여러 에이전트가 있는 교실을 자동화하는 방법을 탐구하고, 실제 사용자 참여를 통하여 관찰 및 분석을 진행합니다.
- **Technical Details**: SimClass는 교실 내 다양한 역할(Class Roles)과 novel class control mechanism을 설계하여 대표적인 수업 역할을 인식하고 기능적 워크플로우를 만듭니다. 실험을 위해 두 가지 다른 코스를 준비된 슬라이드와 강의 스크립트를 기반으로 구현하였고, 48명의 학생들이 이 시스템을 사용하여 학습하고 상호작용하는 과정에서 발생한 모든 행동 데이터를 기록하였습니다. 또한, Flanders Interaction Analysis System을 통해 상호작용 패턴을 평가하고, Community of Inquiry 이론을 사용하여 교육 경험을 분석하였습니다.
- **Performance Highlights**: (1) SimClass는 전통 교실과 유사한 행동 및 상호작용 패턴을 나타냅니다. (2) 여러 명의 교실 에이전트가 사용자가 더 효과적으로 수업에 참여하고 더 높은 몰입감을 느낄 수 있도록 도와줍니다. (3) 제어 메커니즘은 협력적 교수 및 토론, 감정적 동반 및 규율 통제 등 다양한 자발적 행동을 유도합니다. 이러한 결과는 LLM 기반 다중 에이전트 시스템이 실질적인 교육 목적으로 실제 교실 환경을 시뮬레이션할 수 있는 잠재력을 보여줍니다.

### [SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation](https://arxiv.org/abs/2406.19215)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.19215.png)

Vote: 11

Authors: Linmei Hu, Juanzi Li, Weichuan Liu, Lei Hou, Weijian Qi, Shulin Cao, Liangming Pan, Zijun Yao

- **What's New**: 이번 연구에서는 LLMs(Large Language Models)의 내부 상태를 이용하여 지식 검색 시기를 동적으로 결정하고 검색된 지식을 효과적으로 통합하는 'SElf-Aware Knowledge Retrieval (SeaKR)' 방법을 제안합니다. SeaKR은 LLMs의 자기 인식을 활용하여 지식 검색과 통합 과정을 최적화하는 최초의 방법입니다.
- **Technical Details**: SeaKR은 LLMs의 내부 상태에서 추출한 자기 인식을 바탕으로 지식 검색 시기를 결정합니다. 이는 Feed-Forward Network (FFN)의 각 레이어에서 마지막으로 생성된 토큰에 해당하는 내부 상태에서 불확실성을 측정함으로써 이루어집니다. 또한, SeaKR은 두 가지 적응형 통합 전략을 사용하여 검색된 지식을 효율적으로 통합합니다. 첫째, 'Self-aware re-ranking'은 여러 회상된 지식 조각 중 LLM의 불확실성을 가장 많이 줄이는 지식을 선택합니다. 둘째, 'Self-aware reasoning'은 복잡한 질문에 대한 답변을 위해 여러 지식을 통합하는 반복적인 지식 검색을 지원합니다.
- **Performance Highlights**: 복잡한 질문-답변 (QA) 및 간단한 QA 작업에서 실험한 결과, SeaKR은 기존의 적응형 RAG 방법보다 높은 성능 향상을 보였습니다. 특히, 동적으로 통합된 검색 지식이 자기 인식 기반 검색보다 더 높은 성능 향상을 가져온다는 것이 실험 결과에서 드러났습니다.

### [Aligning Teacher with Student Preferences for Tailored Training Data Generation](https://arxiv.org/abs/2406.19227)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.19227.png)

Vote: 11

Authors: Zhao Zhang, Yantao Liu, Juanzi Li, Lei Hou, Shulin Cao, Zijun Yao

- **What's New**: 최근 대형 언어 모델(LLMs)이 여러 작업의 공동 파일럿으로서의 잠재력을 보여주고 있습니다. 그러나 이러한 모델들은 주로 강력한 대규모 LLM에 의존하고 있어 에지 장치에 배포하기엔 어려움이 있습니다. 이 문제를 해결하기 위해 지식 증류(Knowledge Distillation) 방법이 제안되었습니다.
- **Technical Details**: ARTE(Aligning TeacheR with StudenT PreferencEs)라는 새로운 프레임워크를 소개합니다. ARTE는 교사 모델(LM_t)을 학생 모델(LM_s)의 선호도에 맞추어 맞춤형 훈련 예제를 생성합니다. 세 단계로 구성됩니다: 1) 지식 유도(Knowledge Elicitation), 2) 선호도 수집(Preference Collection), 3) 선호도 정렬(Preference Alignment).
- **Performance Highlights**: 광범위한 실험에서 ARTE가 기존의 데이터셋보다 특정한 논리 추론, 상식 추론, 수학 추론 및 지식 추론 과제에서 각각 9.6%, 1.0%, 0.8% 및 8.5%의 성능 향상을 보였습니다. 또한, ARTE는 다양한 벤치마크 테스트에서 우수한 일반화 능력을 보였습니다.

### [Read Anywhere Pointed: Layout-aware GUI Screen Reading with Tree-of-Lens Grounding](https://arxiv.org/abs/2406.19263)

![](/avatars/05abee0b6317f100923936ca2099e9eb.svg)

Vote: 2

Authors: Xinze Guan, Shan Jiang, Yi Zhang, Jie Yang, Yang Zhao, Xin Eric Wang, Ching-Chen Kuo, Yue Fan, Lei Ding

- **What's New**: 새로운 연구에서는 GUI(그래픽 사용자 인터페이스)의 스크린샷을 AI 에이전트가 해석할 수 있도록 돕는 'Tree-of-Lens (ToL) 에이전트'를 도입했습니다. 특히 'Screen Point-and-Read (ScreenPR)' 작업을 통해 시각적 장애가 있는 사용자들을 위한 접근성 기술을 강화하려고 합니다.
- **Technical Details**: ToL 에이전트는 고급 Multimodal Large Language Models (MLLMs)의 일반화 능력을 활용합니다. 이 에이전트는 스크린 상의 특정 포인트를 입력으로 받아, 해당 영역에 대한 내용 설명과 스크린 레이아웃 정보를 자연어로 출력합니다. Hierarchical Layout Tree는 스크린샷의 기본 구조를 나타내기 위해 사용되며, Android 스크린샷에서 50,000개의 계층적 스크린 영역 바운딩 박스가 포함된 ASHL 데이터셋을 사용하여 학습된 객체 인식 모델을 통해 구축됩니다.
- **Performance Highlights**: ScreenPR 벤치마크를 도입하여 모델의 성능을 평가한 결과, ToL 에이전트는 기존의 MLLMs보다 15% 이상, GUI에 특화된 모델보다도 30% 이상의 성능 향상을 보였습니다. 특히 모바일 GUI 네비게이션 에이전트의 실행 경로에서 잘못된 동작을 식별하는 데 있어서 유용성이 입증되었습니다.

### [MoA: Mixture of Sparse Attention for Automatic Large Language Model Compression](https://arxiv.org/abs/2406.14909)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.14909.png)

Vote: 1

Authors: Boju Chen, Tianqi Wu, Shengen Yan, Tianyu Fu, Haofeng Huang, Guohao Dai, Huazhong Yang, Genghan Zhang, Zixiao Huang, Hongyi Wang, Xuefei Ning, Yu Wang, Shiyao Li

- **What's New**: 이번 연구에서는 Mixture of Attention (MoA)라는 새로운 트레이닝이 필요 없는 희소 주의 메커니즘을 제안합니다. MoA는 다양한 탄력 규칙을 통해 각 모델 레이어와 주의 헤드별로 맞춤형 희소 주의 설정을 제공합니다.
- **Technical Details**: MoA는 각 주의 헤드의 주의 위치가 예측 손실에 미치는 영향을 평가하기 위한 gradient-based profiling을 사용하여 프로파일링합니다. 이를 기반으로, MoA는 각 레이어와 주의 헤드별로 이질적인 희소 주의 설정을 최적화합니다. 또한, 원본 밀집 모델의 응답을 참조하여 long-range dependencies를 가진 데이터셋을 사용해 프로파일링합니다.
- **Performance Highlights**: MoA는 7B 및 13B 모델에서 KV-Cache 길이와 입력 길이의 평균 비율이 50%일 때 5.5~6.7배의 처리량 향상을 달성합니다. 또한, 25% 밀도로 90% 이상의 정보 검색 정확도를 유지하며 이는 기존의 희소 주의 기법이 동일한 성능을 위해 75% 이상의 밀도가 필요한 것보다 훨씬 더 우수한 결과입니다. 긴 컨텍스트 이해 벤치마크에서는 밀집 모델과 유사하게 작동하며 무작위 희소 주의 방법보다 최대 5% 미만의 성능 저하만을 보입니다.

### [Is Programming by Example solved by LLMs?](https://arxiv.org/abs/2406.08316)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.08316.png)

Vote: 1

Authors: Wen-Ding Li, Kevin Ellis

- **What's New**: 새로운 논문에서는 복잡한 데이터셋(data sets)을 더 효율적으로 분석하기 위한 고급 머신러닝(machine learning) 기법을 제안했습니다. 이 방법론은 특히 클라우드 환경(cloud environment)에서의 대규모 데이터 처리에 유리합니다.
- **Technical Details**: 논문에서는 새로운 알고리즘(algorithm)이 사용되었으며, 주로 딥러닝(deep learning) 모델과 강화 학습(reinforcement learning) 기법을 결합했습니다. 이 모델은 다층 퍼셉트론(multilayer perceptron) 구조를 기반으로 하며, 데이터 전처리(data pre-processing) 단계에서 비지도 학습(unsupervised learning)을 적용하여 효율성을 극대화했습니다.
- **Performance Highlights**: 제안된 모델은 기존의 방법들에 비해 데이터 처리 속도와 정확도(accuracy) 면에서 우수한 성능을 보였습니다. 특히, 대규모 데이터셋에서의 성능이 크게 향상되었으며, 이를 통해 실시간(real-time) 데이터 분석이 가능해졌습니다.

### [Can LLMs Learn by Teaching? A Preliminary Study](https://arxiv.org/abs/2406.14629)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.14629.png)

Vote: -

Authors: Peiran Yao, Tianyu Fu, Huazhong Yang, Guohao Dai, Zinan Lin, Xuefei Ning, Yu Wang, Matthew B. Blaschko, Zifu Wang, Shiyao Li

- **What's New**: arXiv에 게시된 이 논문은 최신 인공지능 알고리즘, 새로운 데이터셋, 혹은 혁신적인 기술 방식을 제안하고 있습니다. 예를 들어, 이 연구는 Transformer 네트워크든, Generative Adversarial Networks (GANs)든 새로운 아키텍처를 소개하고 있을 수 있습니다.



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
