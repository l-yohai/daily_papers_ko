# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using GPT-4o.

Thanks to [@AK391](https://github.com/AK391) for great work.

## Daily Papers (2024-10-28)

### [Infinity-MM: Scaling Multimodal Performance with Large-Scale and High-Quality Instruction Data](https://arxiv.org/abs/2410.18558)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.18558.png)

Vote: 1

Authors: Yingli Zhao, Jintao Jia, Siyuan Zhou, Yaoqi Liu, Zhenchong Hu, Yixuan Wang, Shuhao Gu, Zhaohu Xing, Zhuoyi Zhang, Guang Liu, Jialing Zhang, Bo-Wen Zhang, Zhou Cao, Dong Liang, Liangdong Wang, Jijie Li, Kevin Yu, Fangxiang Feng, Yulong Ao

- ***What's New***: Infinity-MM은 공개된 VLMs(Vision-Language Models)의 성능을 처음으로 대규모 멀티모달 명령 데이터셋을 사용해 개선한 연구입니다. 이 데이터셋은 까다로운 품질 필터링 및 중복 제거를 통해 강화된 4천만 개 샘플로 구성되어 있으며, Aquila-VL-2B라는 20억 파라미터의 VLM을 훈련하여 동급 모델 중 최신 성능을 달성했습니다.
- ***Technical Details***: Infinity-MM은 공개 소스 VLMs를 기반으로 한 합성 명령 생성 방법을 제안하며, 다양한 질문 생성을 통해 이미지 주석을 세부적으로 진행했습니다. 또한, 데이터 수집 과정에서 400만 개의 이미지 캡션 데이터, 8천 2백만 개의 일반 시각 명령 데이터 등을 활용했습니다. 학습은 단계적으로 어려움을 높이며 진행되었고, 주로 Nvidia A100에 최적화된 멀티모달 데이터 로더를 사용하였습니다.
- ***Performance Highlights***: Aquila-VL-2B는 여러 분야의 벤치마크에서 최신 성능을 달성하였으며, 특히 General Visual Question Answering과 Knowledge & Mathematical Reasoning 분야에서 두드러진 성과를 보였습니다. 합성 데이터가 성능 향상에 큰 기여를 하였음을 입증하였고, 이는 데이터 다양성과 크기의 확장이 모델 성능 향상에 효과적임을 시사합니다.

### [FasterCache: Training-Free Video Diffusion Model Acceleration with High Quality](https://arxiv.org/abs/2410.19355)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.19355.png)

Vote: 2

Authors: Zhenyu Yang, Ziwei Liu, Zhengyao Lv, Junhao Song, Kwan-Yee K. Wong, Chenyang Si, Yu Qiao

- ***What's New***: 이번 연구에서는 FasterCache라는 새로운 트레이닝 불필요 전략을 소개하였습니다. 이 전략은 고품질 비디오 생성 속도를 크게 가속화하기 위한 것으로, 기존 캐시 기반 방법의 부족한 점을 개선하고, 시간단계 내 조건과 비조건 특징의 중복성을 활용하여 비디오 품질 저하 없이 추론 속도를 대폭 증가시킵니다.
- ***Technical Details***: FasterCache는 동적 특징 재사용 전략(Dynamic Feature Reuse Strategy)을 사용하여 특징의 구별성과 시간적 연속성을 유지합니다. 또한 CFG-Cache는 조건과 비조건 출력 간의 잔차를 저장하여 재사용 시 고주파 및 저주파 요소를 동적으로 강화하여 추론 속도를 높입니다. 이 방법은 다양한 비디오 확산 모델(Open-Sora 1.2, Open-Sora-Plan, Latte, CogVideoX, Vchitect-2.0)에서 실험적으로 평가되었습니다.
- ***Performance Highlights***: Vchitect-2.0 모델에서는 FasterCache를 통해 1.67배의 속도 향상을 달성했으며, 영상 품질은 기준모델과 거의 동일합니다(VBench: 기준모델 80.80% → FasterCache 80.84%). 기존 방식보다 추론 속도와 비디오 생성 품질 면에서 탁월한 성능을 보여주었습니다.

### [ROCKET-1: Master Open-World Interaction with Visual-Temporal Context Prompting](https://arxiv.org/abs/2410.17856)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.17856.png)

Vote: 11

Authors: Kewei Lian, Anji Liu, Shaofei Cai, Xiaojian Ma, Yitao Liang, Zihao Wang, Zhancun Mu

- ***What's New***: ROCKET-1은 시각-시간적 맥락 프롬프팅(Visual-Temporal Context Prompting)을 활용하여 개방형 세계(Open-World)에서 상호작용을 최적화하는 새로운 방법을 제시합니다. 이는 시각적 언어 모델(VLMs)과 정책 모델 간의 새로운 의사소통 프로토콜로, 과거와 현재의 관찰에서 객체 세분화를 사용하여 환경 정책 상호작용을 안내합니다.
- ***Technical Details***: ROCKET-1은 SAM-2 모델의 실시간 객체 추적 기능을 통한 시각적 관찰과 세분화 마스크를 바탕으로 행동을 예측하는 저수준 정책(low-level policy)입니다. 이는 VLMs의 시각-언어 추론 능력을 최대한 발휘하게 하여 공간 이해에 강하게 의존하는 복잡한 창의적 작업들을 해결할 수 있게 합니다. 또한, 부분적으로 관찰 가능한 환경에서 중요한 의존성을 나타내기 위해 Transformer 모델을 사용하여 관찰 간의 종속성을 모델링합니다.
- ***Performance Highlights***: Minecraft 실험에서, ROCKET-1은 시각-시간적 맥락 프롬프팅이 구현된 환경에서, 이전에는 달성할 수 없었던 작업을 수행할 수 있음을 보여주어 그 효율성을 입증하였습니다. 이 방법은 장기 목표 과제 해결능력에서도 기존 방법들보다 우수한 성능을 보였습니다.

### [Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning](https://arxiv.org/abs/2410.19290)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.19290.png)

Vote: 1

Authors: Yang Zhang, Yujian Liu, Tommi Jaakkola, Shiyu Chang

- ***What's New***: 이 연구에서는 대형 언어 모델(LLMs)의 착각 현상을 줄이기 위한 새로운 미세 조정(fine-tuning) 전략을 제안합니다. 'PREREQ-TUNE'이라는 전략을 통해 지식과 기술을 분리하여 LLM이 학습할 때 두 가지가 얽히지 않도록 하여 사실성을 향상시킬 수 있습니다.
- ***Technical Details***: PREREQ-TUNE은 두 단계로 구성됩니다: 'Prerequisite Learning' 단계에서는 LLM이 필요한 지식을 학습하도록 설정하고, 이후 'Supervised Fine-Tuning(SFT)' 단계에서는 지식 LoRA(Low-Rank Adaptation) 모듈을 동결하고 기술 LoRA만을 훈련하여 착각을 줄입니다. 또한, 가상의 합성 데이터(fictitious synthetic data)를 활용해 LLM의 출력이 내부 지식에 기반해 더 잘 고정되도록 합니다.
- ***Performance Highlights***: PREREQ-TUNE은 짧은 질문 응답(QA) 및 장문 생성 작업에서 기존의 베이스라인보다 LLM의 사실성을 향상시키는 것으로 나타났습니다. 예를 들어, QA 작업에서 정확도가 47.91%로 향상되었으며, 장문 생성 작업에서도 우수한 성능을 보였습니다. 이는 LLM의 출력이 내부 지식에 더욱 잘 기반하도록 학습할 수 있음을 보여줍니다.

### [Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback](https://arxiv.org/abs/2410.19133)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.19133.png)

Vote: 0

Authors: Noah A. Smith, Faeze Brahman, Yizhong Wang, Yanai Elazar, Hannaneh Hajishirzi, Pradeep Dasigi, Valentina Pyatkin, Sachin Kumar, Lester James V. Miranda

- ***What's New***: 이 연구는 인간과 AI 피드백의 혼합을 통해 학습 모델(LMs)을 개선하기 위한 라우팅 프레임워크(routing framework)를 제안합니다. 이 접근법은 인간의 주관적 피드백 수집에 드는 비용과 시간을 줄이면서 더 높은 품질의 주석(annotation)을 달성할 수 있습니다.
- ***Technical Details***: 연구진은 인간 및 LMs의 라벨이 포함된 새로운 10K 인스턴스의 MULTIPREF 데이터세트를 통해 성능 예측 모델(PPM; Performance Prediction Model)을 학습합니다. 이 모델은 주어진 데이터세트의 성능을 예측하며, 라우팅 전략을 사용하여 최적의 성과를 낼 수 있는 혼합 방안을 추천합니다. MULTIPREF는 다양한 공개 리소스에서 수집된 프롬프트를 기반으로 Llama-2, GPT-3.5 등의 모델을 사용해 응답을 생성합니다.
- ***Performance Highlights***: 인간과 LMs의 혼합 피드백을 통해 학습한 보상 모델은 RewardBench에서 7%에서 13%의 절대 성능 향상을 보여주었습니다. 또한, 다양한 LM 벤치마크에 대한 Best-of-N 재랭킹에서도 성능이 향상되었습니다. 헬프스티어 등 다른 데이터세트에서도 동일한 방식이 잘 작동하며, 최종적으로 라우팅된 데이터가 원래의 데이터를 성능 면에서 크게 초과하는 것을 발견했습니다.

### [Teach Multimodal LLMs to Comprehend Electrocardiographic Images](https://arxiv.org/abs/2410.19008)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.19008.png)

Vote: 7

Authors: Ruoqi Liu, Yuelin Bai, Xiang Yue, Ping Zhang

- **What's New**: 이 연구는 다중모달 대형 언어 모델(multimodal large language models; MLLMs)을 전자 심전도 이미지(Electrocardiographic Images) 해석에 적용하는 것이 목적입니다. 이를 위해, 다양한 심전도 관련 작업을 포함하는 100만 건 이상의 데이터셋인 ECGInstruct를 개발했습니다. 그리고 ECG 이미지 이해에 특화된 MLLM인 PULSE를 개발하였으며, 9개의 서로 다른 데이터셋에서 4개의 주요 심전도 이미지를 해석하는 과제를 포함하는 새로운 평가 벤치마크인 ECGBench도 큐레이션하였습니다.
- **Technical Details**: ECGInstruct는 100만 건 이상의 ECG 이미지-텍스트 샘플을 담고 있으며, 이는 IM전류를 사용하여 현실적인 왜곡을 통해 생성되고, 임상의 통찰을 토대로 다양한 ECG 관련 과제로 구성되어 있습니다. 이 데이터는 다각도에서 수집된 것으로, 실제 임상과 유사하게 다양한 경향을 모델이 학습할 수 있게 합니다. PULSE 모델은 이미지 및 지시문을 바탕으로 텍스트 응답을 생성하는 구조로, 학습 데이터는 텍스트와 이미지의 다중턴 대화 형식으로 구성됩니다.
- **Performance Highlights**: PULSE는 ECGBench 벤치마크에서 기존의 독점 및 오픈소스 MLLMs를 15%에서 30%까지 성능으로 앞질렀습니다. 특히, 심각한 도전 과제인 ECG Arena 벤치마크에서도 가장 높은 성능을 보였는데, 이는 다중 턴의 개방형 질의 응답 형식으로, 실제 임상 환경을 반영합니다. PULSE는 다양한 임상 환경에서의 ECG 분석과 해석을 상당히 개선할 수 있는 가능성을 보여줍니다.

### [Leveraging Skills from Unlabeled Prior Data for Efficient Online Exploration](https://arxiv.org/abs/2410.18076)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.18076.png)

Vote: 1

Authors: Kevin Frans, Sergey Levine, Qiyang Li, Max Wilcoxson

- ***What's New***: 이 연구는 SUPE(Skills from Unlabeled Prior data for Exploration)라는 새로운 방법을 제안하여, 라벨이 없는 이전 데이터로부터 효율적인 온라인 탐색 전략을 학습하는 방법을 제시합니다. 이 방법은 먼저 변이 자동 인코더(VAE)로 저수준 스킬을 추출하고, 낙관적 보상 모델을 통해 라벨이 없는 궤적을 의사-라벨링(pseudo-labeling)하여 고급 수준의 탐색 예제로 변환합니다.
- ***Technical Details***: SUPE는 오프라인 예비 학습 단계와 온라인 학습 단계로 구성됩니다. 오프라인 단계에서는 궤적 세그먼트 인코더와 상태-종속적 사전 분포를 변이 자동 인코더(VAE)로 학습합니다. 이러한 예비 학습된 스킬을 통해 온라인 학습 단계에서 고수준 오프-정책 RL 에이전트를 훈련하여 환경에서 탐색을 수행합니다. 낙관적 보상 모듈은 보상을 추정하며, RL 에이전트는 이 정보를 사용하여 탐색 정책을 개선합니다.
- ***Performance Highlights***: SUPE는 긴 호라이즌과 희소 보상 환경에서 기존 메소드를 능가하며, 목표 달성 속도를 크게 향상시켰습니다. 세 개의 어려운 희소 보상 도메인에서 실험한 결과, SUPE는 기존의 방법들보다 더 빠르고 효과적으로 목표를 찾고 학습할 수 있음을 보여줍니다.

### [MMAU: A Massive Multi-Task Audio Understanding and Reasoning Benchmark](https://arxiv.org/abs/2410.19168)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.19168.png)

Vote: 3

Authors: Sreyan Ghosh, Ashish Seth, Utkarsh Tyagi, Ramaneswaran Selvakumar, Oriol Nieto, Ramani Duraiswami, Dinesh Manocha, S Sakshi, Sonal Kumar

- ***What's New***: MMAU는 멀티모달 오디오 이해와 복합 추론 능력을 평가하기 위한 새로운 벤치마크입니다. 이 벤치마크는 전문 지식과 복잡한 추론을 요구하는 작업에 대해 멀티모달 오디오 이해 모델을 평가합니다. 기존 벤치마크와 달리, MMAU는 도메인별 고급 인식 및 추론을 강조하여 모델이 전문가가 직면하는 작업과 유사한 과제를 해결할 수 있도록 도전합니다.
- ***Technical Details***: MMAU 벤치마크는 신중히 큐레이션된 10,000개의 오디오 클립과 인적 주석이 달린 자연어 질문 및 답변을 포함하고 있으며, 이는 음성, 환경 소리 및 음악과 관련된 것입니다. 이 벤치마크는 모델이 27개의 고유 능력을 개발하고 입증할 것을 요구하고, 각 작업은 고유하고 도전적인 과제를 포함합니다. 또한, 18개의 공개 및 독점적 음악-음성-언어 모델을 평가하여 MMAU가 제기하는 상당한 도전을 보여줍니다.
- ***Performance Highlights***: 현재 가장 진보된 모델인 Gemini Pro v1.5가 52.97%의 정확도를 기록했고, 오픈소스 최첨단 모델인 Qwen2-Audio는 52.50%로 상당한 개선의 여지가 있음을 보여줍니다. 이 실험 결과는 현재 오디오-언어 모델이 복잡한 오디오 작업을 해결하는 데 있어서 중요한 도전 과제를 안고 있음을 나타냅니다.

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
