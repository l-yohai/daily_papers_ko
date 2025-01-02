# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using GPT-4o.

Thanks to [@AK391](https://github.com/AK391) for great work.

## Daily Papers (2025-01-02)

### [OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis](https://arxiv.org/abs/2412.19723)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.19723.png)

Vote: 23

Authors: Yu Qiao, Zhoumianze Liu, Zhenyu Wu, Qiushi Sun, Yian Wang, Guohao Li, Junxian He, Liheng Chen, Zhiyong Wu, Kanzhi Cheng, Chuanyang Jin, Fangzhi Xu, Chengyou Jia, Ben Kao, Zichen Ding

- ***What's New***: OS-Genesis는 GUI 데이터 수집의 병목 현상을 해결하기 위해 역태스크 합성을 통해 GUI 에이전트 궤적(Trajectory) 생성 과정을 자동화하는 혁신적인 파이프라인을 제안합니다. 이는 인간의 감독이나 사전 정의된 태스크 없이 에이전트가 먼저 환경을 감지하고 상호작용을 통해 고품질의 태스크를 추출하게 함으로써 다양한 데이터를 생성할 수 있도록 합니다.
- ***Technical Details***: OS-Genesis는 인터랙션 주도적 접근(Interaction-Driven Approach)을 활용하여 UI 요소들을 탐색하고, 관찰된 상태 및 행동을 통해 저수준 태스크를 생성하며 이는 고수준 태스크로 확장됩니다. 생성된 고수준 태스크는 역태스크 합성(Reverse Task Synthesis)을 통해 궤적을 생성하는 데 활용됩니다. 생성된 궤적의 품질을 보장하기 위해 궤적 보상 모델(Trajectory Reward Model; TRM)을 도입했습니다.
- ***Performance Highlights***: OS-Genesis는 AndroidWorld에서 기존 방법을 크게 능가하며 성능을 두 배 가까이 향상시켰습니다. AndroidControl 및 WebArena 등의 기준에서 대다수의 연기 함수 상에서 다른 태스크 주도적 방법들보다 높은 성능을 보여줍니다. 이는 OS-Genesis로 오픈 소스 기반 에이전트의 성능과 SOTA GPT-4o 기반 M3A 에이전트 사이의 성능 격차를 큰 폭으로 줄이는데 일조합니다.

### [Xmodel-2 Technical Report](https://arxiv.org/abs/2412.19638)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.19638.png)

Vote: 5

Authors: Wang Qun, Liu Yang, Lin Qingquan, Qu Zhijiu, Jiang Ling

- ***What's New***: Xmodel-2는 1.2억 매개변수(1.2-billion-parameter)를 가진 대형 언어 모델(Large Language Model; LLM)로, 복잡한 추론 작업을 위해 설계되었습니다. 특히, Tensor Programs 기반의 혁신적인 아키텍처를 채택하여 다양한 규모의 모델이 동일한 하이퍼파라미터(Hyperparameters)를 공유할 수 있도록 하여, 작은 모델에서의 실험 결과를 큰 모델에 쉽게 이전할 수 있습니다.
- ***Technical Details***: Xmodel-2는 LLama 2와 유사한 아키텍처를 채택했으며, 그룹쿼리 주의 메커니즘(Grouped-Query Attention; GQA)을 사용하여 훈련 및 추론을 최적화합니다. 1.5조 개의 토큰(Tokens)으로 사전훈련되었으며, MiniCPM의 워맙-안정-감쇠(Warmup-Stable-Decay; WSD) 학습률 스케줄러를 사용하여 훈련 효율성과 안정성을 높였습니다. 또한, 데이터 비율 최적화(Data Ratio Optimization)를 통해 SFT 데이터(Supervised Fine-Tuning Data)의 최적 비율을 탐색하였습니다.
- ***Performance Highlights***: Xmodel-2는 1B-매개변수 언어 모델 중에서 최첨단(State-of-the-Art; SOTA) 성능을 입증했으며, 상식 추론, 복잡한 추론 및 에이전트 기반 작업에서 우수한 성능을 보였습니다. 특히 에이전트 작업에서 웹 탐색(WebShop)과 같은 복잡한 환경에서 성공률이 타 모델보다 높았습니다. 이러한 결과는 Xmodel-2의 실제 응용 가능성을 시사합니다.

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
