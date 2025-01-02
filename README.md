# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using GPT-4o.

Thanks to [@AK391](https://github.com/AK391) for great work.

## Daily Papers (2025-01-02)

### [HUNYUANPROVER: A Scalable Data Synthesis Framework and Guided Tree Search for Automated Theorem Proving](https://arxiv.org/abs/2412.20735)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.20735.png)

Vote: 3

Authors: Chen Li, Weikang Wang, Yang Li, Tao Yang, Dong Du, Linfeng Song, Haitao Mi

- ***What's New***: HUNYUANPROVER는 자동 정리 증명(Automated Theorem Proving; ATP)을 위한 축척 가능한 데이터 합성 프레임워크와 안내된 트리 탐색(Guided Tree Search) 알고리즘을 포함하는 새로운 시스템입니다. 이 시스템은 미니F2F 테스트에서 68.4%의 높은 성과를 기록하고, 4개의 국제수학올림피아드(IMO) 문제를 증명했습니다.
- ***Technical Details***: HUNYUANPROVER는 두 가지 핵심 모듈로 구성됩니다: 대규모 증명 데이터 생성기(Prover-Data Generator)와 안내된 트리 탐색 알고리즘입니다. 데이터 생성기는 자연어 수학 문제를 타겟 증명 형식(LEAN4)으로 변환하여 새로운 훈련 데이터를 생성합니다. 테스트 시에는 트리 탐색 알고리즘과 여러 비판(Critic) 모델을 사용한 '느린 사고' 방식으로 복잡한 문제를 해결합니다. 또한, 3가지 유형의 비판 모델을 설계하여 탐색 과정을 안내합니다: 정책 신뢰도(Policy Confidence), 프로세스 보상 모델(Process Reward Model), 거리 기반 비판 모델(Distance Critic).
- ***Performance Highlights***: HUNYUANPROVER는 miniF2F 벤치마크에서 68.4%의 정확도를 기록하며 이전의 최고 기록을 2.5% 포인트 초과했습니다. 개발된 거리 비판 모델(Distance Critic)을 통해 탐색 과정의 효율성을 증가시켰으며, 이는 데이터 다각화 및 선택이 자동 정리 증명 모델의 성능에 중요하다는 것을 시사합니다. 이 시스템은 30,000개의 합성 데이터 인스턴스를 공개할 예정입니다.

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
