# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using ChatGPT.

Thanks to [@AK391](https://github.com/AK391) for great work.


## Daily Papers (2023-12-13)

### [FreeControl: Training-Free Spatial Control of Any Text-to-Image Diffusion Model with Any Condition](https://arxiv.org/abs/2312.07536)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/A4WizM2EGIE61ifD6ibgb.jpeg)

Vote: 11

Authors: Sicheng Mo, Fangzhou Mu, Kuan Heng Lin, Yanli Liu, Bochen Guan, Yin Li, Bolei Zhou

- 최근 접근법인 ControlNet은 텍스트-이미지 변환(T2I) 확산 모델에서 세밀한 공간 조정을 사용자에게 제공하지만, 각종 공간 조건이나 모델 구조, 체크포인트에 대해 별도의 모듈을 훈련해야 합니다.
- 본 연구에서는 여러 조건, 구조, 체크포인트를 동시에 지원하는 훈련 없는 제어 가능한 T2I 생성을 위한 FreeControl을 소개합니다.
- FreeControl은 구조 지침을 통해 지침 이미지와의 구조 정렬을 용이하게 하고, 동일한 시드를 사용하여 생성된 이미지 간의 외관 공유를 가능하게 하는 외관 지침을 설계합니다.
- 다양한 사전 훈련된 T2I 모델에 걸쳐서 FreeControl의 우수한 성능을 입증하는 광범위한 질적 및 양적 실험이 수행되었습니다.
- 특히, FreeControl은 다양한 구조와 체크포인트에서 훈련 없는 편리한 제어를 가능하게 하고, 기존 훈련 없는 방법들이 실패하는 도전적인 입력 조건에서도 작동하며, 훈련 기반 접근법과 경쟁하는 합성 품질을 달성합니다.

### [CCM: Adding Conditional Controls to Text-to-Image Consistency Models](https://arxiv.org/abs/2312.06971)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/mepuZ-w6HiUDMgotLQl1G.jpeg)

Vote: 7

Authors: Jie Xiao, Kai Zhu, Han Zhang, Zhiheng Liu, Yujun Shen, Yu Liu, Xueyang Fu, Zheng-Jun Zha

- 일관성 모델(CMs)은 효율적이고 고품질의 시각적 콘텐츠 생성에 유망함을 보여줬으나, 사전 훈련된 CMs에 새로운 조건부 제어를 추가하는 방법은 탐구되지 않았다.
- 본 기술 보고서에서는 CMs에 ControlNet과 같은 조건부 제어를 추가하기 위한 대안적 전략을 고려하고, 중요한 세 가지 발견을 제시한다.
- 1) 확산 모델(DMs)에 대해 훈련된 ControlNet은 CMs에 대한 고차원 의미 제어에 직접 적용될 수 있으나, 저수준의 세부 사항 및 리얼리즘 제어에는 어려움이 있다.
- 2) CMs는 독립적인 생성 모델 클래스로서, 송(Song) 등이 제안한 일관성 훈련을 사용하여 ControlNet을 처음부터 훈련시킬 수 있다.
- 3) 경량화된 어댑터는 일관성 훈련을 통해 다중 조건 하에서 공동 최적화될 수 있으며, 이를 통해 DMs 기반 ControlNet을 CMs로 신속하게 전송하는 것이 가능하다.
- 이 세 가지 해결책을 에지, 깊이, 인간 자세, 저해상도 이미지 및 텍스트-이미지 잠재 일관성 모델을 사용한 마스크 이미지와 같은 다양한 조건부 제어에 걸쳐 연구했다.

### [Interfacing Foundation Models' Embeddings](https://arxiv.org/abs/2312.07532)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/G4lX6x086B1w9XFt6ttvn.jpeg)

Vote: 5

Authors: Xueyan Zou, Linjie Li, Jianfeng Wang, Jianwei Yang, Mingyu Ding, Zhengyuan Yang, Feng Li, Hao Zhang, Shilong Liu, Arul Aravinthan, Yong Jae Lee, Lijuan Wang

- 이 논문은 파운데이션 모델의 임베딩을 정렬하기 위한 일반화된 인터페이스인 FIND를 소개합니다.
- 경량 트랜스포머 인터페이스를 통해 어떠한 파운데이션 모델의 가중치도 조정하지 않고 통합된 이미지(세분화) 및 데이터셋 수준(검색) 이해를 할 수 있음을 보여줍니다.
- 제안하는 인터페이스는 다양한 작업 수행, 주의 마스크와 임베딩 유형의 프로토타이핑을 통한 구현, 새로운 과제 및 모델에 적응 가능, 다중 과제 다중 모달 트레이닝의 이점을 통한 상호 연결된 공유 임베딩 공간 생성 등 여러 가지 이점을 갖고 있습니다.
- FIND-Bench를 도입하여 COCO 데이터셋에 새로운 트레이닝 및 평가 주석을 추가하고, 상호 연결된 세분화 및 검색 임베딩 공간을 소개합니다.
- 저희의 접근 방식은 FIND-Bench에서 최고 수준의 성능을 달성하고, 표준 검색 및 세분화 설정에서도 경쟁력 있는 성능을 보입니다.
- 관련 훈련, 평가 및 데모 코드와 데이터셋은 https://github.com/UX-Decoder/FIND 에서 공개되어 있습니다.



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
