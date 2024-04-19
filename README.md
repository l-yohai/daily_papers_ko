# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using ChatGPT.

Thanks to [@AK391](https://github.com/AK391) for great work.


## Daily Papers (2024-04-19)

### [Dynamic Typography: Bringing Words to Life](https://arxiv.org/abs/2404.11614)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2404.11614.png)

Vote: 22

Authors: Bolin Zhao, Yihao Meng, Daniel Cohen-Or, Huamin Qu, Hao Ouyang, Yue Yu, Zichen Liu

- 텍스트 애니메이션은 정적인 커뮤니케이션을 동적인 경험으로 변환하여 단어에 동작을 불어넣고 감정을 유발하며 의미를 강조하고 매력적인 내러티브를 구성합니다.
- 본 논문에서는 'Dynamic Typography'라는 이름의 자동 텍스트 애니메이션 방식을 제시하며, 이는 문자의 의미를 전달하고 사용자 프롬프트에 기반한 활기찬 움직임을 문자에 주입하는 두 가지 어려운 과제를 결합합니다.
- 우리의 기술은 벡터 그래픽 표현과 종단간 최적화 기반 프레임워크를 활용하여, 신경 변위 필드를 사용해 글자를 기본 형태로 변환하고 프레임별 모션을 적용함으로써 의도된 텍스트 개념과의 일관성을 촉진합니다.
- 형태 보존 기술과 지각 손실 규제를 사용하여 애니메이션 과정 전반에 걸쳐 가독성과 구조적 무결성을 유지합니다.
- 다양한 텍스트-비디오 모델에서 접근 방식의 일반성을 입증하고, 별도의 작업을 포함할 수 있는 기준 방법보다 우리의 종단간 방법론의 우수성을 강조합니다.
- 우리의 프레임워크가 사용자 프롬프트를 충실히 해석하고 가독성을 유지하면서 일관된 텍스트 애니메이션을 생성하는 효과를 정량적 및 정성적 평가를 통해 시연합니다.
- 관련 코드는 https://animate-your-word.github.io/demo/에서 확인할 수 있습니다.

### [MeshLRM: Large Reconstruction Model for High-Quality Mesh](https://arxiv.org/abs/2404.12385)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2404.12385.png)

Vote: 13

Authors: Sai Bi, Hao Tan, Kai Zhang, Xinyue Wei, Zexiang Xu, Valentin Deschaintre, Fujun Luan, Kalyan Sunkavalli, Hao Su

- MeshLRM은 네 개의 입력 이미지만으로 1초 미만의 시간에 고품질 메시를 재구성할 수 있는 새로운 LRM 기반 접근 방식을 제안합니다.
- 이 모델은 기존의 NeRF 기반 재구성에 집중한 다른 LRMs와 달리, LRM 프레임워크 안에 메시 추출과 렌더링을 차별화 가능하게 통합합니다.
- MeshLRM은 사전 훈련된 NeRF LRM을 메시 렌더링으로 미세 조정함으로써, 최종단 메시 재구성을 실현합니다.
- 또한, 이 모델은 복잡한 설계를 간소화하여 LRM 아키텍처를 개선하고, 저해상도 및 고해상도 이미지로 순차적 훈련을 통해 훨씬 빠른 수렴을 달성합니다.
- 이러한 새로운 LRM 훈련 전략은 더 적은 계산으로 더 나은 품질을 가능하게 합니다.
- MeshLRM은 희소 뷰 입력에서의 최신 메시 재구성을 달성할 뿐만 아니라 텍스트-투-3D, 싱글-이미지-투-3D 생성 등 다양한 하류 애플리케이션에서 활용 가능합니다.

### [EdgeFusion: On-Device Text-to-Image Generation](https://arxiv.org/abs/2404.11925)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2404.11925.png)

Vote: 9

Authors: Hyoung-Kyu Song, Changgwun Lee, Hanyoung Yim, Tairen Piao, Shinkook Choi, Bo-Kyeong Kim, Tae-Ho Kim, Thibault Castells, Jae Gon Kim

- 안정된 확산(SD: Stable Diffusion)을 위한 텍스트-이미지 생성의 높은 연산 부담은 실용적 적용에 중대한 장애로 작용합니다.
- 기존 연구들이 샘플링 단계를 줄이는 방법(LCM: Latent Consistency Model) 및 구조 최적화(프루닝, 지식 증류)에 주목하는 가운데, 저희 연구는 컴팩트 SD 변형인 BK-SDM에서 시작하여 새로운 접근을 시도합니다.
- 일반적으로 사용되는 크롤링 데이터셋에 LCM을 직접 적용할 경우 만족스럽지 못한 결과가 나타남을 확인하였습니다.
- 이에 따라, (1) 선도적 생성 모델에서 고품질 이미지-텍스트 쌍을 활용하고 (2) LCM에 맞춰진 고급 증류 과정을 디자인하는 두 가지 전략을 개발했습니다.
- 양자화, 프로파일링, 그리고 on-device 배치를 통한 철저한 탐구를 통해, 자원이 제한된 엣지 기기에서 1초 미만의 지연 시간으로 사실적이고 텍스트에 부합하는 이미지를 단 두 단계로 빠르게 생성할 수 있게 되었습니다.



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
