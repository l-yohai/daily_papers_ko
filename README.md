# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using ChatGPT.

Thanks to [@AK391](https://github.com/AK391) for great work.


## Daily Papers (2023-11-22)

### [MagicDance: Realistic Human Dance Video Generation with Motions & Facial Expressions Transfer](https://arxiv.org/abs/2311.12052)

[Watch Video](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/QPqDXcHNhksZaqkYOjZNQ.mp4)
<div><video controls src="https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/QPqDXcHNhksZaqkYOjZNQ.mp4" muted="false"></video></div>

Authors: Di Chang, Yichun Shi, Quankai Gao, Jessica Fu, Hongyi Xu, Guoxian Song, Qing Yan, Xiao Yang, Mohammad Soleymani

- 본 연구에서는 도전적인 인간 댄스 비디오에 대한 2D 인간 동작 및 얼굴 표정 전송을 위한 확산 기반 모델인 MagicDance를 제안합니다.
- 타겟 정체성의 인간 댄스 비디오를 새로운 포즈 시퀀스에 의해 생성하면서 정체성은 변하지 않도록 함을 목표로 합니다.
- 이를 위해, 동작과 외모(예: 얼굴 표정, 피부색, 복장)을 분리하기 위한 두 단계 훈련 전략, 즉 외모-제어 블록의 사전 훈련과 동일 데이터셋의 인간 댄스 포즈에 대한 외모-포즈-조인트-제어 블록의 미세 조정을 제안합니다.
- 제안된 모델은 뛰어난 외모 제어와 시간적으로 일관된 상반신, 얼굴 속성, 심지어 배경을 가능하게 하며, 추가 데이터 미세 조정 없이 보지 못한 인간 정체성과 복잡한 동작 시퀀스에 잘 일반화됩니다.
- 모델은 이미지 확산 모델의 사전 지식을 활용하여 다양한 인간 속성을 갖춘 데이타를 필요로 하지 않으며 사용하기 쉽고 Stable Diffusion에 플러그인 모듈/확장으로 간주될 수 있습니다.
- 또한 모델은 단지 포즈 입력을 통해 한 정체성에서 다른 정체성으로의 외모 전송뿐만 아니라 만화 스타일의 스타일 변형을 가능하게 하는 제로샷 2D 애니메이션 생성 능력을 시연합니다.
- TikTok 데이터셋에서의 우수한 성능을 입증하는 광범위한 실험을 통해 우리의 모델이 얼마나 우수한지를 입증합니다.

### [Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models](https://arxiv.org/abs/2311.12092)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/rIACZU_Re1MpOgIJpG032.png)

Authors: Rohit Gandikota, Joanna Materzynska, Tingrui Zhou, Antonio Torralba, David Bau

- 이미지 생성 분야에서 확산 모델의 특정 속성들을 정밀하게 제어할 수 있는 해석 가능한 '컨셉 슬라이더' 생성 방법을 제시합니다.
- 본 연구에서 소개하는 방법은 개별 컨셉에 해당하는 저랭크 매개변수 방향을 파악하면서 다른 속성들과의 간섭을 최소화합니다.
- 텍스트적 또는 시각적 컨셉 모두에 사용할 수 있는 슬라이더 방향은 소량의 프롬프트나 샘플 이미지를 활용하여 만들어집니다.
- 컨셉 슬라이더는 플러그 앤 플레이 가능하며, 효율적으로 구성되고 연속적으로 조정되어, 이미지 생성의 정밀한 제어를 가능하게 합니다.
- 이전의 편집 기술들과 비교한 정량적 실험에서 우리의 슬라이더는 더 강력한 목표 편집과 더 낮은 간섭을 보여줍니다.
- 날씨, 나이, 스타일 및 표정과 같은 다양한 컨셉들을 위한 슬라이더와 슬라이더 조합을 시연하였습니다.
- 텍스트로 설명하기 어려운 시각적 개념에 대한 직관적인 편집을 위해 StyleGAN의 라텐트를 전송할 수 있음을 보여줍니다.
- 본 방법은 Stable Diffusion XL의 지속적인 품질 문제, 예를 들어 객체 변형 수리 및 왜곡된 손 고치기에 도움이 될 수 있음을 발견하였습니다.
- 제공되는 코드, 데이터, 훈련된 슬라이더는 https://sliders.baulab.info/ 에서 확인할 수 있습니다.

### [HierSpeech++: Bridging the Gap between Semantic and Acoustic Representation of Speech by Hierarchical Variational Inference for Zero-shot Speech Synthesis](https://arxiv.org/abs/2311.12454)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/A1PRY8UlPr71zU4jr-81r.png)

Authors: Sang-Hoon Lee, Ha-Yeong Choi, Seung-Bin Kim, Seong-Whan Lee

- 이 논문은 기존의 대규모 언어 모델에 기반한 제로샷 음성 합성의 한계를 극복하기 위해, 텍스트-음성(TTS) 및 음성 변환(VC)을 위한 빠르고 강력한 제로샷 음성 합성기인 HierSpeech++를 제안합니다.
- HierSpeech++는 계층적 음성 합성 프레임워크를 적용하여 인공 음성의 강인성과 표현력을 크게 향상시키는 것을 검증하였습니다.
- 제로샷 음성 합성 상황에서도 텍스트 제시에 따라 자기지도학습 음성 표현 및 F0(기본 음높이) 표현을 생성하는 텍스트-투-벡 프레임워크를 채택하여 음성의 자연스러움과 화자 유사성을 눈에 띄게 향상시켰습니다.
- 계층적 변분 오토인코더(VAE)를 이용하여 16 kHz의 음성을 효율적으로 48 kHz로 슈퍼 해상도로 전환하는 음성 슈퍼 해상도 프레임워크가 소개되었습니다.
- 실험 결과 HierSpeech++가 대규모 언어 모델 기반 및 확산 기반 모델을 초과하는 성능을 보여주며, 첫 인간 수준 품질의 제로샷 음성 합성을 달성했다는 것을 입증하였습니다.
- 오디오 샘플과 소스 코드는 https://github.com/sh-lee-prml/HierSpeechpp에서 확인 가능합니다.

### [NeuroPrompts: An Adaptive Framework to Optimize Prompts for Text-to-Image Generation](https://arxiv.org/abs/2311.12229)

[Watch Video](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/HQ878D7Ac34_1GqKl2TZD.mp4)
<div><video controls src="https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/HQ878D7Ac34_1GqKl2TZD.mp4" muted="false"></video></div>

Authors: Shachar Rosenman, Vasudev Lal, Phillip Howard

- 텍스트-이미지 확산 모델의 최근 발전에도 불구하고 고품질 이미지를 얻기 위해서는 인간이 프롬프트 엔지니어링 기술을 개발해야 했습니다.
- 본 연구에서는 사용자의 프롬프트를 자동으로 향상시켜 텍스트-이미지 모델들에 의해 생성된 이미지의 품질을 개선하는 적응형 프레임워크인 NeuroPrompts를 제시합니다.
- 이 프레임워크는 사람들이 만든 프롬프트와 유사한 프롬프트를 생성하도록 조정된 사전 훈련된 언어 모델을 사용한 제한적 텍스트 디코딩을 활용합니다.
- 사용자는 제약 조건 집합 명세를 통해 스타일적 특징에 대한 제어가 가능하며, 이를 통해 더 고품질의 텍스트-이미지 생성을 가능하게 합니다.
- 저자들은 Stable Diffusion을 사용한 프롬프트 향상 및 이미지 생성을 위한 상호작용식 어플리케이션을 제작하여 프레임워크의 유틸리티를 보여줍니다.
- 또한, 텍스트-이미지 생성을 위해 인간이 제작한 대규모 프롬프트 데이터셋을 사용하여 실험을 수행하고, 제안한 접근법이 자동으로 향상된 프롬프트를 생성하여 이미지 품질을 향상시킴을 보여줍니다.
- 저자들은 NeuroPrompts의 코드, 스크린캐스트 비디오 데모 및 실시간 데모 인스턴스를 공개적으로 제공합니다.

### [GPT4Motion: Scripting Physical Motions in Text-to-Video Generation via Blender-Oriented GPT Planning](https://arxiv.org/abs/2311.12631)

[Watch Video](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/GCwvjLFC_Tga6eXwub-qN.mp4)
<div><video controls src="https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/GCwvjLFC_Tga6eXwub-qN.mp4" muted="false"></video></div>

Authors: Jiaxi Lv, Yi Huang, Mingfu Yan, Jiancheng Huang, Jianzhuang Liu, Yifan Liu, Yafei Wen, Xiaoxin Chen, Shifeng Chen

- 최근 텍스트 기반 비디오 생성 분야에서 발전하였으나 높은 계산 비용과 물리 운동의 일관성 문제를 겪고 있다.
- 'GPT4Motion'이라는 새로운 프레임워크를 제안하여 이러한 문제를 해결하고, 비디오 합성의 질을 향상시키고자 한다.
- 이 프레임워크는 GPT와 같은 큰 언어 모델의 기획 능력과 Blender의 물리 시뮬레이션 강점, 그리고 텍스트-이미지 확산 모델의 이미지 생성 능력을 결합한다.
- GPT4Motion은 사용자의 텍스트 프롬프트에 기반하여 Blender 스크립트를 생성하는 GPT-4를 활용한다.
- Blender의 내장 물리 엔진을 사용하여 시퀀스 간에 일관된 물리 운동을 캡슐화하는 기본 장면 구성 요소를 작성한다.
- 생성된 구성 요소는 Stable Diffusion에 입력되어 텍스트 프롬프트와 일치하는 비디오를 생성한다.
- 단단한 물체의 낙하와 충돌, 천의 드레이핑과 흔들림, 액체 흐름 등 세 가지 기본 물리 운동 시나리오에 대한 실험 결과는 GPT4Motion이 운동 일관성과 개체 일관성을 효율적으로 유지하면서 고품질 비디오를 생성할 수 있음을 보여준다.
- GPT4Motion은 텍스트-비디오 연구에 새로운 통찰력을 제공하며, 그 질을 향상시키고 미래 탐구의 지평을 넓힌다.

### [PF-LRM: Pose-Free Large Reconstruction Model for Joint Pose and Shape Prediction](https://arxiv.org/abs/2311.12024)

[Watch Video](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/PNftOjPrJHc2bdQUuGJiY.mp4)
<div><video controls src="https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/PNftOjPrJHc2bdQUuGJiY.mp4" muted="false"></video></div>

Authors: Peng Wang, Hao Tan, Sai Bi, Yinghao Xu, Fujun Luan, Kalyan Sunkavalli, Wenping Wang, Zexiang Xu, Kai Zhang

- 본 논문에서는 몇 장의 위치가 지정되지 않은 영상만으로도 소규모의 시각적 중첩으로 3D 객체를 재구성하고, 동시에 카메라 상대 위치를 약 1.3초 내 A100 GPU 한 대로 추정하는 Pose-Free Large Reconstruction Model (PF-LRM)을 제안합니다.
- PF-LRM은 3D 객체 토큰과 2D 이미지 토큰 간의 정보 교환을 위해 셀프-어텐션 블록을 활용하는 확장성 높은 방법으로, 각 뷰에 대한 대략적인 포인트 클라우드를 예측한 다음, 다른 뉴러블 Perspective-n-Point (PnP) 솔버를 사용하여 카메라 포즈를 획득합니다.
- 약 100만 개의 다시점 위치 데이터로 훈련되었을 때, PF-LRM은 미리 보지 못한 다양한 평가 데이터셋에서의 자세 예측 정확도 및 3D 재구성 품질 측면에서 기준 모델들을 크게 앞서는 강력한 크로스데이터셋 일반화 능력을 보여줍니다.
- 또한, 빠른 피드포워드 추론을 통해 텍스트/이미지에서 3D로의 작업에 모델의 적용 가능성을 시연합니다.
- 이 프로젝트의 웹사이트는 https://totoro97.github.io/pf-lrm 에서 확인할 수 있습니다.

### [SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering](https://arxiv.org/abs/2311.12775)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/WxnaGTRV7UFFb1PpM5_ms.png)

Authors: Antoine Guédon, Vincent Lepetit

- 본 논문에서는 고정밀도와 극도로 빠른 메쉬 추출을 가능하게 하는 3D 가우시안 스플래팅 방법을 제안한다.
- 가우시안 스플래팅은 실제감있는 렌더링을 할 수 있으면서 NeRF보다 상당히 훈련 속도가 빠르다는 장점이 있으나, 최적화 후에 정리되지 않은 수백만 개의 작은 3D 가우시안들로부터 메쉬를 추출하는 것은 어려운 도전이다.
- 첫 번째 주요 기여는 가우시안들이 장면의 표면에 잘 정렬되도록 하여, 포아송 재구성을 사용한 빠르고, 확장 가능하며, 세부사항을 보존하는 메쉬 추출이 가능하도록 하는 정규화 항을 제안한다.
- 반대로 일반적으로 뉴럴 SDF에서 메쉬를 추출하기 위해 사용되는 Marching Cubes 알고리즘이 세부사항을 놓치기 쉽고 시간 소요가 많다는 단점이 있다.
- 마지막으로, 가우시안들을 메쉬의 표면에 결합시키고, 가우시안 스플래팅 렌더링을 통해 이 가우시안들과 메쉬를 공동 최적화하는 선택적인 정제 전략을 도입한다.
- 이러한 방식을 통해 사용자는 전통적인 소프트웨어를 이용하여 가우시안들 대신 메쉬를 조작함으로써 쉽게 편집, 조각, 리깅, 애니메이션, 합성 및 재조명을 할 수 있다.
- 실제적인 렌더링을 위해 편집 가능한 메쉬를 가져오는 것은 우리의 방법으로 몇 분 내에 가능하며, 다른 뉴럴 SDF 기술들에 비해 수 시간을 절약하면서 더욱 높은 품질의 렌더링을 제공한다.

### [PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics](https://arxiv.org/abs/2311.12198)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/j7gyxiWMux1_LmXSFfvyf.png)

Authors: Tianyi Xie, Zeshun Zong, Yuxin Qiu, Xuan Li, Yutao Feng, Yin Yang, Chenfanfu Jiang

- 'PhysGaussian'이라는 새로운 방법을 소개하며, 이는 뉴턴 역학에 기반한 물리적 원리를 3D 가우스 함수와 통합하여 새로운 모션 합성을 고품질로 달성한다.
- 맞춤형 재질점 방법(Material Point Method, MPM)을 사용하여, 연속체 역학의 원리에 따라 진화하는 물리적 의미가 있는 운동 변형과 기계적 스트레스 속성으로 3D Gaussian 커널을 풍부하게 구성한다.
- 이 방법의 핵심적인 특징은 물리적 시뮬레이션과 시각적 렌더링 사이의 원활한 통합으로, 두 구성요소 모두 동일한 3D 가우스 커널을 이산 표현으로 사용한다.
- 이는 삼각형/사면체 메싱, 마칭 큐브, "케이지 메시" 혹은 다른 기하학적 임베딩이 필요없게 하여, "시뮬레이션하는 것이 보이는 것(WS^2)"이라는 원칙을 강조한다.
- 탄성체, 금속, 비뉴턴유체, 과립성 재료를 포함한 광범위한 재료에 대해서 우수한 다양성을 보이며, 새로운 시점과 움직임을 가진 다양한 시각적 콘텐츠 생성에서 강력한 능력을 과시한다.
- 프로젝트 페이지는 다음 주소에서 제공하고 있다: https://xpandora.github.io/PhysGaussian/.



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
