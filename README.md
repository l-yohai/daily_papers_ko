# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using ChatGPT.

Thanks to [@AK391](https://github.com/AK391) for great work.


## Daily Papers (2024-03-27)

### [The Unreasonable Ineffectiveness of the Deeper Layers](https://arxiv.org/abs/2403.17887)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2403.17887.png)

Vote: 32

Authors: Hassan Shapourian, Kushal Tirumala, Paolo Glorioso, Andrey Gromov, Daniel A. Roberts

- 인기 있는 오픈-웨이트 사전 훈련된 LLM의 간단한 레이어-가지치기 전략을 실증적으로 연구한 결과, 많은 부분(최대 절반까지)의 레이어를 제거하기 전까지는 다양한 질의응답 벤치마크에서 성능 저하가 거의 없음을 발견하였습니다.
- 모델에서 제거할 최적의 레이어 블록을 식별하기 위하여 레이어 간의 유사성을 고려하고, 손상을 "치유"하기 위해 적은 양의 파인튜닝을 수행합니다.
- 높은 효율성을 가진 파인튜닝 방법인 양자화 및 저랭크 어댑터(QLoRA)를 사용하여, 모든 실험이 단일 A100 GPU에서 수행될 수 있도록 하였습니다.
- 실용적 관점에서 이 결과는 레이어 가지치기 방법이 파인튜닝 시 연산 자원을 더욱 줄일 수 있는 보완 전략으로, 또한 추론 시 메모리와 대기 시간을 개선할 수 있는 방법으로 활용될 수 있음을 시사합니다.
- 과학적 관점에서 이 LLM의 레이어 제거에 대한 견고성은 현재의 사전 훈련 방법이 네트워크의 더 깊은 레이어의 매개변수를 제대로 활용하지 못하고 있거나, 얕은 레이어가 지식을 저장하는 데 중요한 역할을 한다는 것을 의미할 수 있습니다.

### [2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://arxiv.org/abs/2403.17888)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2403.17888.png)

Vote: 16

Authors: Binbin Huang, Zehao Yu, Shenghua Gao, Andreas Geiger, Anpei Chen

- 이 논문은 다시점 이미지로부터 기하학적으로 정확한 방사 형상장을 모델링하고 재구성하기 위한 새로운 접근법인 2D Gaussian Splatting(2DGS)을 제시한다.
- 3D Gaussian Splatting(3DGS)는 다시점 일관성이 없는 3D 가우스 함수 때문에 정확한 표면 표현에 실패하는 문제점을 가지고 있는 반면, 2DGS는 2D 가우스 함수를 이용하여 시점 일관성 있는 지오메트리를 제공한다.
- 논문에서는 가느다란 표면을 정확히 복구하고 안정적인 최적화를 달성하기 위해 광선-스플랏 교차점과 래스터화를 활용하는 관점 정확한 2D 스플래팅 프로세스를 소개한다.
- 깊이 왜곡 및 법선 일관성 조건을 추가함으로써 재구성의 품질을 더욱 향상시킨다.
- 제안하는 차별화 가능한 렌더러는 노이즈가 없는 상세한 지오메트리 재구성을 가능하게 하면서도 경쟁력 있는 외관 품질, 빠른 교육 속도, 실시간 렌더링을 유지한다.
- 코드는 공개적으로 제공될 예정이다.

### [InternLM2 Technical Report](https://arxiv.org/abs/2403.17297)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2403.17297.png)

Vote: 12

Authors: Jiaye Ge, Tao Gui, Kai Chen, Zhaoye Fei, Haojiong Chen, Zheng Cai, Xin Chen, Conghui He, Yuzhe Gu, Yang Gao, Qi Fan, Aijia Guo, Zehui Chen, +, Pei Chu, Zhi Chen, Haodong Duan, Xiaoyi Dong, Qipeng Guo, Xun Chen, Chenya Gu, Keyu Chen, Maosong Cao

- 대규모 언어 모델(Large Language Models, LLMs)의 진화는 인공 일반 지능(Artificial General Intelligence, AGI)의 도래에 대한 논의를 촉발시켰지만, 이러한 발전을 오픈 소스 모델에서 재현하는 것은 어려웠습니다.
- 본 논문에서는 6개 차원과 30개 벤치마크에서 모든 전임자들을 능가하는, 긴 문맥 모델링 및 개방형 주관적 평가를 통해 혁신적인 사전 훈련 및 최적화 기법을 통해 우수한 성능을 보이는 오픈 소스 LLM, InternLM2를 소개합니다.
- InternLM2의 사전 훈련 과정은 텍스트, 코드, 그리고 긴 문맥 데이터를 포함한 다양한 데이터 유형의 준비에 대해 면밀히 설명하고 있으며, 4k 토큰에서 시작하여 사전 훈련 및 미세 조정 단계에서 32k 토큰으로 능률적으로 발전하여 200k “Needle-in-a-Haystack” 테스트에서 놀라운 성능을 보이는 긴 기간의 의존성을 효과적으로 포착합니다.
- InternLM2는 상반되는 인간의 선호와 보상 해킹을 다루는 새로운 조건부 온라인 강화 학습(Conditional Online Reinforcement Learning from Human Feedback, COOL RLHF) 전략과 감독된 미세 조정(Supervised Fine-Tuning, SFT)을 사용하여 더욱 정렬되었습니다.
- 연구 커뮤니티에 모델의 발전을 통찰할 기회를 제공하기 위해 다양한 훈련 단계 및 모델 크기의 InternLM2 모델을 공개합니다.

### [TC4D: Trajectory-Conditioned Text-to-4D Generation](https://arxiv.org/abs/2403.17920)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2403.17920.png)

Vote: 10

Authors: Xihui Liu, David B. Lindell, Yifan Wang, Sherwin Bahmani, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, Victor Rong, Gordon Wetzstein, Xian Liu, Ivan Skorokhodov, Ziwei Liu

- 최근 텍스트에서 4D 생성 기술은 사전 훈련된 텍스트-비디오 모델의 감독 하에 동적인 3D 장면을 합성합니다만, 운동량을 나타내는 기존 모델들은 생성할 수 있는 움직임의 범위에 한계가 있어 볼륨 렌더링을 위한 바운딩 박스를 크게 벗어나는 움직임을 합성하지 못합니다.
- 이러한 유연하지 않은 움직임 모델은 4D 생성 기법과 사실적인 비디오 생성 모델 사이의 현실감 격차를 가져옵니다.
- 본 논문에서는 TC4D, 즉 궤적 조건부 텍스트-4D 생성을 제안하며, 이는 전역 및 지역 구성 요소로 움직임을 분해합니다.
- 전역 움직임은 스플라인으로 매개변수화된 궤적을 따라서 강체 변환하는 장면의 바운딩 박스를 사용하여 나타냅니다.
- 텍스트-비디오 모델로부터의 감독을 통해, 전역 궤적에 부합하는 지역 변형을 학습함으로써, 임의 궤적과 구성적 장면 생성을 따라 애니메이션된 장면을 합성하는 것을 가능하게 합니다.
- 우리의 접근 방식은 생성된 이동의 현실감과 양에 대한 현저한 개선을 가능하게 하며, 이는 질적 평가와 사용자 연구를 통해 평가합니다.
- 비디오 결과는 웹사이트 https://sherwinbahmani.github.io/tc4d/ 에서 볼 수 있습니다.

### [Octree-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D Gaussians](https://arxiv.org/abs/2403.17898)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2403.17898.png)

Vote: 8

Authors: Tao Lu, Linning Xu, Zhangkai Ni, Bo Dai, Lihan Jiang, Mulin Yu, Kerui Ren

- 최근의 3D 가우시안 스플래팅(3D-GS) 방법이 NeRF 기반 뉴럴 씬 표현보다 높은 렌더링 퀄리티와 효율성을 보여주었음에도 불구하고, 복잡한 디테일을 포함한 대규모 장면에서 많은 수의 가우시안 원시체로 인한 렌더링 병목 현상을 경험합니다.
- 이러한 한계는 멀리 떨어진 시점에서 두드러지며, 디테일이 다양한 장소에서 렌더링 속도의 일관성을 떨어뜨립니다.
- 또한, 휴리스틱 밀도 제어 작업을 통한 다양한 스케일에서의 디테일 포착에 어려움을 겪습니다.
- 이 논문은 Level-of-Detail(LOD) 기법에 영감을 받아 LOD-구조화된 3D 가우시안 접근 방식인 Octree-GS를 도입하여 장면 표현을 위한 디테일 분해를 지원합니다.
- 모델은 멀티해상도 고정점 세트에서 적절한 레벨을 동적으로 선택하여, 높은 퀄리티 렌더링 결과를 유지하면서 적응적 LOD 조정을 통해 일관된 렌더링 성능을 보장합니다.

### [Improving Text-to-Image Consistency via Automatic Prompt Optimization](https://arxiv.org/abs/2403.17804)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2403.17804.png)

Vote: 7

Authors: Oscar Mañas, Michal Drozdzal, Melissa Hall, Candace Ross, Adriana Romero-Soriano, Aishwarya Agrawal, Jack Urbanek, Pietro Astolfi, Adina Williams

- 본 논문은 텍스트-이미지(T2I) 생성 모델을 활용하여 매력적이고 사실적인 이미지를 생성하는 분야에서 이룩한 인상적인 진보에도 불구하고, 입력된 텍스트 프롬프트와 일관성을 유지하는 데 있어 여전히 어려움이 있음을 밝혀냈습니다.
- 모델의 미세 조정 필요, 근접한 텍스트 샘플에만 중점을 두는 기존 방식의 한계, 그리고 이미지 품질, 대표성 다양성, 텍스트-이미지 일관성 간의 불리한 트레이드오프에 영향을 받는 문제점들이 제시되었습니다.
- 본 연구에서는 대규모 언어 모델(LLM)을 활용하여 T2I 모델에서 프롬프트-이미지 일관성을 향상시키는 텍스트-이미지 최적화 프레임워크인 OPT2I를 소개합니다.
- 이 프레임워크는 사용자 프롬프트로부터 시작하여 일관성 점수를 최대화하는 것을 목표로 반복적으로 수정된 프롬프트를 생성합니다.
- MSCOCO 및 PartiPrompts 데이터셋에 대한 광범위한 검증을 통해 OPT2I가 초기 일관성 점수를 최대 24.9% 향상시키고, FID를 보존하며 생성된 데이터와 실제 데이터 간의 리콜을 증가시킬 수 있음이 확인되었습니다.
- OPT2I를 통한 연구는 대규모 언어 모델의 힘을 활용하여 더 신뢰할 수 있고 견고한 T2I 시스템을 구축하는 길을 제시합니다.

### [AniPortrait: Audio-Driven Synthesis of Photorealistic Portrait Animation](https://arxiv.org/abs/2403.17694)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2403.17694.png)

Vote: 6

Authors: Zhisheng Wang, Huawei Wei, Zejun Yang

- 본 연구에서는 오디오와 참조 초상화 이미지를 이용하여 고품질 애니메이션을 생성하는 새로운 프레임워크인 AniPortrait를 제안합니다.
- 우선 오디오에서 3D 중간 표현을 추출하고 이를 2D 얼굴 랜드마크 시퀀스로 투영하는 방식으로 두 단계로 방법론이 구성됩니다.
- 그 후, 강력한 확산 모델과 모션 모듈을 사용하여 랜드마크 시퀀스를 사실적이고 일관성 있는 초상 애니메이션으로 변환합니다.
- 실험 결과는 AniPortrait가 얼굴의 자연스러움, 포즈 다양성, 그리고 시각적 품질 면에서 우수함을 보여주며, 강화된 지각적 경험을 제공한다는 것을 입증합니다.
- 또한, 우리의 방법론은 얼굴 모션 편집이나 페이스 리엔액트먼트와 같은 분야에 효과적으로 적용될 수 있는 유연성과 조절 가능성 측면에서 상당한 잠재력을 보입니다.
- 코드 및 모델 가중치는 https://github.com/scutzzj/AniPortrait를 통해서 공개됩니다.

### [Fully-fused Multi-Layer Perceptrons on Intel Data Center GPUs](https://arxiv.org/abs/2403.17607)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2403.17607.png)

Vote: 6

Authors: Pascal Baehr, Xiangyi Zhang, Kai Yuan, Michael Paulitsch, Christoph Bauinger, Matthias Kirchhart, Darius Dabert, Pierre Boudier, Adrien Tousnakhoff

- 본 논문은 인텔 데이터 센터 GPU Max 1550을 대상으로 최적화된 다층 퍼셉트론(MLP)의 SYCL 구현을 제시한다.
- 성능 향상을 위해, MLP의 각 층에서 연산을 통합함으로써 느린 글로벌 메모리 접근을 최소화하고, 일반 레지스터 파일 및 공유 로컬 메모리 내의 데이터 재사용을 극대화한다.
- 이러한 방법은 산술 집약도를 크게 향상시켜, 특히 추론 단계에서 성능이 크게 개선되었음을 단순한 루프라인 모델로 보여준다.
- 인텔 데이터 센터 GPU에서 구현된 저자들의 접근법은 비슷한 CUDA 구현된 MLP와 비교하여, 엔비디아의 H100 GPU 상에서 추론에서 최대 2.84배, 훈련에서 1.75배 더 뛰어난 성능을 보여준다.
- 논문은 이미지 압축, 신경 광선 필드, 물리 정보 기계 학습과 같은 세 가지 주요 분야에서 SYCL 구현의 효율성을 보여준다.
- 모든 경우에서 저자들의 구현은 같은 인텔 GPU에서 제공되는 준비된 Intel Extension for PyTorch(IPEX) 구현을 최대 30배, 엔비디아의 H100 GPU에서 실행된 CUDA PyTorch 버전을 최대 19배 뛰어넘는 성능을 보여준다.
- 관련 코드는 https://github.com/intel/tiny-dpcpp-nn 에서 찾을 수 있다.

### [DreamPolisher: Towards High-Quality Text-to-3D Generation via Geometric Diffusion](https://arxiv.org/abs/2403.17237)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2403.17237.png)

Vote: 4

Authors: Yuanze Lin, Philip Torr, Ronald Clark

- DreamPolisher는 텍스트 설명에서 교차 시점 일관성과 복잡한 세부 사항을 학습할 수 있도록 만들어진 구시안 스플래팅(Gaussian Splatting) 기반 방법이다.
- 현재의 텍스트-3D 변환 방법은 종종 시점 일관성과 질감의 풍부함을 보장하지 못하지만, DreamPolisher는 이러한 문제를 해결하기 위해 제안되었다.
- 방법론은 두 단계로 구성되며, 첫 단계에서는 간략한 3D 생성이 기하학적 최적화를 통해 정제된다.
- 두 번째 단계에서는 ControlNet 기반의 정련기와 기하학적 일관성 용어를 결합하여 생성된 3D 자산의 질감 충실도와 전반적인 일관성을 향상시킨다.
- 다양한 객체 범주에 걸친 다양한 텍스트 프롬프트에 대한 경험적 평가를 통해 DreamPolisher가 텍스트 명령의 의미와 밀접하게 일치하는 일관되고 현실적인 3D 객체를 생성하는 데 효과적임을 보여준다.



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
