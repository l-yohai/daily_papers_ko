# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using ChatGPT.

Thanks to [@AK391](https://github.com/AK391) for great work.


## Daily Papers (2023-12-13)

### [FreeInit: Bridging Initialization Gap in Video Diffusion Models](https://arxiv.org/abs/2312.07537)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/UiQlReF6EXNfVLrI_8z2-.png)

Vote: 21

Authors: Tianxing Wu, Chenyang Si, Yuming Jiang, Ziqi Huang, Ziwei Liu

- 비디오 생성을 위한 확산 기반 모델의 발전에도 불구하고, 기존 모델들의 추론 결과는 시간적 일관성과 자연스러운 동작 측면에서 여전히 만족스럽지 못하다.
- 이 논문에서는 비디오 확산 모델의 노이즈 초기화에 대해 깊이 연구함으로써, 추론 품질 적합성에 영향을 미치는 암묵적인 훈련-추론 간극을 발견했다.
- 주요 발견은 다음 두 가지다: 1) 추론 시 사용되는 초기 레이텐트의 공간-시간적 빈도 분포가 훈련시의 것과 본질적으로 다르다는 것, 2) 탈잡음 과정이 초기의 저주파 구성요소에 의해 상당히 영향을 받는다는 것.
- 이러한 관찰에 기반하여, 우리는 간결하면서도 효과적인 추론 샘플링 전략인 FreeInit을 제안한다. 이는 확산 모델에 의해 생성된 비디오의 시간적 일관성을 대폭 향상시킨다.
- 추론 과정 중 초기 레이텐트의 공간-시간적 저주파 구성요소를 반복적으로 정제함으로써, FreeInit은 훈련과 추론 사이의 초기화 간극을 보상하고 생성 결과의 대상 외양과 시간적 일관성을 효과적으로 향상시키는 데 도움을 준다.
- 광범위한 실험을 통해 FreeInit은 추가적인 훈련 없이도 다양한 텍스트-투-비디오 생성 모델들의 생성 결과를 지속적으로 향상시킨다는 것을 입증했다.

### [FreeControl: Training-Free Spatial Control of Any Text-to-Image Diffusion Model with Any Condition](https://arxiv.org/abs/2312.07536)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/A4WizM2EGIE61ifD6ibgb.jpeg)

Vote: 12

Authors: Sicheng Mo, Fangzhou Mu, Kuan Heng Lin, Yanli Liu, Bochen Guan, Yin Li, Bolei Zhou

- 기존의 문제점으로, ControlNet과 같은 기술들은 텍스트-이미지(T2I) 확산 모델에 대해 공간적인 통제를 가능하게 하지만, 여러 가지 공간 조건, 모델 아키텍처, 체크포인트마다 별도의 보조 모듈을 훈련해야 한다는 점이 있었습니다.
- 이 연구에서 제시된 FreeControl은 다양한 조건, 아키텍처, 체크포인트를 동시에 지원하는 훈련 없는 방식으로 통제 가능한 T2I 생성을 위한 접근법입니다.
- FreeControl은 구조 가이드를 설계하여 가이드 이미지와의 구조 정렬을 촉진하고, 같은 시드를 사용하여 생성된 이미지 간의 외형 공유를 가능하게 하는 외형 가이드를 도입합니다.
- 폭넓은 질적 및 양적 실험을 통해, FreeControl이 다양한 사전 훈련된 T2I 모델에서 뛰어난 성능을 보임을 입증합니다.
- 특히, FreeControl은 여러 다른 아키텍처와 체크포인트에 대해 편리한 훈련 없는 제어를 용이하게 하고, 기존의 훈련 없는 방법들이 실패하는 도전적인 입력 조건에도 대응할 수 있으며, 훈련 기반의 접근법과 경쟁하는 합성 품질을 달성합니다.

### [DiffMorpher: Unleashing the Capability of Diffusion Models for Image Morphing](https://arxiv.org/abs/2312.07409)

[Watch Video](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/pPBftW9u_SvGH1HLzFDu-.mp4)
<div><video controls src="https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/pPBftW9u_SvGH1HLzFDu-.mp4" muted="false"></video></div>

Vote: 11

Authors: Kaiwen Zhang, Yifan Zhou, Xudong Xu, Xingang Pan, Bo Dai

- 확산 모델은 뛰어난 이미지 생성 품질을 달성했지만, 구조화되지 않은 잠재 공간으로 인해 두 이미지 샘플 간의 원활한 보간이 어려운 한계를 가지고 있다.
- 본 연구에서는 확산 모델을 사용하여 자연스럽고 부드러운 이미지 보간을 가능하게 하는 첫 번째 방법인 'DiffMorpher'를 소개한다.
- DiffMorpher는 두 이미지의 의미론적 세부 사항을 두 LoRAs를 이용하여 포착하고, LoRA 파라미터와 잠재 노이즈 사이를 보간함으로써 의미론적 전환을 부드럽게 하고 해당하는 부분이 자동적으로 나타나도록 한다.
- 또한, 연속되는 이미지 간의 부드러움을 강화하기 위해 주의 기울임(attention) 보간 및 주입 기술과 새로운 샘플링 일정을 제안한다.
- 광범위한 실험을 통해 DiffMorpher가 다양한 객체 범주에 걸쳐 이전 방법들보다 현저하게 우수한 이미지 모핑 효과를 달성하여, 확산 모델과 GAN의 주요 기능적 격차를 해소함을 보여준다.

### [VILA: On Pre-training for Visual Language Models](https://arxiv.org/abs/2312.07533)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/Yg0zBVAZpvV-jWC3DT13X.png)

Vote: 11

Authors: Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, Song Han

- 비주얼 언어 모델(VLM)의 사전 학습 과정에 관한 연구로, 기존의 대규모 언어 모델의 성공을 바탕으로 시각적 입력과의 결합을 모색했다.
- VLM을 효과적으로 사전 학습하기 위해 언어 모델을 단계적으로 확장하는 방식을 컨트롤 가능한 비교를 통해 조사했다.
- 언어 모델을 사전 학습하는 동안 고정시키면 무지도 학습에서 좋은 성능을 보이지만, 맥락 학습 능력을 위해서는 언어 모델의 학습 제한을 풀어야 함을 발견했다.
- 이미지-텍스트 쌍만의 데이터보다는 교차된 사전 학습 데이터가 더 유리하며, 이미지-텍스트 데이터에 텍스트만의 지시 데이터를 다시 혼합하는 것이 텍스트 전용 작업의 성능 저하를 보완하고 VLM 작업의 정확도를 향상시킨다는 점을 알아냈다.
- 개선된 사전 학습 레시피를 바탕으로 VILA라는 비주얼 언어 모델 패밀리를 구축하여, 주요 벤치마크에서 최첨단 모델들을 일관되게 능가하는 성능을 보였다.
- 다중 이미지 추론, 향상된 맥락 학습 능력, 그리고 더 나은 세계 지식 같은 VILA의 매력적인 특성들을 다중 모달 사전 학습을 통해 밝혀냈다.

### [Alignment for Honesty](https://arxiv.org/abs/2312.07000)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/9hBz8whuve-tqFEH6Gh1X.png)

Vote: 9

Authors: Yuqing Yang, Ethan Chern, Xipeng Qiu, Graham Neubig, Pengfei Liu

- 최근 연구는 큰 언어 모델(LLMs)의 유용성과 무해함을 인간의 의도에 따라 강화하기 위해 정렬 기술을 적용하는 데 상당한 진전을 이루었습니다.
- 본 논문에서는 LLM이 지식이 없을 때 질문에 답변하지 않도록 적극적으로 거부하면서도 지나치게 조심스럽지 않도록 하는 정직성을 위한 정렬의 중요성을 주장합니다.
- LLM의 지식 한계를 파악하는 것은 복잡한 문제이며, 이를 해결하기 위해 메트릭 개발, 벤치마크 생성, 그리고 훈련 방법론 측면에서 종합적인 해결책이 필요합니다.
- 연구팀은 공자의 《논어》에서 영감을 받아 "정직성"을 정의하고 이를 토대로 LLM의 정직성을 측정하는 메트릭을 개발하여 정렬 후의 진전을 정량화합니다.
- 또한 다른 작업의 성능을 희생하지 않으면서 정직성을 강조하는 여러 효율적인 미세 조정 기법을 도입한 유연한 훈련 프레임워크를 소개합니다.
- 광범위한 실험을 통해 이러한 정렬된 모델이 제안된 메트릭에 의해 표시된 바와 같이 정직성에서 뚜렷한 증가를 보였습니다.
- 연구를 촉진하기 위해 https://github.com/GAIR-NLP/alignment-for-honesty 에 정직성에 맞춘 모델, 정직성 정렬을 위한 훈련 및 평가 데이터셋, 개념 용어집 및 모든 관련 소스 코드를 공개합니다.

### [CCM: Adding Conditional Controls to Text-to-Image Consistency Models](https://arxiv.org/abs/2312.06971)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/mepuZ-w6HiUDMgotLQl1G.jpeg)

Vote: 8

Authors: Jie Xiao, Kai Zhu, Han Zhang, Zhiheng Liu, Yujun Shen, Yu Liu, Xueyang Fu, Zheng-Jun Zha

- 일관성 모델(CMs)은 시각 콘텐츠를 효율적으로 고품질로 생성하는데 있어 유망함을 보여주었지만, 사전 훈련된 CMs에 새로운 조건적 제어를 추가하는 방법은 아직 탐구되지 않았다.
- 본 기술 보고서에서는 CMs에 ControlNet과 유사한 조건적 제어를 추가하기 위한 대체 전략을 고려하고 세 가지 주요 발견을 제시한다.
- 1) 확산 모델(DMs)을 위해 훈련된 ControlNet을 CMs에 직접 적용하여 고수준 의미 제어는 가능하지만, 저수준의 세부 사항 및 리얼리즘 제어에서는 어려움이 있다.
- 2) CMs는 독립적인 급의 생성 모델로서, Song 등이 제안한 일관성 훈련을 사용하여 ControlNet을 처음부터 훈련할 수 있다.
- 3) 다양한 조건 하에서 경량의 어댑터를 일관성 훈련을 통해 공동으로 최적화함으로써, DMs 기반의 ControlNet을 CMs로 신속하게 전달할 수 있다.
- 이 세 가지 솔루션을 텍스트-이미지 일관성 모델을 사용하는 다양한 조건적 제어, 포함하여 에지, 깊이, 인간 포즈, 저해상도 이미지 및 텍스트와의 마스킹된 이미지를 가로질러 연구하였다.

### [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/TD3n0VQb1Cus5m8_0yFmX.png)

Vote: 7

Authors: Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner

- 본 논문에서는 언어 모델의 전방향 패스 중 활성화 수정을 통해 모델을 조종하는 새로운 방법인 Contrastive Activation Addition (CAA)를 제안합니다.
- CAA는 사실적 대비 허구적 반응 등 특정 행동의 긍정 및 부정 예시 쌍 사이의 잔류 스트림 활성화 차이를 평균내어 '조종 벡터'를 계산합니다.
- 사용자의 프롬프트 이후 모든 토큰 위치에 이 벡터를 긍정 혹은 부정 계수와 함께 추가함으로써, 목표 행동의 정도를 정밀하게 조절할 수 있습니다.
- 논문은 다중 선택형 행동 질문 데이터셋과 개방형 생성 작업을 사용하여 Llama 2 Chat에서 CAA의 효과를 평가합니다.
- CAA는 모델 행동을 현저하게 변화시키며, 파인튜닝이나 소수샘플 프롬프팅 같은 전통적인 방법보다 우월한 성능을 보이고, 능력 감소를 최소화합니다.
- 또한, 여러 활성화 공간 해석 방법을 활용하여 CAA의 기작에 대한 더 깊은 통찰을 얻을 수 있었습니다.
- CAA는 대규모 언어 모델(LLMs)에서 고차원 개념이 어떻게 표현되는지를 밝히는데도 도움이 됨으로써, 모델 출력을 정확히 조정하는 것뿐만 아니라 이해도를 증진시킵니다.

### [Interfacing Foundation Models' Embeddings](https://arxiv.org/abs/2312.07532)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/G4lX6x086B1w9XFt6ttvn.jpeg)

Vote: 6

Authors: Xueyan Zou, Linjie Li, Jianfeng Wang, Jianwei Yang, Mingyu Ding, Zhengyuan Yang, Feng Li, Hao Zhang, Shilong Liu, Arul Aravinthan, Yong Jae Lee, Lijuan Wang

- 본 논문에서는 기초 모델의 임베딩을 조정하기 위한 일반화된 인터페이스인 FIND를 제시합니다.
- 가벼운 트랜스포머 인터페이스만으로도, 기초 모델 가중치를 조정하지 않고도 통합된 이미지(세분화) 및 데이터셋 수준(검색) 이해를 가능하게 합니다.
- 제안된 인터페이스는 다양한 작업에 적용 가능하며(검색, 세분화 등), 같은 구조와 가중치를 사용합니다.
- 작업별 프로토타입은 주의 마스크와 임베딩 유형을 통해 구현할 수 있습니다.
- 이 인터페이스는 새로운 작업 및 모델에 적응 가능하여 확장성이 뛰어납니다.
- 멀티 태스크 멀티 모달 훈련의 이점을 통해, 제안된 인터페이스는 교차하는 공유 임베딩 공간을 생성합니다.
- 이 교차하는 임베딩 공간을 바탕으로, COCO 데이터셋에 새로운 훈련 및 평가 주석을 도입하는 FIND-Bench를 소개하고 있습니다.
- 접근 방식은 FIND-Bench에서 최첨단 성능을 달성하며, 표준 검색 및 세분화 설정에서도 경쟁력 있는 성능을 보입니다.
- 훈련, 평가 및 시연 코드와 데이터셋은 https://github.com/UX-Decoder/FIND 에서 공개되었습니다.

### [COLMAP-Free 3D Gaussian Splatting](https://arxiv.org/abs/2312.07504)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/DV8LfQsmYeC8g09vzSg5U.png)

Vote: 6

Authors: Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A. Efros, Xiaolong Wang

- 신경 렌더링은 현장 재구성과 새로운 시점 합성에서 인상적인 발전을 이루었지만, 정확한 사전 계산된 카메라 포즈에 크게 의존합니다.
- 카메라 포즈를 미리 처리하지 않고도 신경 복사장(NeRFs)을 훈련하기 위해 여러 노력이 이루어졌지만, NeRF의 내재적인 표현은 3D 구조와 카메라 포즈를 동시에 최적화하는 데 추가적인 도전을 제시합니다.
- 최근 제안된 3D 가우시안 스플래팅은 그것의 명시적인 포인트 클라우드 표현으로 새로운 기회를 제공합니다.
- 본 논문은 입력 비디오 스트림의 연속성과 명시적인 기하학적 표현을 활용해, 구조물 사진 측량(SfM) 전처리 없이 새로운 시점 합성을 수행합니다.
- 입출력 프레임을 순차적으로 처리하고, 카메라 포즈를 사전에 계산할 필요 없이 한 번에 하나의 입력 프레임을 취하여 3D 가우시안 집합을 점진적으로 확장합니다.
- 본 방법은 큰 동작 변경 하에서 시점 합성과 카메라 포즈 추정에서 이전 방법론을 상당히 향상시킵니다.
- 프로젝트 페이지는 https://oasisyang.github.io/colmap-free-3dgs 에서 확인할 수 있습니다.

### [Rethinking Compression: Reduced Order Modelling of Latent Features in Large Language Models](https://arxiv.org/abs/2312.07046)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/sqzUUhXvvM9ZIuL_mmT7v.png)

Vote: 6

Authors: Arnav Chavan, Nahush Lele, Deepak Gupta

- 대규모 언어 모델(Large Language Models, LLMs)의 규모가 방대하여 기존의 압축 방법론을 직접 적용하기 어려운 문제를 해결하기 위해, 이 논문은 특징 공간의 저차원 분해(low-rank decomposition) 및 가중치 공간의 재매개변수화(re-parameterization)에 기반한 참신한 압축 접근법을 소개합니다.
- 이 새로운 압축 기법은 층별(layer-wise) 방식으로 작동해 GPU 장치가 필요없으며, 메모리와 시간이 엄격한 제약 조건 내에서도 10억 규모의 모델을 압축할 수 있게 합니다.
- 제안된 방법은 행렬 분해를 활용하여 지배적인 구조화된 가지치기(structured pruning) 방법에 비해 우수한 효과를 입증함으로써 모델 압축 분야에서 상당한 발전을 나타냅니다.

### [Honeybee: Locality-enhanced Projector for Multimodal LLM](https://arxiv.org/abs/2312.06742)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/9VVzgks022Bj6Iq8RTKpu.png)

Vote: 6

Authors: Junbum Cha, Wooyoung Kang, Jonghwan Mun, Byungseok Roh

- 본 연구에서는 시각 프로젝터가 사전 훈련된 비전 인코더와 대규모 언어 모델(MLLMs)을 연결하는데 중요한 역할을 하여, 강력한 LLM 기능을 활용하는 동시에 시각적 이해를 깊게 할 수 있음을 지적하고 있다.
- 프로젝터에 있어서 시각 토큰의 수를 관리하는 능력의 유연성과 시각적 특성에서의 지역 맥락을 보존하는 것이 공간 이해에 필수적이라는 두 가지 주요 속성을 도출하였다.
- 이러한 발견을 바탕으로, 저자들은 두 가지 바람직한 속성을 효과적으로 만족시킬 수 있는 유연하고 지역성 강화된 새로운 프로젝터 디자인을 제안한다.
- 이와 함께, 다양하고 멀티페이스된 지시 데이터셋을 효과적으로 활용하기 위한 종합적인 전략을 제시한다.
- 광범위한 실험을 통해 개별 디자인 선택이 미치는 영향을 분석하였으며, Honeybee라 명명된 제안된 MLLM은 MME, MMBench, SEED-Bench, LLaVA-Bench를 포함한 다양한 벤치마크에서 이전의 최신 기술보다 현저히 우수한 성능을 보였다.
- 더 높은 효율성을 달성하는 Honeybee의 코드와 모델은 https://github.com/kakaobrain/honeybee에서 확인할 수 있다.

### [How Well Does GPT-4V(ision) Adapt to Distribution Shifts? A Preliminary Investigation](https://arxiv.org/abs/2312.07424)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/6etjFARZEbazSch-T6SxA.png)

Vote: 5

Authors: Zhongyi Han, Guanglin Zhou, Rundong He, Jindong Wang, Xing Xie, Tailin Wu, Yilong Yin, Salman Khan, Lina Yao, Tongliang Liu, Kun Zhang

- 기계 학습에서 훈련 시나리오와 다른 배포 조건으로부터 발생하는 분포 이동에 대한 일반화는 중요하며, 기후 모델링, 생의학, 자율 주행 등 여러 분야에서 필수적입니다.
- 대규모 사전학습과 과제 다양성을 특징으로 하는 기초 모델의 출현으로 인해 이러한 모델들의 분포 이동에 대한 적응성에 대한 관심이 증가했습니다.
- GPT-4V(ision)는 공개적으로 접근 가능한 가장 발전된 다모달 기초 모델로서, 비정상 탐지, 비디오 이해, 이미지 생성, 의료 진단 등 여러 분야에 걸쳐 광범위하게 적용되고 있습니다.
- 그러나 데이터 분포에 대한 그것의 견고성은 대체로 잘 알려지지 않았습니다.
- 이 연구는 GPT-4V의 분포 변화에 대한 적응성과 일반화 능력을 체계적으로 평가하며, CLIP과 LLaVA와 같은 주목할만한 모델들과의 비교 분석을 진행합니다.
- 이 조사는 GPT-4V의 제로샷 일반화를 자연, 의학, 분자 도메인을 포함하는 13개 다양한 데이터셋을 통해 깊이 있게 탐구합니다.
- 또한, 제어된 데이터 변화에 대한 그것의 적응성을 조사하고, 그 적응을 강화하기 위한 인-콘텍스트 학습의 효율성을 검토합니다.
- 우리의 발견은 GPT-4V의 분포 변화에 대한 능력 경계를 설명하며, 다양한 시나리오에서의 장점과 한계를 밝힙니다.
- 중요하게도, 이 조사는 인공지능 기초 모델이 어떻게 분포 이동에 일반화되는지에 대한 이해에 기여하며, 그들의 적응성과 견고성에 대한 중요한 통찰을 제공합니다.
- 코드는 https://github.com/jameszhou-gl/gpt-4v-distribution-shift 에서 공개적으로 이용 가능합니다.

### [Fast Training of Diffusion Transformer with Extreme Masking for 3D Point Clouds Generation](https://arxiv.org/abs/2312.07231)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/0X51hP2vD0fcaE1QkLXkC.png)

Vote: 4

Authors: Shentong Mo, Enze Xie, Yue Wu, Junsong Chen, Matthias Nießner, Zhenguo Li

- 최근 3D 포인트 클라우드를 고품질로 생성하게 하는 데 효과적인 것으로 나타난 디퓨전 트랜스포머의 비용 문제를 해결하기 위해, 'FastDiT-3D'라는 새로운 마스킹 디퓨전 트랜스포머를 제안한다.
- 이 모델은 마스킹된 자동인코더에서 영감을 받아 마스킹 된 화소화된 포인트 클라우드에서 탈잡음 과정을 동적으로 수행함으로써 훈련 비용을 크게 줄인다.
- 새로운 보케시언-인식 마스킹 전략을 사용하여 화소화된 포인트 클라우드에서 배경/전경 정보를 적응적으로 집약한다.
- FastDiT-3D는 거의 99%의 극단적인 마스킹 비율로 최신 성능을 달성하고, 다중 카테고리 3D 생성을 개선하기 위해 Mixture-of-Expert (MoE) 접근 방식을 도입한다.
- 각 카테고리가 다른 전문가들과 함께 상이한 디퓨전 경로를 학습함으로써 기울기 충돌을 완화한다.
- ShapeNet 데이터셋에 대한 실험 결과는 제안된 방법이 고품질 및 다양한 3D 포인트 클라우드 생성에서 최신 성능을 달성함을 보여준다.
- 우리의 FastDiT-3D는 단지 원래 훈련 비용의 6.5%를 사용하여 128 해상도 디퓨전 포인트 클라우드를 생성할 때 1-Nearest Neighbor 정확도 및 커버리지 지표를 향상시킨다.

### [Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations](https://arxiv.org/abs/2312.06674)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/nn0XgqkxcVD2HWh56_vWC.png)

Vote: 4

Authors: Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, Madian Khabsa

- 본 논문은 인간-AI 대화에 사용하기 위해 안전한 위험 분류(taxonomy)를 포함하는 LLM 기반 입력-출력 안전모델, Llama Guard를 소개합니다.
- 이 모델은 라마2-7b 모델이며, 고품질의 데이터셋에 대해 명령어 최적화(instruction-tuning)를 거쳐 생성된 응답과 프롬프트의 안전 위험을 분류하는 데 사용됩니다.
- Llama Guard는 기존 벤치마크인 OpenAI Moderation Evaluation 데이터셋과 ToxicChat에서 강력한 성능을 보여 주며, 현재 이용 가능한 내용 검토 도구들과 동등하거나 그 이상의 결과를 달성했습니다.
- 다중 클래스 분류와 이진 결정 점수를 생성하는 언어 모델로서 동작하며, 명령어 미세 조정 기능을 통해 특정 사례에 맞는 분류 카테고리 조정 및 다양한 분류 체계에서의 제로 샷 혹은 퓨 샷 프롬프팅을 가능하게 합니다.
- 연구자들이 AI 안전을 위해 Llama Guard 모델 가중치를 개발하고 적응해나갈 수 있도록, 논문 저자들은 이를 공개함으로써 지속적인 커뮤니티의 발전을 지원합니다.

### [PEEKABOO: Interactive Video Generation via Masked-Diffusion](https://arxiv.org/abs/2312.07509)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/BEB9BbR70W_YtxAyw6TvR.qt)

Vote: 3

Authors: Yash Jain, Anshul Nasery, Vibhav Vineet, Harkirat Behl

- 최근 텍스트를 이용한 비디오 생성 분야는 고품질, 현실적인 비디오를 생성할 수 있는 최신 모델이 등장하면서 큰 진보를 이루었다.
- 그러나 이 모델들은 사용자가 상호작용적으로 비디오를 제어하고 생성하는 능력이 부족하여, 새로운 응용 분야의 가능성을 탐색하기 위해 이 문제를 해결하고자 한다.
- 우리는 분할 문헌의 최근 발전에서 영감을 얻어, 시공간 마스크 주의 모듈인 Peekaboo를 제안하고, 이는 기존의 비디오 생성 모델에 추가하여 시공간 제어가 가능하게 한다.
- 이 모듈은 훈련이 필요 없으며 추론 시간도 추가로 발생하지 않으며, 상호작용적인 비디오 생성을 위한 컨트롤을 가능하게 한다.
- 상호작용적인 비디오 생성 작업을 평가하기 위한 벤치마크도 제시한다.
- 폭넓은 질적 및 양적 평가를 통해, Peekaboo가 기존 모델 대비 최대 3.8배까지 mIoU 성능 향상을 가져올 수 있는 제어 비디오 생성을 활성화한다는 것을 입증한다.

### ["I Want It That Way": Enabling Interactive Decision Support Using Large Language Models and Constraint Programming](https://arxiv.org/abs/2312.06908)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/taoZ0wjnCyOWlTnmM8mYU.png)

Vote: 2

Authors: Connor Lawless, Jakob Schoeffer, Lindy Le, Kael Rowan, Shilad Sen, Cristina St. Hill, Jina Suh, Bahar Sarrafzadeh

- 이 연구는 사용자의 선호도를 정확하게 모델링하는 것이 의사 결정 지원 시스템의 성공에 있어 중요한 요소임을 밝힙니다.
- 심리학 연구에 따르면, 사용자는 종종 정보를 얻는 과정에서 선호도를 개발하므로, 개인화 시스템을 개발하는데 있어 시스템-사용자 상호작용의 중요성이 강조됩니다.
- 본 논문은 대규모 언어 모델(Large Language Models, LLMs)과 제약 프로그래밍(Constraint Programming)을 결합하여 인터랙티브 의사결정 지원을 가능하게 하는 새로운 접근법을 소개합니다.
- 일상적으로 정보 작업자들이 직면하는 회의 일정 조정이라는 맥락을 통해 이 하이브리드 프레임워크를 연구합니다.
- 연구팀은 세 가지 연구를 수행하여 새로운 프레임워크를 평가하는데, 일기 연구(n=64)로 맥락에 따른 일정 조정 선호도를 특성화하고, 시스템 성능의 정량적 평가를 행하며, 프로토타입 시스템을 사용한 사용자 연구(n=10)를 진행합니다.
- 이 작업은 반복적 선호도 수집과 의사 결정 시스템 설계를 위한 하이브리드 LLM 및 최적화 접근법의 잠재력을 강조하며, 인간-시스템 공동 의사결정 과정을 지원하는 시스템을 구축하기 위한 설계 고려사항을 제시합니다.



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
