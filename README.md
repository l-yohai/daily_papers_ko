# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using ChatGPT.

Thanks to [@AK391](https://github.com/AK391) for great work.


## Daily Papers (2023-11-21)

### [Exponentially Faster Language Modelling](https://arxiv.org/abs/2311.10770)
![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/J4Fjt08cj7la_ElXbOyoR.png)
Authors: Peter Belcak, Roger Wattenhofer

- 언어 모델은 개별 추론에 수많은 뉴런 중 일부분만을 사용할 필요가 있음을 보여주는 논문에서, FastBERT라는 BERT 변형을 소개하고 있습니다.
- FastBERT는 유사한 BERT 모델들과 동등한 성능을 유지하면서 추론 단계에서 0.3%의 뉴런만을 사용하며, 각 계층 추론에 4095개 중 단 12개의 뉴런을 선택적으로 활성화합니다.
- 이는 피드포워드 네트워크 대신 빠른 피드포워드 네트워크(FFFs)를 사용함으로써 달성되었습니다.
- 현재는 조건부 뉴럴 실행의 전체 가속 잠재력을 해제할 효율적인 구현체가 없지만, 최적화된 기본 피드포워드 구현에 비해 78배 빠른 고수준 CPU 코드를 제공하며, 해당 배치 피드포워드 추론에 비해 40배 빠른 PyTorch 구현체를 제공합니다.
- 연구팀은 학습 코드, 벤치마킹 설정 및 모델 가중치를 공개합니다.

### [Make Pixels Dance: High-Dynamic Video Generation](https://arxiv.org/abs/2311.10982)

<iframe src="https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/D-WQoAl7gR1AwRZ0RXNkW.mp4" ></iframe>

<video width="320" height="240" controls>
  <source src="https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/D-WQoAl7gR1AwRZ0RXNkW.mp4" type="video/mp4">
</video>

Authors: Yan Zeng, Guoqiang Wei, Jiani Zheng, Jiaxin Zou, Yang Wei, Yuchen Zhang, Hang Li

- 본 논문은 인공지능 분야에서 동적인 모션과 고급 비주얼 효과가 풍부한 고동적 비디오 생성의 어려움을 다루고 있습니다.
- 현재 최고 수준의 비디오 생성 방법들은 주로 텍스트-비디오 생성에 중점을 두며, 고화질에도 불구하고 최소한의 움직임을 지닌 비디오 클립을 생산하는 경향이 있습니다.
- 저자들은 비디오 생성을 위해 오직 텍스트 지시만을 의존하는 것은 불충분하며 최적이 아니라고 주장합니다.
- 이 논문은 텍스트 지시 뿐만 아니라 첫 번째와 마지막 프레임에 대한 이미지 지시를 결합하는 새로운 접근법인 PixelDance를 소개합니다.
- 공개 데이터로 훈련된 PixelDance가 복잡한 장면과 복잡한 모션을 가진 비디오를 합성하는 데 있어 더 뛰어난 능력을 보임을 실험 결과가 입증하며, 비디오 생성 분야의 새로운 기준을 제시합니다.


### [Orca 2: Teaching Small Language Models How to Reason](https://arxiv.org/abs/2311.11045)
![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/wonQoQEEuCjCtDCJibROj.png)
Authors: Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Xuxi Chen, Kriti Aggarwal, Hamid Palangi, Guoqing Zheng, Corby Rosset, Hamed Khanpour, Ahmed Awadallah

- Orca 1은 설명 추적과 같은 풍부한 신호로부터 학습하여, BigBench Hard와 AGIEval 같은 벤치마크에서 기존의 지시어 조절 모델들을 능가하는 성능을 보여준다.
- Orca 2는 더 나은 훈련 신호가 작은 언어 모델(LMs)의 추론 능력을 향상시키는 방법을 탐구한다.
- 기존의 작은 LMs 훈련 연구는 더 능력 있는 모델들의 출력물을 모방하는 모방 학습에 의존해 왔으나, 이러한 방법은 작은 모델들의 잠재력을 제한할 수 있다.
- Orca 2는 작은 모델들이 다양한 해결 전략(단계별, 상기한 후 생성, 상기-추론-생성, 직접 대답 등)을 사용하는 것을 가르친다.
- 더 중요하게는 모델이 각 작업에 가장 효과적인 해결 전략을 스스로 결정하도록 학습하는 것을 목표로 한다.
- 대략 100개의 작업 및 36,000개의 고유 프롬프트에 해당하는 15가지 다양한 벤치마크를 사용하여 Orca 2를 평가한다.
- Orca 2는 비슷한 크기의 모델들을 크게 능가하고, zero-shot 설정에서 고급 추론 능력을 테스트하는 복잡한 작업에서 모델 크기가 5-10배 더 큰 모델들의 성능과 비슷하거나 뛰어난 성능을 달성한다.
- 더 많은 연구를 장려하기 위해 Orca 2를 오픈 소스로 제공한다, 작은 LMs의 개발, 평가 및 조정에 대한 추가 연구를 장려함을 목적으로 한다.

### [MultiLoRA: Democratizing LoRA for Better Multi-Task Learning](https://arxiv.org/abs/2311.11501)
![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/53uzqP9Y7qvyV8KrmEq0u.png)
Authors: Yiming Wang

- LoRA는 특정 작업을 위해 대규모 언어모델(LLMs)을 조정할 때 뛰어난 자원 효율성과 비교적 우수한 성능을 보여줍니다.
- 그러나 LoRA의 명시적인 저랭크(low-rank) 구조는 복잡한 다작업 시나리오에서의 조정 성능에 제한을 주며, 상위 특이 벡터들이 지배적이어서 미세조정이 덜 중요한 단위 변환으로 분해됩니다.
- 본 논문에서는 LoRA에서 관찰된 상위 특이 벡터의 우세를 줄임으로써 보다 나은 다작업 조정을 위한 MultiLoRA를 제안합니다.
- MultiLoRA는 수평으로 LoRA 모듈을 확장하고 조정 매트릭스의 파라미터 초기화를 변경하여 파라미터 의존성을 감소시키고, 더 균형 잡힌 단위 부공간을 생성합니다.
- 지시 이행, 자연어 이해, 세상 지식에 대한 데이터셋을 혼합하여 의미론적으로나 구문론적으로 다른 샘플들을 포함하는 특화된 훈련 데이터를 전례 없이 구축합니다.
- 추가 파라미터가 단지 2.5%임에도 불구하고, MultiLoRA는 다수의 벤치마크와 모델 스케일에서 단일 LoRA 버전 및 미세조정을 능가하는 성능을 보여줍니다.
- MultiLoRA의 가중치 업데이트 매트릭스에 대한 추가 분석은 상위 특이 벡터에 대한 의존도가 줄어들었으며, 더 민주적인 단위 변환 기여도를 나타냅니다.

### [System 2 Attention (is something you might need too)](https://arxiv.org/abs/2311.11829)
![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/lC0r8M6JFyi_kvKygxRaG.png)
Authors: Jason Weston

- 트랜스포머 기반의 대형 언어 모델(Large Language Models, LLMs)의 소프트 어텐션은 문맥에서 관련 없는 정보를 잠재적 표현에 통합하는데, 이는 다음 토큰 생성에 부정적인 영향을 미칩니다.
- 이 문제를 해결하기 위해, 우리는 언어 모델이 자연어로 이유를 추론하고 지시를 따르는 능력을 활용하여 주목해야 할 대상을 결정하는 '시스템 2 어텐션(System 2 Attention, S2A)'을 도입했습니다.
- S2A는 입력 문맥을 중요한 부분만 포함하도록 재생성하고, 최종적인 응답을 이끌어내기 위해 재생성된 문맥에 주목합니다.
- 실험에서 S2A는 의견이나 관련 없는 정보를 포함하는 세 가지 작업(질문 답변, 수학 문제 해결, 장문 생성)에서 표준 어텐션 기반 LLMs보다 성능이 우수하며, 사실성과 객관성을 증가시키고 아첨을 감소시키는 효과가 있음을 확인했습니다.

### [Adapters: A Unified Library for Parameter-Efficient and Modular Transfer Learning](https://arxiv.org/abs/2311.11077)
![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/-hvy4p00GWBmMx67NyR3K.png)
Authors: Clifton Poth, Indraneil Paul, Sukannya Purkayastha, Leon Engländer, Timo Imhof, Ivan Vulić, Sebastian Ruder

- 'Adapters'는 대규모 언어 모델에서 매개변수 효율성과 모듈성을 갖춘 전이 학습을 위한 오픈 소스 라이브러리를 소개합니다.
- 10개의 다양한 어댑터 방법론을 통합하여 사용자의 편의성 및 유연한 구성을 제공하는 통합 인터페이스를 제공합니다.
- 연구원과 실무자가 복잡한 어댑터 설정을 설계할 수 있도록 구성 블록을 통해 어댑터의 모듈성을 활용할 수 있습니다.
- 다양한 자연어 처리(NLP) 작업에서 전체 미세조정 방법과 비교하여 이 라이브러리의 효능을 입증합니다.
- 'Adapters'는 기존의 미세조정 패러다임의 도전을 해결하고 더 효율적이고 모듈식의 전이 학습을 촉진하는 강력한 도구를 제공합니다.
- 이 라이브러리는 https://adapterhub.ml/adapters 사이트를 통해 사용할 수 있습니다.

### [Text-to-Sticker: Style Tailoring Latent Diffusion Models for Human Expression](https://arxiv.org/abs/2311.10794)
![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/iaJY8sQLnqKAaVHc-pLqT.png)
Authors: Animesh Sinha, Arantxa Casanova, Winnie Zhang, Licheng Yu, Sonal Gupta, Dhruv Mahajan

- 본 연구에서는 새로운 "Style Tailoring" 방식을 소개하며, 이를 통해 Latent Diffusion Models (LDMs)을 고화질, 알맞은 프롬프트 정렬, 그리고 다양한 장면이 특징인 특정 도메인에 미세조정하는 방법을 제시한다.
- 스티커 이미지 생성을 목표 도메인으로 선택하였으며, 이는 대규모 LDMs가 생성하는 사실적인 샘플들과 크게 달라진다.
- 기존의 텍스트-이미지 모델인 Emu를 시작점으로 하여, 사실적인 모델을 이용한 프롬프트 엔지니어링이 스티커 생성에 있어 프롬프트 정렬과 장면 다양성에서 떨어짐을 보여준다.
- 이러한 단점을 극복하기 위해, 먼저 Emu를 약한 감독을 받는 수백만 개의 스티커 같은 이미지들을 사용하여 미세조정함으로써 다양성을 유도한다.
- 이후에 모델 생성물에서 사람이 직접 선택한 (HITL) Alignment와 Style 데이터셋을 통해 프롬프트 정렬과 스타일 정렬을 각각 개선하기 위해 미세조정을 진행한다.
- 이러한 순차적 미세조정은 스타일 정렬과 프롬프트 정렬의 향상 간의 균형을 맞추는 데 있어서 트레이드오프를 제시한다.
- 이 트레이드오프를 해결하기 위해, "Style Tailoring"이라는 새로운 미세조정 방법을 제안하며, 이는 내용과 스타일 분포를 함께 조절하면서 최상의 균형을 달성한다.
- 평가 결과, 우리의 방법은 기본 Emu 모델에 스티커 생성을 위한 프롬프트 엔지니어링을 적용했을 때에 비해 시각적 품질을 14%, 프롬프트 정렬을 16.2%, 장면 다양성을 15.3% 향상시킴을 보여준다.

### [Memory Augmented Language Models through Mixture of Word Experts](https://arxiv.org/abs/2311.10768)
![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/w3iQFRkBxEcuOM8vYb8Ey.png)
Authors: Cicero Nogueira dos Santos, James Lee-Thorp, Isaac Noble, Chung-Ching Chang, David Uthus

- 언어 모델의 파라미터 수를 늘리는 것이 성능 향상에 효과적임이 입증되었으나, 밀집 모델의 경우 모델 크기 증가는 계산 요구량을 비례적으로 증가시킵니다.
- 본 연구에서는 풍부한 지식을 갖춘 어휘 기반 라우팅 기능과 전문가들을 포함한 Mixture-of-Experts (MoE) 스타일 모델을 통해 학습 용량과 FLOPs를 공격적으로 분리하기 위한 접근법을 모색합니다.
- 제안하는 '단어 전문가의 혼합(Mixture of Word Experts, MoWE)'이라는 방식은 희소 메모리 역할을 하는 대규모 단어별 전문가 세트를 포함한 메모리 증강 모델로 볼 수 있습니다.
- MoWE는 유사한 FLOPs를 가진 T5 모델 계열과 비교하여 다양한 NLP 과제에서 현저히 뛰어난 성능을 보여줍니다.
- 또한 MoWE는 지식 집약적인 과제에서 표준 MoE 모델을 능가하며, 종종 희소 메모리 검색을 위한 맞춤형 메커니즘이 필요한 더 복잡한 메모리 증강 방식과 비슷한 성과를 보입니다.


### [LucidDreamer: Towards High-Fidelity Text-to-3D Generation via Interval Score Matching](https://arxiv.org/abs/2311.11284)
![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/rsKut6q6m7cpsUBzsmF-V.png)
Authors: Yixun Liang, Jiantao Lin, Haodong Li, Xiaogang Xu, Yingcong Chen

- 본 논문은 텍스트-3차원 생성에서 상세하고 고품질의 3D 모델 렌더링에 어려움을 겪고 있는 점을 인식하고, 이를 해결하기 위해 새로운 접근법인 간격 점수 매칭(Interval Score Matching, ISM)을 제안합니다.
- 기존의 점수 증류 샘플링(Score Distillation Sampling, SDS) 기반 방법들이 3D 모델의 업데이트 방향에 있어 불일치와 저품질 결과를 초래하는 문제를 초래한다는 것을 지적하며, ISM은 이러한 과잉 평활화 효과를 방지하기 위해 결정적 확산 궤적과 간격 기반의 점수 매칭을 활용합니다.
- 또한, 3D 가우시안 스플래팅 기법을 텍스트-3D 생성 파이프라인에 통합하여, 모델의 품질과 훈련 효율성 면에서 상태-의-예술적 수준을 대폭 개선함을 실험적으로 입증합니다.

### [AutoStory: Generating Diverse Storytelling Images with Minimal Human Effort](https://arxiv.org/abs/2311.11243)
![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/cSMBrukPVtZyza4FAEX89.png)
Authors: Wen Wang, Canyu Zhao, Zhekai Chen, Kecheng Zheng

- 이 연구는 텍스트로 묘사된 스토리에 부합하는 일련의 이미지를 생성하는 스토리 시각화를 목표로 한다.
- 생성된 이미지는 높은 품질, 텍스트 설명과의 일치, 그리고 등장인물 신원의 일관성을 만족시켜야 한다.
- 복잡성 때문에 기존 방법들은 특정 등장인물과 시나리오만 고려하거나 사용자로 하여금 이미지별로 제어 조건(예: 스케치)을 제공하도록 하여 실제 응용에는 부적합하였다.
- 이에 대응하기 위해 본 연구는 최소한의 인간 상호작용으로 다양하고, 고품질이며, 일관된 스토리 이미지 세트를 효과적으로 생성할 수 있는 자동화된 스토리 시각화 시스템을 제안한다.
- 연구팀은 대규모 언어 모델의 이해력과 계획 능력을 사용하여 레이아웃 계획을 세우고, 대규모 텍스트-이미지 모델을 활용해 레이아웃에 기반한 정교한 스토리 이미지를 생성한다.
- 성공적인 레이아웃 계획에는 희소 제어 조건(예: 바운딩 박스)이 적합하며, 고품질 이미지 콘텐츠 생성에는 밀접한 제어 조건(즉, 스케치나 키포인트)이 적합하다는 점을 실증적으로 발견했다.
- 이 연구는 바운딩 박스 레이아웃을 스케치나 키포인트 제어 조건으로 변환하는 밀접한 조건 생성 모듈을 고안하여, 이미지 품질을 향상시키고 사용자 상호작용을 용이하고 직관적으로 만든다.
- 또한, 인간 노동에 의존하지 않고 일관된 여러 시점의 등장인물 이미지를 생성하는 간단하면서도 효과적인 방법을 제안한다.

### [GPQA: A Graduate-Level Google-Proof Q&A Benchmark](https://arxiv.org/abs/2311.12022)
![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/rHGl2R_Q4q_BWQDqjfvey.png)
Authors: David Rein, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani

- 이 연구는 생물학, 물리학, 화학 분야의 전문가들이 작성한 448개의 객관식 문제로 구성된 도전적인 데이터셋 GPQA를 소개합니다.
- 해당 문제들은 고품질이며 상당히 어려워, 해당 분야에서 박사 학위를 가지고 있거나 박사 과정에 있는 전문가들조차 문제를 인지하지 못한 명백한 실수를 감안하였을 때 74%의 정확도(실제는 65%)를 기록했습니다.
- 인터넷을 무제한으로 활용하고 평균 30분 이상의 시간을 할애함에도 불구하고, 숙련된 비전문가 검증자들은 34%의 정확도에 그쳤습니다(즉, 문제들이 "구글에 의한 검증이 불가능").
- 최첨단 인공지능 시스템에게도 이 질문들은 어려워, GPT-4 기반의 가장 강력한 베이스라인 모델조차 39%의 정확도에 그쳤습니다.
- 미래의 인공지능 시스템을 사용하여 새로운 과학 지식 개발과 같이 매우 어려운 질문에 답하기 위해서는, 전문 지식을 가진 인간이 그 출력을 감독할 수 있는 확장 가능한 감독 방법을 개발해야 합니다.
- 숙련된 비전문가와 최첨단 AI 시스템 모두에게 어려운 GPQA의 난이도는 현실적인 확장 가능한 감독 실험을 가능하게 하며, 인간 전문가가 인간의 능력을 뛰어넘는 AI 시스템으로부터 신뢰할 수 있는 정보를 얻을 수 있는 방법을 고안하는 데 도움이 될 수 있습니다.

### [TPTU-v2: Boosting Task Planning and Tool Usage of Large Language Model-based Agents in Real-world Systems](https://arxiv.org/abs/2311.11315)
![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/umQyHD5UPMjR04BWdA_Y4.png)
Authors: Jingqing Ruan, Yihong Chen, Hangyu Mao, Rui Zhao

- 대규모 언어 모델(LLMs)은 API와 같은 외부 도구를 사용하는 작업 계획을 필요로 하는 작업에 능숙함을 보여주었지만, 실제 복잡한 시스템은 작업 계획과 도구 사용에 대해 세 가지 일반적인 도전을 제시합니다.
- 이 논문은 실제 시스템에서 LLM 기반 에이전트의 작업 계획 및 도구 사용(TPTU) 능력을 향상시키기 위한 포괄적인 프레임워크를 제안합니다.
- 프레임워크는 세 가지 주요 구성요소를 포함하고 있는데, (1) API 리트리버는 방대한 어레이에서 가장 관련성있는 API를 선택합니다; (2) LLM 파인튜너는 기본 LLM을 조정하여 미세 조정된 LLM이 작업 계획 및 API 호출에 더 용이하게 할 수 있도록 합니다; (3) 데모 셀렉터는 구별하기 어려운 API와 관련된 서로 다른 데모를 적응형으로 검색하여 컨텍스트 학습을 위해 사용됩니다.
- 실제 상용 시스템과 오픈 소스 학술 데이터셋을 사용하여 분석한 결과, 개별 구성요소뿐만 아니라 통합된 프레임워크의 효과성이 명확하게 입증되었습니다.

### [ToolTalk: Evaluating Tool-Usage in a Conversational Setting](https://arxiv.org/abs/2311.10775)
![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/hWXP1WW3AZsku69WnSX9e.png)
Authors: Nicholas Farn, Richard Shin

- 대화 설정에서 대화형 사용자 인터페이스의 툴 사용을 평가하기 위한 새로운 벤치마크인 ToolTalk를 소개합니다.
- 이 연구는 대규모 언어 모델(Large Language Models, LLMs)의 추론 및 의사결정 능력이 크게 향상되었으며 자연스러운 대화를 유지할 수 있음을 보여줍니다.
- 최신 연구들은 개인정보나 최신 정보에 접근하고 사용자를 대신해 행동을 수행할 수 있는 외부 도구로 LLM 기반 도우미를 강화하고자 합니다.
- 사용자의 복잡한 의도를 대화를 통해 명시하고 다단계 툴 사용을 요구하는 ToolTalk 벤치마크를 도입합니다.
- ToolTalk는 7개의 플러그인에 속하는 28개의 도구를 포함하고 있으며 모든 도구의 완전한 시뮬레이션 구현을 포함하여 실행 피드백에 의존하는 도우미의 완전 자동화된 평가를 허용합니다.
- ToolTalk는 정보 검색이나 참조를 위한 도구뿐만 아니라 외부 세계에 실제 영향을 미치는 도구를 강조합니다.
- GPT-3.5와 GPT-4는 ToolTalk에서 각각 26% 및 50%의 성공률을 기록하였습니다.
- 오류 분석을 통해 세 가지 주요 카테고리를 밝히고 향후 개선을 위한 몇 가지 방향을 제시합니다.
- https://github.com/microsoft/ToolTalk 에서 ToolTalk를 공개합니다.

### [ProAgent: From Robotic Process Automation to Agentic Process Automation](https://arxiv.org/abs/2311.10751)
![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/PY8YpZ0z3q3JMPWebL51q.png)
Authors: Yining Ye, Xin Cong, Yujia Qin, Huadong Wang

- 고대 수차부터 로봇 프로세스 자동화(RPA)에 이르기까지 자동화 기술은 역사를 통해 인간을 고된 작업에서 해방시켰지만 RPA는 인간과 같은 지능을 필요로 하는 작업, 특히 워크플로우 구축의 복잡한 설계와 실행에서의 역동적 의사 결정에서 어려움을 겪고 있습니다.
- 이 논문에서는 대행 프로세스 자동화(APA)라는 새로운 자동화 패러다임을 소개하는데, 이는 Large Language Models(LLMs)를 기반으로 한 에이전트들을 사용하여 고급 자동화를 가능하게 해주며 생성 및 실행 과정에서의 인간 노동을 에이전트에게 오프로딩합니다.
- 연구자들은 ProAgent라는 LLM 기반 에이전트를 개발하여 인간의 지시로부터 워크플로우를 만들고, 전문화된 에이전트들을 조정하여 복잡한 결정을 내릴 수 있도록 했습니다.
- 실증 실험을 통해 워크플로우의 구축과 실행 절차를 상세하게 기술하며, APA의 실행 가능성을 보여주고 에이전트에 의해 추진되는 자동화의 새로운 패러다임의 가능성을 밝혔습니다.
- 해당 연구의 코드는 https://github.com/OpenBMB/ProAgent 에서 공개되어 있습니다.

### [GPT-4V(ision) for Robotics: Multimodal Task Planning from Human Demonstration](https://arxiv.org/abs/2311.12015)
![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/ZbjKeKEamGahtKOrLjJek.png)
Authors: Naoki Wake

- GPT-4V(ision)라는 범용 시각 언어 모델을 향상시켜 인간의 행동을 관찰하고 로봇 조작을 용이하게 하는 파이프라인을 소개합니다.
- 이 시스템은 인간이 수행하는 작업을 비디오로 분석하고, 환경 및 행동 상세 정보를 텍스트로 변환하여 실행 가능한 로봇 프로그램을 생성합니다.
- 이후 작업 계획에 따라 시각 시스템이 비디오를 재분석하며, 개방 어휘형 객체 탐지기를 사용하여 객체 이름을 실제 객체에 연결합니다.
- 손과 객체의 관계에 주목함으로써, 잡거나 놓는 순간을 탐지하여 시공간적 연결을 가능하게 합니다.
- 이러한 연결은 시각 시스템이 잡기 유형, 경유지, 신체 자세 등과 같은 추가적인 지원 데이터를 수집할 수 있게 합니다.
- 다양한 시나리오에서의 실험을 통해 이 방법이 인간의 시연으로부터 제로샷 방식으로 실제 로봇 작업을 성공적으로 달성함을 입증했습니다.
- GPT-4V 및 GPT-4의 프롬프트는 해당 프로젝트 페이지에서 확인할 수 있습니다: https://microsoft.github.io/GPT4Vision-Robot-Manipulation-Prompts/.

### [M$^{2}$UGen: Multi-modal Music Understanding and Generation with the Power of Large Language Models](https://arxiv.org/abs/2311.11255)
![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/27VY9R5sOdajsjesLABeG.png)
Authors: Atin Sakkeer Hussain, Shansong Liu

- 대규모 언어 모델(LLMs)을 활용한 연구가 급증하는 가운데, 이 모델들은 텍스트, 말소리, 이미지, 비디오 등 다양한 매체를 포괄적으로 이해하고 사용자의 의도를 파악해 원하는 이미지, 비디오, 음악 등을 생성하는데 활용되지만, 이해와 생성을 결합하는 연구는 여전히 초기 단계에 있다.
- 다중 모델 음악 이해 및 생성(M$^{2}$UGen) 프레임워크는 음악을 이해하고 생성하기 위하여 LLM의 능력을 통합하며, 사전 훈련된 MERT, ViT, 그리고 ViViT 모델을 활용하여 각각 음악, 이미지, 비디오 등 다양한 영감의 원천에서 창의적 가능성을 이끌어낸다.
- 음악 생성을 위해 AudioLDM 2와 MusicGen을 탐구하고 LLaMA 2 모델을 통해 다중 모델 이해와 음악 생성 사이의 연결다리를 놓는다.
- MU-LLaMA 모델 사용을 통해 텍스트/이미지/비디오-음악 생성을 지원하는 광범위한 데이터셋을 생성하며, 이는 M$^{2}$UGen 프레임워크 훈련에 도움을 준다.
- 제안된 프레임워크에 대해 철저히 평가하였고 실험 결과 현재 최신 모델들의 성능을 달성하거나 능가함을 보여준다.



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
