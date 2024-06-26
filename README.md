# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using ChatGPT.

Thanks to [@AK391](https://github.com/AK391) for great work.


## Daily Papers (2024-07-01)

### [HuatuoGPT-Vision, Towards Injecting Medical Visual Knowledge into Multimodal LLMs at Scale](https://arxiv.org/abs/2406.19280)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.19280.png)

Vote: 46

Authors: Shunian Chen, Ruyi Ouyang, Guiming Hardy Chen, Junying Chen, Benyou Wang, Xiang Wan, Zhenyang Cai, Ke Ji, Ruifei Zhang, Anningzhe Gao, Xidong Wang, Guangjun Yu

- **What's New**: 이번 논문은 의료 분야에서의 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLM) 성능을 향상시키기 위한 새로운 접근 방식을 제시합니다. 저자들은 PubMed 데이터를 정교하게 선택하고 GPT-4V를 활용한 'unblinded' 방식으로 포맷팅하여, 고품질의 대규모 의료 멀티모달 데이터셋 'PubMedVision'을 구축했습니다.
- **Technical Details**: PubMedVision은 130만 개의 의료 이미지와 텍스트 페어로 구성되었습니다. 이를 위해 PubMed의 914,960 개 정제된 의료 이미지와 그에 해당하는 텍스트를 활용하여 의료 시각적 질문 응답(Visual Question Answering, VQA) 데이터를 생성했습니다. 또한, 'blinded' 포맷팅 대신 'unblinded' MLLM을 사용하여 데이터를 정제하고 시각적 정보와 콘텍스트를 더 정확히 연관 지었습니다.
- **Performance Highlights**: 실험 결과, PubMedVision을 활용한 MLLM은 기존 MLLM에 비해 의료 멀티모달 작업에서 뛰어난 성능을 보였습니다. LLaVA-v1.5-LLaMA-3-8B는 공개된 MLLM 중 최고 성능을 달성했습니다. 또한, 전문가 검토와 실험 결과에서 현재 데이터 생성 방법에 비해 높은 데이터 품질을 보여주었습니다. 준비된 HuatuoGPT-Vision 모델(34B 파라미터)은 다양한 의료 멀티모달 벤치마크에서 우수한 성능을 입증했습니다.

### [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/abs/2406.20094)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.20094.png)

Vote: 44

Authors: Xin Chan, Dian Yu, Dong Yu, Haitao Mi, Xiaoyang Wang

- **What's New**: 이 논문은 대규모 언어 모델(LLM)을 사용하여 다양하고 대규모의 합성 데이터를 만드는 새로운 방법론으로 '페르소나 기반 데이터 합성(persona-driven data synthesis)'을 제안합니다. 이전 연구들이 데이터 합성을 위한 프롬프트 다양화를 시도했지만, 실용적인 확장성을 달성하지 못한 반면, 이 방법론은 특정 페르소나를 추가하면 LLM이 해당 시각에서 독특한 합성 데이터를 생성하는 데 도움을 줍니다. 이를 통해 페르소나 허브(Persona Hub)를 구축하여 10억 개 이상의 다양한 페르소나를 생성하고, 광범위한 데이터 합성 시나리오에 적용할 수 있습니다.
- **Technical Details**: 페르소나 허브 구축을 위해 두 가지 접근 방식을 사용합니다: 텍스트 기반 페르소나(Text-to-Persona)와 페르소나 간 관계 확장을 통한 페르소나(Persona-to-Persona). 웹 데이터에서 무한한 텍스트를 바탕으로 특정 페르소나를 유추하고, 페르소나 간의 관계를 통해 추가 페르소나를 도출합니다. 우리는 용어 MinHash와 텍스트 임베딩을 사용하여 유사한 페르소나를 제거하여 다각도에서 페르소나의 다양성을 보장합니다. 이 과정에서 적절한 기록 제거 임계값을 설정하여 페르소나 허브의 품질을 유지합니다.
- **Performance Highlights**: 페르소나 허브의 활용 사례는 수학 및 논리 문제, 사용자 프롬프트, 지식이 풍부한 텍스트, 게임 NPC, 도구 개발 등에서 두드러집니다. 이 방법론은 진보적으로 대규모 합성 데이터의 창조를 가능하게 하며, LLM 연구 및 개발에 중대한 영향을 미칠 수 있습니다. 초기에는 20만 개의 페르소나와 여러 합성 데이터 샘플을 배포하였으며, 위험과 우려를 평가한 후 더 많은 데이터 공개를 고려하고 있습니다.

### [LLaRA: Supercharging Robot Learning Data for Vision-Language Policy](https://arxiv.org/abs/2406.20095)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.20095.png)

Vote: 13

Authors: Jinghuan Shang, Kanchana Ranasinghe, Yong Jae Lee, Yoo Sung Jang, Kumara Kahatapitiya, Mu Cai, Xiang Li, Cristina Mata, Jongwoo Park, Ryan Burgert, Michael S. Ryoo

- **What's New**: 이번 연구에서는 로봇 행동 정책을 위한 VLM(표현 언어 모델) 기반의 새로운 접근법인 Visuomotor Instruction Tuning을 제안합니다. 이 방법은 기존의 동작 클로닝(Behavior Cloning, BC) 데이터셋을 자연어 기반의 지도 데이터로 변환하여 로봇 행동 정책 학습에 활용하는 것이 핵심입니다.
- **Technical Details**: 비주얼-텍스트 모달리티를 기반으로 주어진 상태에서 적합한 동작을 텍스트로 생성하는 VLM을 훈련합니다. 이를 위해 우리는 Instruct-BC (inBC)를 만들어 VLM을 미세 조정합니다. LLaRA (Large Language and Robotics Assistant)라는 프레임워크를 통해 데이터 생성, 모델 포뮬레이션, 그리고 학습 파이프라인을 제공합니다.
- **Performance Highlights**: 제안된 방법의 효과를 입증하기 위해 다양한 로봇 과제에서 실험을 수행했습니다. 자사 연구는 로봇 동작 예측을 위한 데이터 생성의 자동화, 자가 지도 학습을 통해 보조 지시 데이터의 생성, 그리고 높은 질과 다양한 지시 데이터셋의 확장 가능성을 강화시켰습니다.

### [Direct Preference Knowledge Distillation for Large Language Models](https://arxiv.org/abs/2406.19774)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.19774.png)

Vote: 9

Authors: Yuxian Gu, Yu Cheng, Furu Wei, Dequan Wang, Yixing Li, Li Dong

- **What's New**: 지금까지의 연구들은 KL 다이버전스(KL divergence)의 한계를 극복하기 위해 다양한 방법을 모색해 왔습니다. DPKD(Direct Preference Knowledge Distillation)는 이러한 문제를 해결하기 위해 제안된 새로운 최적화 목표를 사용합니다. 이는 암묵적 보상 함수(implicit reward function)를 KL 다이버전스에 보완적으로 도입함으로써, 고효율을 유지하면서 성능을 최적화하는 방법을 제시합니다.
- **Technical Details**: DPKD는 시퀀스 레벨 KD를 최적화 문제로 재정의합니다. KL 다이버전스 대신, Bradley-Terry(BT) 모델을 활용하여 두 출력 간의 선택 확률을 평가합니다. 암묵적 보상 함수(r)를 도입해, 기존 KL 다이버전스의 한계를 보완하는 새로운 최적화 목적함수를 정의합니다. 이 최적화 목표는 교사 모델의 출력이 학생 모델의 출력보다 더 선호될 확률을 최대화하도록 설계되었습니다.
- **Performance Highlights**: 실험 결과, DPKD 방법은 기존의 기준 방법을 능가하며, 다양한 생성을 길이에 걸쳐 장점을 보여주었습니다. RougeL 지표를 사용한 평가에서 뛰어난 성능을 확인할 수 있었으며, 암묵적 보상 함수와 선호 모델링의 잠재력을 입증했습니다. GPT-2와 OPT를 비롯한 다양한 LLM(Large Language Models)을 대상으로 실험이 진행되었으며, 총 120M에서 13B 파라미터 범위를 다루는 데이터셋에서 유효성을 검증했습니다.

### [GaussianDreamerPro: Text to Manipulable 3D Gaussians with Highly Enhanced Quality](https://arxiv.org/abs/2406.18462)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.18462.png)

Vote: 7

Authors: Xiaopeng Zhang, Xinggang Wang, Wenyu Liu, Zanwei Zhou, Qi Tian, Jiemin Fang, Junjie Wang, Lingxi Xie, Guanjun Wu, Taoran Yi

- **What's New**: 이번 연구에서는 3D Gaussian Splatting (3D-GS) 기술의 장점을 최대한 활용하면서 텍스트로부터 고품질의 3D 객체를 생성하는 새로운 프레임워크인 GaussianDreamerPro를 제안합니다. 이 기술은 이전 기술들보다 더 상세하고 선명한 3D 자산을 생성할 수 있습니다.
- **Technical Details**: GaussianDreamerPro는 기본적으로 3단계의 생성 과정을 통해 3D-GS의 성능을 극대화합니다. 먼저, 3D diffusion 모델 (예: Shap-E)로부터 초벌의 3D 객체를 생성합니다. 이 객체는 그 후 2D Gaussians로 변환되며, 텍스트 조건을 반영하여 2D diffusion 모델을 사용해 최적화됩니다. 마지막 단계에서는 이러한 2D Gaussians를 기반으로 메쉬 구조를 생성하고, 이를 바탕으로 3D Gaussians를 초기화하여 최종 세부사항을 강화합니다. 결과적으로, 메쉬 구조에 바인딩된 3D Gaussians는 보다 정밀하고 선명한 3D 자산을 만드는데 이상적입니다.
- **Performance Highlights**: GaussianDreamerPro의 성능은 이전의 연구 결과와 비교했을 때 상당한 품질 향상을 보여줍니다. 특히, 생성된 3D 자산은 매우 세밀하고 명확한 표면을 가지며, 게임, 영화, XR 등의 산업에서 실용적으로 사용될 수 있는 수준의 품질을 제공합니다. 또한, 생성된 자산은 애니메이션, 결합, 시뮬레이션 등의 후속 처리를 위한 파이프라인과도 매끄럽게 통합될 수 있습니다.

### [EVF-SAM: Early Vision-Language Fusion for Text-Prompted Segment Anything Model](https://arxiv.org/abs/2406.20076)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.20076.png)

Vote: 6

Authors: Xinggang Wang, Longjin Ran, ei Liu, Wenyu Liu, Yuxuan Zhang, Tianheng Cheng, Xiaoxin Chen, Heng Liu, Rui Hu

- **What's New**: 새로운 연구는 Segment Anything Model (SAM)을 텍스트 프롬프트로 참조 표현 분할(Referring Expression Segmentation, RES)에 효과적으로 적용하는 방법을 탐구하고 있습니다. SAM의 기존 변형들에 비해 가장 중요한 것은 멀티모달 인코더(Multimodal Encoder)를 사용한 이전 비전을 언어와 융합하는 전략으로, 이로 인해 SAM은 텍스트 프롬프트에 기반한 더욱 효과적인 분할 성능을 보여줍니다. 이를 통해 기존의 대형 언어 모델(LLM) 기반 방법보다 더 경량화되고 효율적인 새로운 프레임워크인 EVF-SAM을 제안합니다.
- **Technical Details**: EVF-SAM은 멀티모달 인코더와 조기 비전-언어 융합(Early Vision-Language Fusion, EVF)을 통합하여 SAM을 텍스트 프롬프트로 작동하도록 하였습니다. BEIT-3 같은 비전-언어 모델을 사용하여 텍스트와 이미지의 멀티모달 프롬프트를 입력받아 조기에 융합시키고, 단순한 프로젝트로 SAM을 위한 프롬프트 임베딩을 생성합니다. EVF-SAM은 복잡한 모듈 없이 확장이 용이하며, RefCOCO[31]와 같은 참조 분할 데이터셋에서 간단하게 학습되었습니다.
- **Performance Highlights**: EVF-SAM은 기존의 대형 언어 모델 기반 접근 방식(LISA[20, 21, 27])에 비해 82%의 파라미터를 줄이면서도 뛰어난 참조 표현 분할 성능을 보여줍니다. 실험 결과, 텍스트와 이미지를 포함한 멀티모달 입력과 조기 융합이 SAM의 참조 표현 분할 능력을 크게 향상시키는 것으로 나타났습니다. 또한, 복잡한 템플릿이나 지시 없이도 안정적이며 효율적인 학습을 가능하게 합니다. EVF-SAM은 RefCOCO/+/g와 같은 데이터셋에서 최첨단 성능을 달성하였습니다.

### [AutoRAG-HP: Automatic Online Hyper-Parameter Tuning for Retrieval-Augmented Generation](https://arxiv.org/abs/2406.19251)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.19251.png)

Vote: 5

Authors: Lu Wang, Jue Zhang, Fangkai Yang, Qingwei Lin, Dongmei Zhang, Qi Zhang, Saravan Rajmohan, Yubo Chen, Xiaoting Qin, Jia Fu

- **What's New**: AutoML(자동화 머신러닝)의 개념을 확장하여 LLM 시대에 맞게 하이퍼파라미터 튜닝을 자동화하는 AutoRAG-HP 시스템을 도입하였습니다. 이 시스템은 온라인 설정에서 다중 무장 강도 문제(Multi-Armed Bandit, MAB)로 하이퍼파라미터 선택 문제를 정의하고 계층적 MAB(Hierarchical MAB, Hier-MAB) 방법을 제안합니다.
- **Technical Details**: AutoRAG-HP는 Retrieval-Augmented Generation(RAG) 시스템에서 하이퍼파라미터 튜닝 문제를 온라인 멀티암드 밴딧 문제로 정의합니다. Hier-MAB 방법은 상위 레벨 MAB가 각 모듈의 최적화를 안내하고, 여러 하위 레벨 MAB가 각 모듈 내에서 최적 설정을 탐색합니다. RAG 시스템의 예로는 검색 모듈과 프롬프트 압축 모듈이 있으며, 각 모듈에서 top-k 문서 조각 및 압축 비율 등의 튜닝 가능한 하이퍼파라미터가 포함됩니다.
- **Performance Highlights**: 계층적 멀티암드 밴딧 방법이 더 큰 탐색 공간에서 효율적으로 탐색할 수 있음을 입증하였으며, 다른 기본선 방법들보다 더 우수한 성능을 보였습니다. 이는 하이퍼파라미터 튜닝에 있어 보다 효율적이고 적응적인 최적화가 가능함을 보여줍니다.

### [Arboretum: A Large Multimodal Dataset Enabling AI for Biodiversity](https://arxiv.org/abs/2406.17720)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.17720.png)

Vote: 3

Authors: Kelly Marshall, Shivani Chiranjeevi, Arti Singh, Nirav Merchant, Md Zahid Hasan, Chinmay Hegde, Nirmal Baishnab, Benjamin Feuer, Zaki Jubery, Zi K. Deng, Andre Nakkab, Asheesh K Singh, Soumik Sarkar, Baskar Ganapathysubramanian, Chih-Hsuan Yang

- **What's New**: AI가 생물다양성 보존, 생태계 관리, 농업 분야에서 중요한 역할을 할 전망입니다. 기존의 AI 도구는 종의 자동식별, 생태 변화 모니터링, 작물 관리 최적화에 도움을 주었지만, 현재의 표준 AI 접근법은 여러 문제에 직면해 있습니다. 이에 대한 해결책으로, 논문에서는 1억 3천 4백 6십만 개의 사진을 포함한 Arboretum 데이터셋을 새롭게 제작하고 공개했습니다. 이 데이터셋은 바이오다이브에서 최대 규모로, 다양한 종에 대한 이미지-텍스트 데이터가 포함되어 있습니다.
- **Technical Details**: Arboretum 데이터셋은 iNaturalist 커뮤니티 과학 플랫폼에서 수집된 1억 3천 4백 6십만 개의 캡션 이미지로 구성되어 있으며, 약 32만 6천 9백 종의 생명을 포함하고 있습니다. 이 데이터셋의 메타데이터는 전문가들에 의해 검증된 텍스트 주석을 포함하고 있으며, 이미지와 텍스트 페어링 데이터를 제공합니다. 또한, 공통 이름, 학명, 세밀한 분류 계통 구조 등의 메타데이터를 통합하여 AI 모델 훈련에 필요한 정확한 데이터를 제공합니다.
- **Performance Highlights**: Arboretum 데이터셋을 활용해 훈련된 ArborCLIP 모델은 약 4천만 개의 Arboretum 샘플에서 훈련된 비전-언어 기반 모델입니다. 이 모델은 우수한 일반화 성능을 보이며, 본문 또는 과학 이름을 사용한 Zero-Shot 또는 Few-Shot 분류를 지원합니다. ArborCLIP 모델은 기존 비전-언어 모델과 비교해 매우 뛰어난 퍼포먼스를 보여주었으며, 특히 특정 생물 다양성 관련 응용 분야에서 유용하게 활용될 것으로 기대됩니다.

### [Efficient World Models with Context-Aware Tokenization](https://arxiv.org/abs/2406.19320)

![](/avatars/a660039d2152a2ab8139bc3b4a8cb439.svg)

Vote: 3

Authors: François Fleuret, Vincent Micheli, Eloi Alonso

- **What's New**: 최근 논문에서 도입한 \\Delta-iris 에이전트는 시각적으로 복잡한 환경에서도 확장 가능한 새로운 에이전트입니다. 이 에이전트는 관찰 및 행동의 진행 궤적에 주목하며 프레임을 인코딩하여 시간 단계 간의 확률적 델타를 효과적으로 설명합니다. 이를 통해 프레임을 인코딩하는 데 필요한 토큰 수를 크게 감소시킵니다.
- **Technical Details**: 이 연구에서는 부분적으로 관찰 가능한 마르코프 결정 프로세스 (POMDP)를 고려합니다. Δ-iris 에이전트는 이미지 관측값과 이산적 행동을 기반으로 다음 전이, 보상 및 에피소드 종료를 예측합니다. 에이전트는 월드 모델에서 행동을 학습하고 실제 경험은 환경 동역학을 학습하는 데만 활용됩니다.
- **Performance Highlights**: Crafter 벤치마크에서 Δ-iris 에이전트는 10M 프레임 데이터를 수집한 후 22개의 작업 중 17개를 해결하였으며, 여러 프레임 예산에서 DreamerV3를 능가합니다. 또한, Δ-iris 에이전트는 iris 대비 10배 더 빠르게 학습합니다. 실험을 통해, Δ-iris 에이전트가 월드 모델링의 결정론적 및 확률적 측면을 분리하여 학습할 수 있음을 증명했습니다.

### [RaTEScore: A Metric for Radiology Report Generation](https://arxiv.org/abs/2406.16845)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.16845.png)

Vote: 2

Authors: Weike Zhao, Ya Zhang, Weidi Xie, Chaoyi Wu, Xiaoman Zhang, Yanfeng Wang

- **What's New**: RaTEScore와 RaTE-NER 및 RaTE-Eval의 개발로 의학 인공지능 평가의 새로운 지평을 열었습니다. 특히, 방사선 보고서 평가를 위한 새로운 메트릭인 RaTEScore와 방대한 의학 명명 개체 인식(NER) 데이터셋인 RaTE-NER, 다양한 임상 텍스트를 평가하기 위한 RaTE-Eval 벤치마크를 소개합니다.
- **Technical Details**: 이 메트릭은 방사선 보고서의 질을 평가하기 위해 키 의료 개체(entities)에 우선순위를 두고, 복잡한 의학 동의어와 부정 표현에 대한 내성을 갖습니다. RaTEScore는 방사선 보고서의 문장을 세분화하고, 이를 통해 추출한 의료 개체의 타입(해부학, 질병 등)을 구별하며, 동의어 해소 모듈을 사용하여 개체 임베딩을 계산하고, 코사인 유사도로 평가하는 방식으로 작동합니다. RaTE-NER 데이터셋은 MIMIC-IV와 Radiopaedia에서 파생된 9개 모달리티와 22개 해부학적 부위의 데이터를 포함하며, RaTE-Eval 벤치마크는 문장 수준 및 문단 수준 인간 평가와 합성 보고서 비교라는 세 가지 하위 과제로 구성됩니다.
- **Performance Highlights**: 공공 데이터셋 ReXVal 및 RaTE-Eval 벤치마크에서 RaTEScore는 일관되게 다른 메트릭보다 우수한 성능을 보였습니다. 이 메트릭은 인간의 선호도와 더 잘 맞아떨어지며, 임상 텍스트의 평가에서 높은 정확성을 입증했습니다.



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
