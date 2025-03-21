# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using GPT-4o.

Thanks to [@AK391](https://github.com/AK391) for great work.

## Daily Papers (2025-03-21)

### [One-Step Residual Shifting Diffusion for Image Super-Resolution via Distillation](https://arxiv.org/abs/2503.13358)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.13358.png)

Vote: 70

Authors: Evgeny Burnaev, Daniil Selikhanovych, Nikita Gushchin, Alexander Filippov, Iaroslav Koshelev, Alexander Korotin, Sergei Kushneriuk, David Li, Aleksei Leonov

- ***What's New***: 이 논문에서는 ResShift를 최적화하여 단일 단계에서 이미지를 초해상도(Super-Resolution; SR)로 복원하는 새로운 RSD(RSD,RSD; Residual Shifting Diffusion) 디스틸레이션(Distillation) 방법을 제안합니다. 이 방법은 기존 ResShift 모델의 성능을 대폭 개선하며, 다른 최신 디스틸레이션 기반 SR 방법과 비교해 더 높은 품질의 시각적 결과를 제공합니다.
- ***Technical Details***: RSD는 ResShift를 기반으로 도입된 새로운 목표를 설정하여, 학생 네트워크가 ResShift 모델과 일치하는 이미지를 생성하도록 훈련합니다. 이를 통해 단일 단계에서 복원을 수행하며, 추가적인 지도 손실(Supervised losses)을 결합하여 더 나은 성능을 발휘합니다. RSD는 다양한 현실 및 합성 데이터셋(RealSR, RealSet65, DRealSR, ImageNet, DIV2K)에서 실험을 통해 그 효율성을 입증합니다.
- ***Performance Highlights***: RSD는 최신 텍스트-이미지 기반 SR 모델들(OSEDiff 및 SUPIR)에 비해서도 경쟁력 있는 시각 품질을 제공합니다. ResShift-1과 비교하여 더 나은 시각적 품질과 낮은 계산 비용을 요구하며, 다양한 벤치마크에서 우세한 성능을 기록했습니다.

### [Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models](https://arxiv.org/abs/2503.16419)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16419.png)

Vote: 38

Authors: Jiamu Zhang, Xia Hu, Hongyi Liu, Yu-Neng Chuang, Zhong, Yang Sui, Andrew Wen, Shaochen, Tianyi Zhang, Guanchu Wang, Jiayi Yuan, Hanjie Chen

- ***What's New***: 이 논문은 대형 언어 모델(Large Language Models; LLMs)이 '과잉 사고 현상(Overthinking Phenomenon)'으로 인해 비효율적인 추론을 하게 되는 문제를 해결하고자 합니다. 효율적인 추론(Efficient Reasoning)을 목표로 최근 LLMs에서의 추론 효율성을 탐구하는 첫 번째 체계적인 설문조사를 제공합니다.
- ***Technical Details***: 이 논문은 LLMs의 추론 효율성을 모델 기반(Model-Based), 출력 기반(Reasoning Output-Based), 입력 프롬프트 기반(Input Prompts-Based)으로 분류합니다. 모델 기반 접근법에서는 전체 추론 모델을 최적화하여 더 간결한 모델로 변환하거나 직접 효율적인 추론 모델을 훈련시키는 방법이 포함됩니다. 출력 기반 접근법은 추론 단계와 길이를 동적으로 줄이는 방법을, 입력 프롬프트 기반 접근법은 입력 프롬프트의 속성을 활용하여 추론 효율성을 향상시키는 방법을 다룹니다.
- ***Performance Highlights***: 이 논문은 다양한 효율적인 추론 기법을 제시하며, 이러한 방법들이 실제 응용 분야에서 LLMs의 실용성을 높일 수 있음을 강조합니다. 예를 들어, RL(강화 학습)과 SFT(지도 미세 조정)를 사용해 추론 길이를 최적화하는 방식은 추론 시 발생하는 불필요한 계산 비용을 줄이고 응답성을 향상시킬 수 있음을 보여줍니다.

### [Survey on Evaluation of LLM-based Agents](https://arxiv.org/abs/2503.16416)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16416.png)

Vote: 34

Authors: Roy Bar-Haim, Guy Uziel, Asaf Yehudai, Lilach Eden, Michal Shmueli-Scheuer, Alan Li, Arman Cohan, Yilun Zhao

- ***What's New***: 이 논문은 LLM 기반 에이전트를 평가하기 위한 포괄적인 방법론 설문 조사를 제공합니다. 특히, LLM 에이전트의 근본적인 에이전트 능력, 웹, 소프트웨어 엔지니어링, 과학 및 대화형 에이전트의 분야별 벤치마크, 범용 에이전트를 위한 벤치마크, 에이전트 평가를 위한 프레임워크를 체계적으로 분석합니다.
- ***Technical Details***: 논문은 LLM 기반 에이전트의 평가를 네 가지 주요 측면에서 분석합니다: (1) 계획, 도구 사용, 자기반성, 메모리를 포함한 근본적인 에이전트 능력, (2) 애플리케이션 특정 벤치마크, (3) 범용 에이전트를 위한 벤치마크, (4) 에이전트를 평가하는 프레임워크. 이 논문은 현실적이고 도전적인 평가로의 변화와 비용 효율성, 안전성 및 강건성을 평가하는 데 있어 중요한 격차를 식별합니다.
- ***Performance Highlights***: LLM 기반 에이전트 평가에서 나타난 중요한 트렌드는 현실적이고 도전적인 평가 및 지속적으로 업데이트되는 벤치마크로의 전환입니다. 현재의 평가 방식은 주로 정확성에 중점을 두고 있으나 비용과 효율성을 측정하는 메트릭의 통합이 필요합니다. 이는 특히 LLM 에이전트 개발자가 시스템을 평가, 개선 및 향상시키기 위한 필수 도구로 작용합니다.

### [Unleashing Vecset Diffusion Model for Fast Shape Generation](https://arxiv.org/abs/2503.16302)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16302.png)

Vote: 28

Authors: Xiangyu Yue, Jinwei Huang, Yuhong Liu, Xianghui Yang, Zibo Zhao, Chunchao Guo, Zeqiang Lai, Qinxiang Lin, Haolin Liu, Yunfei Zhao, Jie Jiang, Fuyun Wang, Huiwen Shi

- ***What's New***: FlashVDM은 Vecset Diffusion Model(VDM)의 VAE 및 DiT 가속화를 통해 고속 3D 형상 생성에 대한 획기적인 발전을 이루었습니다. 특히, Adaptive KV Selection, Hierarchical Volume Decoding, 그리고 Efficient Network Design을 통해 VAE 디코딩에서 FLOP를 크게 줄이고, 코드 생성 및 시각적 추론을 크게 가속화했습니다.
- ***Technical Details***: FlashVDM은 Progressive Flow Distillation이라는 새로운 다단계 증류 방법을 통해 Diffusion Sampling을 최소 5 단계의 추론으로 가속화합니다. VAE 가속화를 위해서는 Adaptive KV Selection, Hierarchical Volume Decoding, Efficient Decoder Architecture가 도입되었습니다. 이들은 효과적으로 VDM의 로컬 성과 형태 표면의 희소성을 이용하여 성능을 최적화합니다.
- ***Performance Highlights***: 최첨단 모델들과 비교했을 때, FlashVDM은 기존의 고속 3D 생성 방법보다 최대 45배 빠른 VAE 디코딩과 32배 빠른 전체 생성 성능을 달성하였습니다. 모델의 성능은 여전히 동급의 품질을 유지하면서 3D 형상을 1초 미만의 시간에 생성할 수 있게 되었습니다.

### [Scale-wise Distillation of Diffusion Models](https://arxiv.org/abs/2503.16397)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16397.png)

Vote: 25

Authors: Artem Babenko, Dmitry Baranchuk, Nikita Starodubcev, Denis Kuznedelev

- ***What's New***: 이 논문에서는 새로운 스케일 방식(SWD; Scale-wise Distillation) 확산 모델(Diffusion Models)을 제안하였습니다. 이는 텍스트-이미지 확산 모델(Text-to-Image Diffusion Models)에서 샘플링 단계 동안 공간적 해상도를 점진적으로 증가시켜, 효율성을 높이고 계산 비용을 줄이는 혁신적인 접근법입니다.
- ***Technical Details***: SWD는 현재 스펙트럼 분석을 활용하여 노이즈가 높은 수준에서 고주파를 억제하는 특징을 이용하여 구성됩니다. 그리고 패치 분포 매칭 손실(Patch Distribution Matching Loss; PDM)을 도입하여 목표 분포와의 더 정밀한 유사성을 확보하고 있습니다.
- ***Performance Highlights***: 실험 결과, SWD는 두 개의 전체 해상도 단계에 소요 시간을 접근하면서도 동급의 다른 모델들보다 뛰어난 성능을 보여주었으며, 특히 2.5배에서 10배 더 빠른 추론 속도를 경험하였습니다.

### [DiffMoE: Dynamic Token Selection for Scalable Diffusion Transformers](https://arxiv.org/abs/2503.14487)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.14487.png)

Vote: 22

Authors: Haotian Yang, Di Zhang, Wenzhao Zheng, Xintao Wang, Jiwen Lu, Minglei Shi, Jie Zhou, Wenliang Zhao, Kun Gai, Ziyang Yuan, Xin Tao, Pengfei Wan, Mingwu Zheng

- ***What's New***: DiffMoE는 시각적 생성 작업에서의 확산 변환기(Diffusion Transformers)의 한계를 극복하기 위한 새로운 혼합 전문가(Mixture-of-Experts; MoE) 기반 아키텍처입니다. DiffMoE는 교육 중 배치 수준의 글로벌 토큰 풀을 활용해 다양한 노이즈 수준과 샘플 복잡성에 따라 동적으로 계산 자원을 할당하여 확산 모델에서의 성능을 향상시킵니다.
- ***Technical Details***: DiffMoE는 글로벌 토큰 분포 접근성을 향상시키기 위해 배치 수준의 글로벌 토큰 풀을 도입하였습니다. 이 풀은 서로 다른 노이즈 수준과 샘플들을 아우르며 전문가들이 포괄적인 글로벌 토큰 정보를 학습할 수 있게 합니다. 또한, 적응형 용량 예측기를 통해 학습 시 토큰 라우팅 패턴을 분석하여 계산 자원을 효율적으로 배분합니다. 이를 통해 복잡한 경우에는 더 많은 자원을, 간단한 경우에는 적은 자원을 사용하여 확산 모델을 최적화합니다.
- ***Performance Highlights***: ImageNet 벤치마크에서 DiffMoE는 기존의 밀집 모델(dense)과 기존 MoE 접근보다 뛰어난 성능을 보였습니다. DiffMoE는 3배의 활성화된 파라미터를 가진 모델보다 우수한 성능을 유지하면서도 1배의 활성화된 파라미터로 실행됩니다. 또한, 텍스트-이미지 생성 같은 더 어려운 작업에서도 우수한 범용성을 나타냅니다.

### [JARVIS-VLA: Post-Training Large-Scale Vision Language Models to Play Visual Games with Keyboards and Mouse](https://arxiv.org/abs/2503.16365)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16365.png)

Vote: 21

Authors: Xiaojian Ma, Kaichen He, Zihao Wang, Muyao Li, Yitao Liang

- ***What's New***: JARVIS-VLA는 대규모 시각-언어 모델(Vision Language Models; VLMs)을 비디오 게임 환경에서 키보드와 마우스를 활용하여 조작하는 최초의 모델입니다. 새로운 훈련 패러다임인 ActVLP(Visual Language Post-Training)를 도입하여 시각적-언어적 작업을 통한 모델 성능 향상을 목표로 하고 있습니다.
- ***Technical Details***: JARVIS-VLA는 Llava와 유사한 구조를 채택했으며, 기본적으로 Vision Transformer를 활용하여 이미지를 처리합니다. 모델은 시각적 랜드마크 및 공간 정렬 데이터셋을 사용하여 비주얼-언어 정렬을 개선하며, 마인크래프트에서 다양한 원자적 작업을 수행할 수 있습니다. 또한, 행동 디코더를 통합하여 일반 언어 모델로부터 행동을 생성하는 새로운 방식을 채택했습니다.
- ***Performance Highlights***: JARVIS-VLA는 기존의 VPT 및 다른 최첨단 무대에서 두 배 이상의 성공률을 기록하며, '아이템 제작' 및 '재료 제련'과 같은 복잡한 작업에서도 뛰어난 성능을 보였습니다. GPU 32대에서 훈련되었으며, 성공적인 행동 생성과 높은 정확도의 의사결정 능력을 보여줍니다.

### [Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning](https://arxiv.org/abs/2503.15558)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.15558.png)

Vote: 21

Authors: Lindsey Pavao, Francesco Ferroni, Tsung-Yi Lin, Yao Xu, Elena Lantz, Yun Ni, Zhaoshuo Li, Nayeon Lee, Andrew Mathau, Andrew Z. Wang, Yen-Chen Lin, Xiaodong Yang, Hannah Brandon, Xiaohui Zeng, Brendan Johnson, Jenna Diamond, Huayu Chen, Zhe Zhang, NVIDIA, Xuan Li, Fangyin Wei, Yin Cui, Boxin Wang, George Kurian, Alisson Azzolini, Haoxiang Wang, Jinju Chu, Jinwei Gu, Jingyi Jin, Rama Govindaraju, Jiashu Xu, Lyne Tchapmi, Zhuolin Yang, Imad El Hanafi, Misha Smelyanskiy, Jacob Huffman, Ming-Yu Liu, Siddharth Gururani, Rizwan Khan, Wei Ping, David W. Romero, Zekun Hao, Yifan Ding, Prithvijit Chattopadhyay, Shuran Song

- ***What's New***: Cosmos-Reason1은 물리적 세상에서 물리적 AI 시스템의 이해와 결정 능력을 향상시키기 위한 새로운 멀티모달 대형언어모델(multi-modal large language models)입니다. 이는 공간, 시간, 그리고 물리학에 대한 근본적인 지식을 포착하는 물리적 상식(physical common sense)과 다양한 물리적 구현을 일반화하는 구현 추론(embodied reasoning)을 포함하여 물리적 세계를 이해합니다.
- ***Technical Details***: Cosmos-Reason1 모델들은 Vision Pre-Training, General Supervised Fine-Tuning (SFT), Physical AI SFT, 그리고 Physical AI Reinforcement Learning (RL)의 네 가지 단계로 데이터 기반 학습을 거칩니다. 특히 물리적 상식을 위한 계층적 온톨로지(hierarchical ontology)와 구현 추론을 위한 2차원 온톨로지를 정의하고 있습니다. 또한, 두 가지 모델 사이즈, Cosmos-Reason1-8B와 Cosmos-Reason1-56B가 존재합니다.
- ***Performance Highlights***: Cosmos-Reason1은 새로 개발된 물리적 상식과 구현 추론 벤치마크에서 강력한 성능을 보였습니다. 특이하게도 Physical AI SFT를 통해 기본 VLM 모델보다 10% 이상 성능이 개선되었으며, Reinforcement Learning을 통해서는 8% 이상의 추가적인 성능 향상이 이루어졌습니다. 특히, 직관적 물리학(intuitive physics), 시간의 화살표(arrow of time), 및 물체 영속성(object permanence)과 같은 기존 모델이 어려워하는 영역에서도 더욱 강력한 성능을 발휘합니다.

### [MathFusion: Enhancing Mathematic Problem-solving of LLM through Instruction Fusion](https://arxiv.org/abs/2503.16212)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16212.png)

Vote: 17

Authors: Yu Li, Honglin Lin, Qizhi Pei, Zhuoshi Pan, Xin Gao, Conghui He, Chenlin Ming, Rui Yan, Lijun Wu

- ***What's New***: MathFusion은 대형 언어 모델(LLMs)의 수학적 문제 해결 능력을 향상시키기 위해 도입된 혁신적인 프레임워크로, 여러 수학적 문제들을 융합하여 문제 해결 효율성을 높입니다. 이는 인간 학습 프로세스를 모방하여 서로 다른 수학적 지침을 합성함으로써 문제 해결 능력을 발전시킵니다.
- ***Technical Details***: MathFusion은 세 가지 융합 전략을 채택합니다: (1) Sequential Fusion은 관련 문제를 연결하여 해결 의존성을 모델링합니다, (2) Parallel Fusion은 유사한 문제를 결합하여 개념적 이해를 강화합니다, (3) Conditional Fusion은 맥락 인식 기반 선택 문제를 생성하여 추론 유연성을 향상시킵니다. MathFusion은 각 전략에 따라 새로운 데이터를 생성하고 이를 기반으로 모델을 미세 조정합니다.
- ***Performance Highlights***: MathFusion을 적용한 결과, 수학적 추론 정확도가 다양한 벤치마크에서 평균 18.0점 향상되었으며, 추가된 45,000개의 합성 지침만으로 이룬 성과입니다. 이는 전통적인 단일 지침 접근법에 비해 약 1.4점의 정확도 향상을 보이며, 오랫동안 사용되어 온 DART-Math 기술과 비교할 때 데이터 활용 효율성 면에서도 우수합니다.

### [Plug-and-Play 1.x-Bit KV Cache Quantization for Video Large Language Models](https://arxiv.org/abs/2503.16257)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16257.png)

Vote: 17

Authors: Haoxuan You, Huan Wang, Keda Tao, Yang Sui, Can Qin

- ***What's New***: VidKV는 비디오 대형 언어 모델(Video Large Language Models; VideoLLMs)에서 기존 연구보다 더 낮은 비트로의 KV 캐시(KV Cache) 양자화를 탐구하며, 1.x-비트의 플러그 앤 플레이(Plug-and-Play)형 KV 캐시 양자화 방법을 제안합니다. 이 연구는 특히 1.5비트와 1.58비트 정밀도로 KV 캐시를 효과적으로 압축하는 방법을 최초로 소개합니다.
- ***Technical Details***: VidKV에서는 키(Key)에 대해 채널 차원에서 혼합 정밀도 양자화 전략을 도입하여, 이상 채널에는 2-비트 양자화를, 정상 채널에는 FFT와 결합된 1-비트 양자화를 수행합니다. 값(Value) 부분에는 1.58-비트 양자화를 적용하며, 의미적으로 중요한 비주얼 토큰은 선택적으로 보호합니다. 이런 방식으로 정밀도와 모델 성능 사이의 최적의 균형을 찾아냅니다. 이 방법은, 이전의 LLM 연구에서 보였던 토큰별 양자화 방식을 넘어 채널별 양자화가 더 적절하다는 점을 제시합니다.
- ***Performance Highlights***: LLaVA-OV-7B 및 Qwen2.5-VL-7B 모델을 사용한 실험에서, VidKV는 FP16과 거의 차이가 없는 성능 저하로 1.5-bit와 1.58-bit 정밀도로 KV 캐시를 효과적으로 압축할 수 있음을 보여줬습니다. 중대한 성능 하락 없이 모델의 메모리 사용을 크게 줄이면서, 다양한 벤치마크에서 탁월한 결과를 기록했습니다.

### [InfiniteYou: Flexible Photo Recrafting While Preserving Your Identity](https://arxiv.org/abs/2503.16418)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16418.png)

Vote: 16

Authors: Xin Lu, Liming Jiang, Hao Kang, Zichuan Liu, Qing Yan, Yumin Jia

- ***What's New***: 이 논문에서는 최신 디퓨전 트랜스포머(Diffusion Transformers; DiTs)를 활용하여, 높은 품질의 개인 식별 정보를 보존한 이미지 생성 프레임워크 'InfiniteYou (InfU)'를 제안합니다. InfU는 기존 방법들이 가지고 있는 식별 정보의 부족, 텍스트-이미지 정렬의 낮은 수준, 생성된 이미지의 품질 및 미적 감각 부족 등의 문제를 개선하고, 플러그 앤 플레이(Plug-and-Play) 디자인으로 여러 방법과의 호환성을 제공합니다.
- ***Technical Details***: InfU의 핵심은 InfuseNet으로, 이는 ControlNet의 일반화로서 DiT 기본 모델에 정체성 특징을 잔여 연결(residual connections)을 통해 주입하여 식별 유사성을 강화합니다. 또한, 다단계 학습 전략으로 사전 학습 및 감독 미세 조정(Supervised Fine-Tuning; SFT)을 활용하며, 합성 단일 인물-다중 샘플(SPMS) 데이터를 사용하여 텍스트-이미지 정렬과 이미지 품질 및 미적 감각을 개선합니다.
- ***Performance Highlights***: InfU는 기존의 최첨단 방법들과 비교하여 정체성 유사성, 텍스트-이미지 정렬, 그리고 이미지 품질 측면에서 우수한 성능을 보여줍니다. 'Plug-and-Play' 설계로 다양한 방법과의 호환성을 보장하여, 다양한 시나리오에서 유용한 가치가 있습니다.

### [Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.13657.png)

Vote: 14

Authors: Kannan Ramchandran, Shuyi Yang, Ion Stoica, Lakshya A. Agrawal, Rishabh Tiwari, Joseph E. Gonzalez, Bhavya Chopra, Dan Klein, Aditya Parameswaran, Matei Zaharia, Kurt Keutzer, Melissa Z. Pan, Mert Cemri

- ***What's New***: 이 논문은 Multi-Agent Systems (MAS)의 성능 개선이 기존 단일 에이전트 프레임워크와 비교하여 미미하며, 이 시스템의 효과를 저해하는 문제들을 분석하는 최초의 포괄적인 연구를 제시합니다. 이 연구는 150개 이상의 작업에서 5개의 인기 있는 MAS 프레임워크를 분석하며, 14개의 독특한 실패 모드를 식별하고 이를 다양한 MAS 프레임워크에 적용 가능한 포괄적인 분류법으로 제안합니다.
- ***Technical Details***: 이 논문에서는 각각 전문 인간 주석자들이 참여한 6명의 주석자와 함께 150개 이상의 대화 추적을 사용하여 MAS의 결함을 체계적으로 분석합니다. 식별된 실패 모드 중에는 명세 및 시스템 설계 실패, 에이전트 간 불일치, 작업 검증 및 종료와 관련된 요소가 있습니다. 평가의 확장성을 위해 LLM-as-a-Judge 파이프라인을 MASFT와 통합하고, 제안된 개입으로 에이전트 역할의 강화와 통합 전략을 제시합니다.
- ***Performance Highlights***: ChatDev와 같은 개방형 소스 MAS의 정확도가 최대 25% 이하일 수 있음을 실험적으로 보여주며, 개선된 명세와 프롬프트 전략이 일부 성공률의 향상을 가져오지만 모든 실패 사례를 해결하지는 못했습니다. 제안된 개입이 ChatDev에서 14%의 성능 향상을 나타냈지만, 실제 세계의 배포에는 여전히 충분하지 않음을 시사합니다.

### [Inside-Out: Hidden Factual Knowledge in LLMs](https://arxiv.org/abs/2503.15299)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.15299.png)

Vote: 13

Authors: Roi Reichart, Eran Ofek, Hadas Orgad, Yonatan Belinkov, Zorik Gekhman, Eyal Ben David, Jonathan Herzig, Idan Szpector

- ***What's New***: Inside-Out 논문은 대형 언어 모델(LLMs)이 출력보다 더 많은 사실적 지식을 내부에 인코딩할 수 있다는 것을 평가하는 새로운 프레임워크를 제시합니다. 이는 LLMs의 내부 및 외부 지식의 격차를 체계적으로 연구할 수 있는 최초의 구체적인 정의를 제공합니다.
- ***Technical Details***: 이 연구는 파라미터 내에 인코딩된 지식과 출력된 지식의 격차를 '숨은 지식(hidden knowledge)'으로 정의합니다. 이를 위해 내부적으로 모델의 중간 계산을 활용하여 정답 후보군을 평가하는 방법과 외부적으로 토큰 기반의 확률을 활용하는 방법을 비교합니다. 또한 1,700개의 질문에 대해 LLM이 생성한 1,000개의 답변을 통해 정답성을 평가합니다.
- ***Performance Highlights***: 세 가지 오픈웨이트 LLM을 대상으로 한 실험 결과, 내부 스코어링 방법이 외부 스코어링 방법보다 평균 40% 더 많은 지식을 보여주었습니다. 그러나 답을 생성할 수 없는 경우도 존재하며, 이는 LLMs의 생성 능력에 대한 근본적인 한계를 나타냅니다. 특히, 일부 질문에서 모델은 1,000번의 시도에도 불구하고 올바른 답을 생성하지 못했습니다.

### [CaKE: Circuit-aware Editing Enables Generalizable Knowledge Learners](https://arxiv.org/abs/2503.16356)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16356.png)

Vote: 11

Authors: Jizhan Fang, Yunzhi Yao, Shumin Deng, Jia-Chen Gu, Huajun Chen, Ningyu Zhang, Nanyun Peng

- ***What's New***: CaKE는 대형 언어 모델(LLM)이 수정된 지식을 다중 단계 추론 과제에 효과적으로 통합할 수 있도록 설계된 새로운 지식 편집 방법입니다. CaKE는 모델이 수정된 지식을 사용할 수 있는 추론 회로(Reasoning Circuit)를 개발하도록 유도하여 이전 방법에 비해 20% 향상된 MQuAKE 데이터셋에서의 다중 단계 추론 정확성을 달성합니다.
- ***Technical Details***: CaKE는 신중하게 큐레이트된 데이터를 활용하여 LLMs가 수정된 지식을 사용하는 전략을 강화합니다. 이는 새로운 지식을 사용하도록 모델을 훈련시키는 회로 기반 분석을 통해 수행됩니다. 또한, 최적화된 수정 과제는 무작위 선택된 관련 엔티티를 사용하여 생성되어 데이터 유출을 방지합니다.
- ***Performance Highlights***: CaKE는 기존의 모든 지식 편집 방법을 능가하며, 특히 MQuAKE 다중 홉 편집 벤치마크에서 LLaMA3-8B-Instruct 및 Qwen2.5-7B-Instruct 모델을 사용한 실험에서 57.3% 이상의 다중 단계 추론 정확도를 기록했습니다. 이는 다른 방법들과 비교하여 상당한 이점입니다.

### [Expert Race: A Flexible Routing Strategy for Scaling Diffusion Transformer with Mixture of Experts](https://arxiv.org/abs/2503.16057)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16057.png)

Vote: 10

Authors: Defa Zhu, Yike Yuan, Ziyu Wang, Jingyi Yu, Qiyang Min, Xun Zhou, Zihao Huang

- ***What's New***: Race-DiT는 새로운 Mixture of Experts (MoE) 모델로, Expert Race라는 유연한 라우팅 전략을 도입하여 확장성을 증진했습니다. 이 모델은 시각 생성 분야에서의 확산 변환기(Diffusion Transformer)의 규모와 성능을 크게 향상시킬 수 있는 가능성을 보여줍니다.
- ***Technical Details***: Expert Race는 MoE(Mixture of Experts) 구조 내에서 토큰과 전문가가 경쟁하여 중요한 토큰에 전문가를 동적으로 할당하는 방식을 학습합니다. 얕은 층에서의 학습 문제를 해결하기 위해 레이어별 규제(Per-Layer Regularization)를 제안하며, 라우터 유사성 손실(Router Similarity Loss)로 모드 붕괴를 방지하고 전문가의 활용을 최적화합니다.
- ***Performance Highlights***: ImageNet에서의 실험 결과, Race-DiT는 기존 방법들 대비 상당한 성능 향상을 보여주었습니다. 특히, FID 측정치에서 기존보다 낮은 수치를 기록하며 이미지 생성의 품질을 크게 향상시켰습니다. MoE 확장 분석을 통해 다양한 확산 작업에 대한 적용 가능성을 확인하였습니다.

### [Ultra-Resolution Adaptation with Ease](https://arxiv.org/abs/2503.16322)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16322.png)

Vote: 10

Authors: Zhenxiong Tan, Xinchao Wang, Songhua Liu, Ruonan Yu

- ***What's New***: 이 논문에서는 울트라 해상도 영상 생성 문제를 다루며, 한정된 데이터와 자원으로 효과적으로 고해상도 이미지를 생성할 수 있는 URAE라는 가이드라인을 제안합니다. 이 방법은 데이터 효율성 및 파라미터 효율성을 통해 기존 텍스트-이미지 변환 모델의 해상도 적응을 쉽게 만듭니다.
- ***Technical Details***: URAE는 두 가지 주요 관점에서 시작합니다. 첫째, 몇몇 교사 모델(teacher models)을 통한 합성 데이터는 훈련 수렴을 촉진하는 데 실질적 도움을 줄 수 있습니다. 둘째, 합성 데이터를 사용할 수 없을 때, 사전 학습된 가중치 행렬의 소수 구성 요소 조정을 통해 성능을 극대화할 수 있으며, 이는 LoRA와 같은 저랭크 어댑터를 사용할 때보다 우수한 결과를 제공합니다. 또한, FLUX와 같은 모델의 가이드라인 증류(fine-tuning guidance-distilled models)에 대해서는 분류기-프리 가이드라인을 비활성화하는 것이 중요하다는 사실을 발견했습니다.
- ***Performance Highlights***: URAE는 2K 해상도 생성 성능에서 FLUX1.1 [Pro] Ultra와 같은 고급 폐쇄형 모델에 비교 가능한 결과를 3K 샘플과 2K 반복으로 달성하며, 4K 해상도 생성에서는 새로운 벤치마크를 달성했습니다. 이로써 현재의 SOTA 모델 대비 적은 데이터와 파라미터로도 고성능의 초고해상도 이미지를 생성할 수 있음을 시연했습니다.

### [M3: 3D-Spatial MultiModal Memory](https://arxiv.org/abs/2503.16413)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16413.png)

Vote: 10

Authors: Xiaolong Wang, Xuanbin Peng, Sifei Liu, Ri-Zhao Qiu, Yuchen Song, Jianglong Ye, Xueyan Zou

- ***What's New***: 본 논문에서는 M3로 명명된 3D 스페이셜 멀티모달 메모리(3D Spatial MultiModal Memory)를 제안합니다. 이는 비디오 소스를 통해 시각적 인식을 위한 정적인 중규모 장면 정보를 저장할 수 있는 멀티모달 메모리 시스템으로, Gaussian Splattig 기법과 기초 모델들(Foundation Models)을 통합하여 멀티모달 메모리를 구축합니다. 이 시스템은 3D 특징 증류(Feature Distillation)에서의 핵심 압축 문제를 최초로 다루고 있습니다.
- ***Technical Details***: M3는 Gaussian Splattig 기법과 멀티모달 기초 모델들을 통합하여 3D 구조에서의 다층적 정보 표현을 가능하게 합니다. 구현의 핵심 요소로 'Principal Scene Components'와 'Gaussian Memory Attention'을 제안하여 효율적인 훈련과 추론을 지원합니다. 이를 통해, 높은 차원의 2D 특징 맵을 메모리 은행에 저장하고, 3D Gaussian에서 저차원 Principal Queries를 인덱스로 사용하여 공간 질의를 용이하게 합니다.
- ***Performance Highlights***: M3 시스템은 각종 벤치마크 실험에서 이전 연구들을 뛰어넘는 특징 기억 및 다운스트림 태스크 성능을 보였습니다. Cosine 및 L2 distance와 같은 저수준 지표에서 우수한 비교 결과를 보였으며, mIoU 및 AP 같은 고수준 지표에서도 향상된 성능을 확인할 수 있었습니다. 또한 낮은 계산 비용을 유지하며, 질의 및 하이-레벨 태스크에서 강력한 표현력을 발휘합니다.

### [SynCity: Training-Free Generation of 3D Worlds](https://arxiv.org/abs/2503.16420)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16420.png)

Vote: 9

Authors: Iro Laina, Andrea Vedaldi, Aleksandar Shtedritski, Christian Rupprecht, Paul Engstler

- ***What's New***: SynCity는 3D 세계를 생성하기 위한 최초의 훈련 필요 없는(training-free) 접근 방식을 제안합니다. 이 방법은 사전 학습된 3D 생성 모델의 기하학적 정밀성과 2D 이미지 생성기의 예술적 다양성을 활용하여 큰 규모의 고품질 3D 공간을 생성합니다.
- ***Technical Details***: SynCity는 타일 기반 접근 방식을 통해 텍스트 설명으로부터 3D 세계를 생성합니다. 이는 각 타일을 세밀하게 제어할 수 있는 가능성을 허용하며, 각 타일은 자신만의 세계 맥락에서 생성된 후 장면과 결합됩니다. 이 시스템은 이미지를 생성하기 위한 텍스트 투 이미지 생성기(예: Flux)와 이미지 투 3D 생성기(TRELLIS)를 사용하며, 타일 간의 기하학적 일관성을 보장하기 위해 3D 표현을 결합하는 메커니즘을 제안합니다.
- ***Performance Highlights***: SynCity는 인간 선호 조사에서 BlockFusion보다 더 나은 결과를 보여줍니다. 평가 항목으로는 전체적 선호도, 기하학, 탐험의 재미, 다양성, 사실감이 있으며, SynCity는 모든 항목에서 BlockFusion을 능가했습니다.

### [1000+ FPS 4D Gaussian Splatting for Dynamic Scene Rendering](https://arxiv.org/abs/2503.16422)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16422.png)

Vote: 9

Authors: Xingyi Yang, Yuheng Yuan, Xinchao Wang, Qiuhong Shen

- ***What's New***: 4DGS-1K은 현대적인 GPU에서 1000 FPS 이상의 렌더링 속도를 달성할 수 있는 동적 장면 렌더링을 위한 고속 및 고압축 4D Gaussian Splatting 기법입니다. 이 모델은 기존 4D Gaussian Splatting (4DGS) 기법의 시간적 중복성을 해결함으로써 보다 효율적인 메모리 사용을 가능하게 합니다.
- ***Technical Details***: 4DGS-1K은 공간-시간 변동 점수(Spatial-Temporal Variation Score)를 사용하는 새로운 가지치기(pruning) 기준을 도입하여 짧은 수명의 Gaussian을 제거하고, 시간 필터를 사용하여 각 프레임마다 필요한 활성 Gaussian만을 렌더링하는 전략을 채택합니다. 이를 통해 불필요한 계산을 줄이고, 저장 요구 사항을 크게 감소시킵니다.
- ***Performance Highlights***: 4DGS-1K은 Neural 3D Video Dataset 및 D-NeRF Dataset에서 원본 4DGS에 비해 최대 41배의 저장 효율성과 9배의 빠른 속도를 기록하면서 동등한 시각 품질을 유지합니다. 특히, 실시간 렌더링 속도가 1000 FPS를 초과하여 고품질 동적 장면 모델링에 실용적인 솔루션을 제공합니다.

### [Tokenize Image as a Set](https://arxiv.org/abs/2503.16425)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16425.png)

Vote: 8

Authors: Shuyang Gu, Han Hu, Zigang Geng, Mengde Xu

- ***What's New***: 이 논문에서는 이미지 생성에 대해 세트를 기반으로 한 전혀 새로운 방법론을 제안합니다. 기존의 고정된 위치의 잠재 코드를 사용하는 방법과 달리, 지역적 의미 복잡성에 따라 코딩 용량을 동적으로 배정하는 무순서 토큰 집합(Token Set) 표현을 도입합니다. 이것은 전역적 문맥 집계를 강화하고, 지역적 변동에 대한 견고성을 개선합니다.
- ***Technical Details***: 중요한 과제를 해결하기 위해, 세트를 고정 길이의 정수 시퀀스로 변환할 수 있는 이중 변환 메커니즘(Dual Transformation)을 개발했습니다. 이로써 세트 모델링 문제를 시퀀스 모델링 문제로 변화시킵니다. 제안된 고정 합 이산 확산 모델(Fixed-Sum Discrete Diffusion)은 이산 값, 고정 시퀀스 길이, 합 불변성을 동시에 처리할 수 있는 최초의 프레임워크로, 효과적인 세트 분포 모델링을 가능하게 합니다.
- ***Performance Highlights***: 실험 결과, 제안된 방법이 의미 인식 표현 및 생성 품질에서 뛰어난 성능을 발휘하는 것을 보여줍니다. 이는 전통적인 순차 토큰 패러다임을 넘어선 시각적 생성의 진전을 나타냅니다.

### [MagicMotion: Controllable Video Generation with Dense-to-Sparse Trajectory Guidance](https://arxiv.org/abs/2503.16421)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16421.png)

Vote: 8

Authors: Hui Zhang, Quanhao Li, Zhen Xing, Zuxuan Wu, Rui Wang, Qi Dai

- ***What's New***: MagicMotion은 이미지에서 비디오로 변환하는 새로운 프레임워크로, 세 가지 조건 수준(마스크, 박스, 희박 박스)을 통해 경로 제어를 지원합니다. 이 모델은 다양한 경로 입력(trajectory input)에 대해 고품질 비디오를 생성하며, 개체 일관성과 시각적 품질을 유지합니다.
- ***Technical Details***: MagicMotion은 Trajectory ControlNet을 사용하여 ControlNet과 유사한 아키텍처로 경로 정보를 주입합니다. 모델은 단계별 훈련 전략을 활용하며, 각 단계는 이전 단계에서 학습한 가중치를 초기화하여 더 나은 성능을 달성합니다. 추가적으로, 모델은 희소한 경로 조건 하에 미세한 객체 형태를 더 잘 인식하도록 Latent Segment Loss를 도입했습니다.
- ***Performance Highlights***: 다양한 메트릭에서의 실험 결과, MagicMotion은 기존 방법들을 뛰어넘는 성능을 보여주었습니다. MagicBench 및 DAVIS 데이터셋에서 MagicMotion은 더 높은 비디오 품질과 더 정확한 경로 제어를 달성하였습니다. 특히 많거나 적은 수의 물체를 제어할 때에도 일관된 성능을 유지합니다.

### [Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning](https://arxiv.org/abs/2503.16252)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16252.png)

Vote: 7

Authors: Fangqi Lou, Xueqian Zhao, Zhaowei Liu, Jinyi Niu, Chao Li, Sheng Xu, Yun Chen, Liwen Zhang, Zixuan Wang, Zuo Bai, Ziwei Yang, Jiajie Xu, Weige Cai, Lingfeng Zeng, Dezhi Chen, Xin Guo

- ***What's New***: Fin-R1은 금융 추론을 위한 대규모 언어 모델로, 강화 학습(Reinforcement Learning)을 통해 금융 문제 해결 능력을 크게 개선한 최초의 시도입니다. 7천만 개의 파라미터로 경량화를 실현하여 배포 비용을 줄이는 동시에, 파편화된 금융 데이터, 비추적성 추론 논리, 약한 비즈니스 일반화라는 주요 금융 문제를 효과적으로 해결합니다.
- ***Technical Details***: Fin-R1은 두 단계의 프레임워크를 사용하여 설계되었습니다. 첫 단계에서는 Fin-R1-Data라는 고품질의 데이터셋을 구축하였는데, 이는 여러 권위 있는 데이터셋에서 추출된 60,091개의 사고의 흐름(Chain of Thought; CoT)을 포함합니다. 후속 단계에서는 이 데이터셋을 기반으로 강화 학습과 감독 학습(Supervised Fine-Tuning; SFT)을 수행하여 모델의 금융 추론 능력을 강화했습니다.
- ***Performance Highlights***: Fin-R1은 다양한 금융 비즈니스 시나리오에 걸친 벤치마크 평가에서 평균 점수 75.2를 기록하여 전체 2위에 올랐습니다. 특히 금융 추론에 중점을 둔 ConvFinQA와 FinQA에서 각각 85.0, 76.0을 기록하여 최고 성과를 거뒀습니다. DeepSeek-R1-Distill-Llama-70B보다 8.7점 높은 성과를 보여, 경량 구조임에도 불구하고 우수한 성능을 자랑합니다.

### [BigO(Bench) -- Can LLMs Generate Code with Controlled Time and Space Complexity?](https://arxiv.org/abs/2503.15242)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.15242.png)

Vote: 7

Authors: Pierre Chambon, Benoit Sagot, Baptiste Roziere, Gabriel Synnaeve

- ***What's New***: BigO(Bench)는 생성적 언어 모델(LLMs)의 시간 및 공간 복잡성을 제어하면서 코드를 이해하고 생성하는 능력을 평가하는 새로운 코드 벤치마크입니다. 이 벤치마크는 Python 함수의 알고리즘 복잡성을 추론하는 도구와 코드 경연 대회에서 유래하여 복잡성 레이블을 가진 3,105개의 코딩 문제와 1,190,250개의 솔루션을 포함합니다.
- ***Technical Details***: BigO(Bench)는 다른 복잡성 제한이 주어진 상태에서 LLMs가 문제를 해결하도록 요구합니다. 이 프레임워크는 Python 코드를 분석하여 시간 및 공간 복잡성을 추론하는 규칙 기반 알고리즘을 사용합니다. 실행 메트릭의 영향을 평가하기 위해 입력 크기를 증가시키며, 이를 통해 함수의 전체 복잡성을 파악합니다.
- ***Performance Highlights***: 성능 평가 결과, DeepSeek-R1 Llama 70B 모델은 시간 복잡성 예측에서 29.2%, 생성에서 4.8%의 `Pass@1` 점수를 기록했으며, 이는 현재 모델들이 복잡성 이해 및 코드 생성에서 상당한 도전 과제를 안고 있음을 보여줍니다. 다른 모델들과 비교했을 때 DeepSeek 모델들이 프로그래밍 과제에서 비교적 높은 성능을 보입니다.

### [Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't](https://arxiv.org/abs/2503.16219)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16219.png)

Vote: 7

Authors: Chris Ngo, Quy-Anh Dang

- ***What's New***: 이 연구는 제한된 컴퓨팅 자원과 데이터로 작은 LLM(1.5-billion-parameter model)의 추론 능력을 강화하기 위한 강화 학습(Reinforcement Learning; RL)의 잠재력을 조사합니다. 엄격한 자원 제한 조건 하에서도 RL 기반 미세 조정이 작은 LLM의 추론 성능을 크게 향상시킬 수 있음을 보여주며, 이를 통해 대규모 접근 방식에 대한 비용 효율적인 대안을 제공합니다.
- ***Technical Details***: 이 연구에서는 Group Relative Policy Optimization (GRPO) 알고리즘을 활용하여 작은 LLM의 학습 프레임워크를 최적화했습니다. 또한, 고품질 수학 추론 데이터셋을 제한된 리소스 환경에 맞게 제작하여 훈련에 소요되는 비용을 최소화하고 성능을 극대화했습니다. 특히, NVIDIA A40 GPU 4대로 24시간 이내에 학습을 완료할 수 있도록 모델을 설계했습니다.
- ***Performance Highlights***: AMC23 정확도가 63%에서 80%로, AIME24는 46.7%에 도달하며, 이는 기존의 o1-preview를 초과합니다. 총 7,000개의 샘플과 $42의 훈련 비용으로 이루어진 이 연구는, 수천 달러가 소요되던 기존의 모델보다 저렴하면서도 경쟁력을 유지하는 LLM 개발의 가능성을 보여줍니다.

### [XAttention: Block Sparse Attention with Antidiagonal Scoring](https://arxiv.org/abs/2503.16428)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16428.png)

Vote: 7

Authors: Song Han, Haofeng Huang, Junxian Guo, Guangxuan Xiao, Ruyi Xu

- ***What's New***: XAttention은 Long-Context Transformer Models(LCTMs)의 연산 효율성을 크게 향상시키는 새로운 프레임워크로, 블록 희소 Attention(Block Sparse Attention)과 역대각선 점수(Antidiagonal Scoring)를 결합합니다. 이 방법을 통해 비필요한 블록을 효과적으로 식별하고 제거하여 높은 희소성을 유지하면서도 정확도를 유지할 수 있습니다.
- ***Technical Details***: XAttention은 Attention 매트릭스 내에서 역대각선의 합을 블록 중요도 측정에 활용하여, 중요도가 낮은 블록을 손쉽게 제거합니다. 중요한 Attention 블록을 선택하기 위해 역대각선 합을 계산하고, softmax 정규화를 통해 이 합의 확률 분포를 생성하여 최적의 블록을 결정합니다. 이를 통해 LCTMs에서 블록 희소 Attention을 효율적으로 구현할 수 있습니다.
- ***Performance Highlights***: XAttention은 다양한 긴 컨텍스트 벤치마크(RULER, LongBench, VideoMME, VBench)에서 완전한 Attention과 유사한 정확도를 기록하면서 최대 13.5배의 연산 가속을 달성했습니다. 이는 LCTMs의 실용적인 적용 가능성을 높여, 멀티모달 AI 분야에서의 확장성과 효율성을 증진시킵니다.

### [SALT: Singular Value Adaptation with Low-Rank Transformation](https://arxiv.org/abs/2503.16055)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16055.png)

Vote: 7

Authors: Hu Wang, Sarim Hashmi, Mohammad Yaqub, Mohammed Elseiagy, Ibrahim Almakky, Abdelrahman Elsayed

- ***What's New***: SALT는 SVD(Singular Value Decomposition) 기반의 지배적인 특이값 적응과 저랭크(LoRA; Low-Rank Adaptation) 잔여 업데이트를 통해 매개변수 효율적인 도메인 적응을 달성하는 새로운 하이브리드 PEFT(Parameter-Efficient Fine-Tuning) 방법입니다.
- ***Technical Details***: SALT는 세분화 모델의 이미지 인코더 내 다중-헤드 어텐션 레이어에서 SVD와 LoRA 기법을 결합하여 최적화된 적응을 수행합니다. 이는 특이값 적응을 통해 중요한 정보를 유지하며, 저랭크 업데이트를 통해 잔여 성분을 처리하여 최소한의 매개변수만을 활용하면서 적응성을 높입니다. SALT는 파라미터 효율성을 유지하면서도 높은 적응성을 제공하며, SAM 기반의 이미지를 정밀하게 분할할 수 있도록 설계되었습니다.
- ***Performance Highlights***: 5개의 도전적인 의료 데이터셋에서 SALT는 SOTA 라이벌인 LoRA 및 SVD를 Dice 점수 기준으로 2%에서 5%까지 앞서며, 훈련 가능한 매개변수 비율은 3.9%에 불과합니다. 다양한 데이터셋에서 LoRA 및 기존 DL 방법론보다 더 높은 Dice 점수를 지속적으로 기록하고, 주요 부분에서 HD95 지표에서도 뛰어난 성능을 보여줍니다.

### [LHM: Large Animatable Human Reconstruction Model from a Single Image in Seconds](https://arxiv.org/abs/2503.10625)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.10625.png)

Vote: 7

Authors: Weichao Shen, Lingteng Qiu, Weihao Yuan, Guanying Chen, Zilong Dong, Liefeng Bo, Kejie Qiu, Peihao Li, Qi Zuo, Xiaodong Gu, Junfei Zhang

- ***What's New***: LHM(Large Animatable Human Reconstruction Model)은 단일 이미지로부터 3D 사람 아바타를 초 단위로 재구성할 수 있는 피드포워드 방식의 새로운 모델을 제안합니다. LHM은 복합 모달의 트랜스포머 구조를 활용하여 사용자가 원하는 포즈로 조절 가능한 애니메이션과 실시간 렌더링을 지원하는 높은 정밀도의 3D 아바타를 생성합니다.
- ***Technical Details***: LHM은 3D 가우시안 스플래팅(3D Gaussian Splatting)을 사용하여 현실감 있는 렌더링과 포즈 제어가 가능한 기본적인 3D 아바타를 생성합니다. 이 모델은 멀티모달 바디-헤드 트랜스포머(Multimodal Body-Head Transformer; MBHT)를 도입하여 3D 지오메트리에서 추출한 특징들과 이미지 특징들을 주의 기법을 통해 효과적으로 융합합니다. 얼굴의 높은 정체성 보존과 세부 정보 복원을 위해 머리 특징 피라미드 인코딩(head feature pyramid encoding)을 추가적으로 사용합니다.
- ***Performance Highlights***: LHM은 기존의 모형들을 뛰어넘는 재구성 정확도와 일반화 능력을 보였으며, 실험 결과 PSNR와 SSIM 지표에서 우수한 성능을 보였습니다. 이 모델은 2초에서 최대 6초 내외의 짧은 시간 안에 3D 아바타를 생성하며, GPU 메모리 사용량은 대략 18GB~24GB 사이입니다. 전반적인 성능 향상은 실세계 이미지에 대한 모델 적응성을 증명합니다.

### [Towards Unified Latent Space for 3D Molecular Latent Diffusion Modeling](https://arxiv.org/abs/2503.15567)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.15567.png)

Vote: 6

Authors: Yanchen Luo, Xiang Wang, Zhiyuan Liu, Sihang Li, Tat-Seng Chua, Kenji Kawaguchi, Yi Zhao

- ***What's New***: UAE-3D는 3D 분자 생성을 위한 통합 잠재 공간(Unified Latent Space)을 제안하는 혁신적인 모델입니다. 다중 양식 데이터를 단일 잠재 공간에 압축하여 분자의 원자 유형, 화학 결합, 3D 좌표를 통합하고, 근접 손실 없는 재구성 오류(near-zero reconstruction error)를 유지하면서 모델링의 복잡성을 줄입니다. 이 방식은 Diffusion Transformer를 활용하여, 분자에 대한 고유한 편향(bias) 없이도 잠재 공간에서 효과적인 생성 성능을 발휘합니다.
- ***Technical Details***: UAE-3D는 3D 분자의 다중 양식을 통합된 잠재 공간으로 압축합니다. Relational Transformer를 활용하여 여러 양식의 특징을 효과적으로 통합하고, SE(3)-나머지 변환을 통해 3D 좌표에 관련된 변환의 일관성을 유지합니다. 잠재 공간에서 생성 모델은 Diffusion Transformer(DiT)를 사용하여 효율성을 높이고, 분자 데이터에 대한 편향 없이 강력한 성능을 냅니다.
- ***Performance Highlights***: UAE-3D는 QM9 및 GEOM-Drugs 데이터셋에서 de novo 및 조건부 3D 분자 생성 모두에서 최첨단 성능을 달성했습니다. 특히, 분자 구조의 정확성과 화학적 타당성을 동시에 개선함으로써 대조군 대비 성능을 크게 앞서갔습니다. UDM-3D는 기존 모델들 보다 학습과 샘플링에서 2.7배에서 7.3배 더 빠른 효율성을 보였습니다.

### [NuiScene: Exploring Efficient Generation of Unbounded Outdoor Scenes](https://arxiv.org/abs/2503.16375)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16375.png)

Vote: 6

Authors: Angel X. Chang, Qinghong Han, Han-Hung Lee

- ***What's New***: NuiScene는 대형 실외 장면의 효율적인 생성을 다루며 씬 청크를 균일한 벡터 집합으로 인코딩하여 기존의 공간 구조된 잠재(latents) 방법보다 더 나은 압축 및 성능을 제공합니다. 또한 뷰 마다 지오메트리를 고정된 스케일로 통합하는 NuiScene43 데이터셋을 큐레이션하여 서로 다른 스타일의 장면을 융합할 수 있는 모델을 훈련합니다.
- ***Technical Details***: NuiScene는 씬 청크(scene chunk)를 벡터 세트로 인코딩하여 일관된 표현을 제공하며, 3D-Shape2VecSet을 사용해 압축 효율성을 높입니다. 또한, 명시적으로 아웃페인팅(outpainting)하는 확산(diffusion) 모델을 훈련하여 재샘플링 기반의 인페인팅(inpainting) 방식보다 빠른 추론 속도를 보입니다. 이 과정에서 43개의 높은 품질 장면 데이터셋이 사용된다는 점이 특징입니다.
- ***Performance Highlights***: NuiScene의 벡터 세트 기반 모델은 전통적인 트라이플레인(triplane) 기반 모델보다 훈련 속도와 성능면에서 우위에 있습니다. 특히, 벡터 세트 모델은 더 적은 토큰으로 VRAM 사용량을 절반으로 줄이며 트라이플레인 대비 2.5배 빠르게 훈련됩니다. 또한, 훈련된 여러 씬을 기반으로 다양한 스타일들이 융합된 새로운 장면 생성이 가능합니다.

### [Uni-3DAR: Unified 3D Generation and Understanding via Autoregression on Compressed Spatial Tokens](https://arxiv.org/abs/2503.16278)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16278.png)

Vote: 5

Authors: Linfeng Zhang, Guolin Ke, Zhifeng Gao, Lin Yao, Shuqi Lu, Haowei Lin, Weinan E, Xiaohong Ji

- ***What's New***: Uni-3DAR는 3D 구조 생성과 이해를 하나의 프레임워크로 통합한 혁신적인 모델로, 압축된 공간 토큰(Compressed Spatial Tokens)을 이용한 오토리그레션(Autoregressive) 방식을 채택하여 기존의 확산 모델(Diffusion Models)을 크게 능가하면서도 빠른 추론 속도를 제공합니다.
- ***Technical Details***: Uni-3DAR는 옥트리(Octree)를 활용한 새로운 계층적 토큰화(Hierarchical Tokenization) 방법을 사용합니다. 이 방법은 3D 구조의 공간적 스파시티(Sparsity)를 활용하여 토큰의 수를 줄이고, 중요한 원자 유형과 좌표를 캡처할 수 있습니다. 또한, 2단계 서브트리 압축(2-Level Subtree Compression)과 동적 다음 토큰 위치에 맞춘 마스크된 다음 토큰 예측(Masked Next-Token Prediction)을 제안하여 모델의 효율성과 성능을 향상시킵니다.
- ***Performance Highlights***: Uni-3DAR은 다양한 실험에서 기존 최첨단 모델을 크게 뛰어넘었으며, 최대 256%의 성능 향상을 이루고 추론 속도에서 최대 21.8배 더 빠른 결과를 보였습니다. 이는 현재의 3D 구조 모델링에서 중요한 혁신을 나타냅니다.

### [CLS-RL: Image Classification with Rule-Based Reinforcement Learning](https://arxiv.org/abs/2503.16188)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16188.png)

Vote: 5

Authors: Yuxiang Lai, Ming Li, Jike Zhong, Shitian Zhao, Kaipeng Zhang

- ***What's New***: 이 연구는 소수 샷(Small-shot) MLLMs의 최적화에 있어 CLS-RL이라는 새로운 루프 기반 강화 학습(RL) 기법을 도입했습니다. 이것은 클래스 이름을 검증 신호로 사용하여 더 나은 모델 성능을 이끌어내는 방식입니다.
- ***Technical Details***: CLS-RL은 DeepSeek-R1의 성공에 영감을 받아 그룹 상대 정책 최적화(Group Relative Policy Optimization; GRPO)를 사용하여 최적화를 수행합니다. 이 방법은 포맷 보상(Format Reward)과 정확도 보상(Accuracy Reward)으로 구성된 보상 함수를 활용하여 모델이 응답 전에 사고 과정을 거치도록 유도합니다. 반면 No-Thinking-CLS-RL은 사고 과정을 최소화하고, 정확성 보상만 활용하여 모델이 즉각적으로 답을 출력하도록 설계되었습니다.
- ***Performance Highlights***: CLS-RL은 대부분의 데이터셋에서 SFT를 능가하고, 평균 정확도에서 더 높은 성과를 보여주었습니다. 특히, '무료 점심 현상(Free-lunch phenomenon)'을 발견했으며, 이는 특정 데이터셋에 대해 미세 조정된 모델이 다른 독립적인 데이터셋에서도 성능이 개선되는 현상입니다. No-Thinking-CLS-RL 방법은 훈련 시간과 자원을 크게 줄이면서도 CLS-RL보다 더 높은 성과를 기록했습니다.

### [Make Your Training Flexible: Towards Deployment-Efficient Video Models](https://arxiv.org/abs/2503.14237)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.14237.png)

Vote: 5

Authors: Kunchang Li, Yi Wang, Xiangyu Zeng, Tianxiang Jiang, Limin Wang, Chenting Wang

- ***What's New***: 후렉스(Flux)라는 새로운 증강 도구를 제안하여 비디오 모델의 샘플링 그리드를 유연하게 만들어 비용을 거의 추가하지 않고도 모델의 강건성을 높입니다. Flux를 대규모 비디오 사전 학습에 통합하여 기존의 최첨단 결과를 표준 비용으로 초과 달성합니다.
- ***Technical Details***: 토큰 선택(Token Optimization)을 최대화하여 입력 정보를 최대로 하기 위해 설계된 Flux는 비디오에서 보다 적절하게 샘플링된 토큰 중 선택을 통해 입력 토큰 집합을 최적화합니다. 기본적으로 UMT(Unmasked Teacher) 사전 학습 및 감독 학습, 다중 모달 대비 학습에서 Flux를 통합하여 입출력 가변성을 지원합니다. 두 개의 플러그인 모듈, 즉 이중 패치 정규화(Dual Patch Normalization)와 전역-로컬 위치적 임베딩(Global-Local Positional Embedding)을 제안합니다.
- ***Performance Highlights***: FluxViT-S는 K400 기준으로 기존 소규모 최첨단 모델인 InternVideo2-S보다 2.2% 더 높은 성능을 보였고, 실행 비용의 약 10%만 사용하여 유사한 성능을 달성했습니다. FluxViT-B는 장면 기반 행동 인식 및 모션 집약적 작업에서 최첨단 대규모 모델과 경쟁력 있는 결과를 달성했습니다.

### [Agents Play Thousands of 3D Video Games](https://arxiv.org/abs/2503.13356)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.13356.png)

Vote: 5

Authors: Tao Yu, Zhongwen Xu, Qiang Fu, Liang Wang, Siyi Li, Xianliang Wang, Wei Yang

- ***What's New***: PORTAL라는 새로운 프레임워크를 소개합니다. 이는 수천 개의 3D 비디오 게임을 플레이할 수 있는 AI 에이전트를 개발하는 데 사용하는 혁신적인 접근 방식을 제시합니다. 이 접근 방식은 대형 언어 모델(LLMs)을 활용하여 행동 트리를 생성하며, 기존 강화 학습의 한계를 극복하고, 전략적 깊이를 유지하면서 빠른 적응성을 제공합니다.
- ***Technical Details***: PORTAL은 LLMs를 전략 디자이너로 활용하여 도메인 특화 언어(DSL)로 표현된 행동 트리를 생성합니다. 하이브리드 정책 구조로 규칙 기반 노드와 신경망 구성 요소를 결합하여 높은 수준의 전략적 추론과 정교한 저수준 제어가 가능합니다. 이 접근 방식은 게임 메트릭과 비전-언어 모델 분석을 통합하여 반복적인 정책 개선을 지원합니다. 정책들은 즉시 배포 가능하며, 인간이 해석할 수 있고, 다양한 게임 환경에서 일반화가 가능합니다.
- ***Performance Highlights***: PORTAL은 수천 개의 1인칭 슈팅 게임에서의 실험을 통해 개발 효율성, 정책 일반화 및 행동 다양성에서 전통적인 접근 방식에 비해 상당한 개선을 나타냈습니다. 이 시스템은 현실적인 게임 엔진과의 통합을 통해 실시간 성능을 유지하면서 LLM 기반의 전략적 설계의 이점을 제공함으로써 시간 민감한 어플리케이션에 적합합니다.

### [Zero-1-to-A: Zero-Shot One Image to Animatable Head Avatars Using Video Diffusion](https://arxiv.org/abs/2503.15851)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.15851.png)

Vote: 5

Authors: Zhou Zhenglin, Ma Fan, Fan Hehe, Chua Tat-Seng

- ***What's New***: Zero-1-to-A는 비디오 확산(video diffusion) 모델을 사용하여 한 장의 이미지로부터 애니메이션 가능한 4D 아바타를 생성하는 새로운 접근법을 제안합니다. 이 방법은 SymGEN이라는 새로운 메소드를 도입하여 공간적, 시간적 일관성을 갖춘 데이터셋을 점진적으로 합성하여 아바타 재구성을 수행합니다.
- ***Technical Details***: Zero-1-to-A는 두 가지 주요 구성요소로 나뉩니다. 첫째, SymGEN은 데이터셋의 갱신 가능성을 이용하여 비디오 확산 결과를 캐시하고 아바타 생성과 데이터셋 구조의 상호 이익을 통한 일관성을 향상합니다. 둘째, 점진적 학습(Progressive Learning) 전략을 활용하여, (1) 공간적 일관성 학습(Spatial Consistency Learning)과 (2) 시간적 일관성 학습(Temporal Consistency Learning)을 통해 간단한 시나리오에서 복잡한 시나리오로 발전시킵니다.
- ***Performance Highlights***: Zero-1-to-A는 기존의 확산 기반 방법들에 비해 충실도, 애니메이션 품질과 렌더링 속도 전반에서 우수한 성능을 보여줍니다. 특히, 대조 실험을 통해 몇몇 변형된 방법들과 비교하여 더 나은 결과를 보이며, 아바타 생성에 있어서 견고하고 데이터 효율적인 솔루션을 제공합니다.

### [Sonata: Self-Supervised Learning of Reliable Point Representations](https://arxiv.org/abs/2503.16429)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16429.png)

Vote: 4

Authors: Jakob Engel, Nan Yang, Duncan Frost, Hengshuang Zhao, Xiaoyang Wu, Richard Newcombe, Tianwei Shen, Chris Xie, Daniel DeTone, Julian Straub

- ***What's New***: Sonata는 신뢰할 수 있는 포인트 표현(Point Representations)의 자가 지도 학습(Self-Supervised Learning)을 위한 프레임워크를 제안합니다. 이 연구는 기존의 3D 자가 지도 학습 접근법들이 표현의 질을 평가할 때 낮은 성능을 보이는 이유가 '기하학적 쇼트컷(Geometric Shortcut)' 때문이라고 보고 이를 해결하기 위해 두 가지 핵심 전략을 통해 접근합니다.
- ***Technical Details***: Sonata는 140k 포인트 클라우드 데이터셋을 자가-디스틸레이션(Self-Distillation) 프레임워크로 활용하여 구성되며, 로컬 뷰(Local View)와 마스크드 뷰(Masked View)를 사용한 로컬-글로벌 뷰 정렬과 마스크-언마스크 뷰 정렬을 통해 기하학적 쇼트컷을 해결하고자 합니다. 이 프레임워크는 하이퍼칼럼(Hypercolumn)과 같은 기능 업캐스팅(Feature Up-casting)을 도입하여 다중 스케일 인코딩을 제공합니다.
- ***Performance Highlights***: Sonata는 ScanNet에서 선형 프로빙(Linear Probing)으로 21.8%에서 72.5%로 성능이 3.3배 증가했으며, 이는 이전 SOTA 방식의 성능을 뛰어넘습니다. 또한, DINOv2 특징을 결합하여 정확도를 더욱 향상시킵니다. 다양한 실내 및 실외 인식 작업에서 SOTA를 초과하는 성능을 보였습니다.

### [MagicID: Hybrid Preference Optimization for ID-Consistent and Dynamic-Preserved Video Customization](https://arxiv.org/abs/2503.12689)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.12689.png)

Vote: 3

Authors: Xi Xiao, Deng Cai, Tianyang Wang, Hengjia Li, Boxi Wu, Lifan Jiang, Hongwei Yi

- ***What's New***: MagicID는 사용자 이미지 참조를 기반으로 아이디 일관성을 유지하면서 동적인 비디오를 생성할 수 있는 하이브리드 선호도 최적화 프레임워크입니다. 기존의 자기 재구성 문제를 해결하기 위해, 쌍별 선호 비디오 데이터를 구축하고 하이브리드 샘플링 전략을 사용하여 아이디와 역동성을 모두 우선시합니다.
- ***Technical Details***: MagicID는 Direct Preference Optimization (DPO)을 기반으로 사용자 제공 이미지에서 아이디 일관성을 유지하고 자연스러운 동작 역동성을 보존하도록 설계되었습니다. 하이브리드 샘플링 전략을 도입하여 첫 번째로 정적 비디오에서 아이디를 우선시하고, 이를 바탕으로 Frontier 기반의 샘플링 방법을 사용하여 역동성을 강화한 비디오를 생성합니다. 이렇게 생성된 쌍별 선호 데이터는 ID 일관성과 동적 품질을 모두 최적화합니다.
- ***Performance Highlights***: MagicID는 기존의 T2V 커스터마이징 방법들보다 우수한 성능을 보여주며, 얼굴 유사도와 동작 역동성 측면에서 특히 뛰어난 결과를 보였습니다. 이는 사용자 선호도에 더 잘 부합하는 비디오 출력을 가능하게 하며, 최종 사용자의 평가에서도 높은 선호도를 기록했습니다.

### [Deceptive Humor: A Synthetic Multilingual Benchmark Dataset for Bridging Fabricated Claims with Humorous Content](https://arxiv.org/abs/2503.16031)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16031.png)

Vote: 3

Authors: Sai Kartheek Reddy Kasu, Sunil Saumya, Shankar Biradar

- ***What's New***: 이 논문에서는 기만적인 유머(Deceptive Humor) 데이터셋(DHD)을 소개합니다. 이는 사실 왜곡과 허위 주장에서 비롯된 유머를 연구하기 위한 새로운 다국어 벤치마크 데이터셋으로, 사실 인식 유머(Fact-Aware Humor)를 다양한 언어와 코드-믹스된 문맥에서 분석하는 최초의 합성 데이터셋을 제공합니다.
- ***Technical Details***: ChatGPT-4o 모델을 사용하여 허위 주장과 조작된 정보를 기반으로 유머 코멘트를 생성하였습니다. 각 샘플은 1부터 3까지의 사티르 레벨(Satire Level)로 레이블링되며, 다섯 가지 유머 유형(다크 유머(Dark Humor), 아이러니(Irony), 사회적 논평(Social Commentary), 워드플레이(Wordplay), 부조리(Absurdity)) 중 하나로 분류됩니다. 데이터셋은 영어, 텔루구, 힌디, 칸나다, 타밀어 등과 이러한 언어들의 코드-믹스 변형(Te-En, Hi-En, Ka-En, Ta-En)을 포함합니다.
- ***Performance Highlights***: 다양한 공개 소스 모델을 사용하여 데이터셋을 평가한 결과, mBART 모델이 사티르 레벨의 분류에서 가장 우수한 성능을 보였으며, BERT 모델은 유머 속성의 분류에서 높은 성능을 보였습니다. QLoRA 방식으로 Fine-Tuning한 결과가 Zero-Shot 및 Few-Shot 모델보다 개선된 성능을 보였으나, 기망적인 유머는 여전히 파악하기 어려운 영역으로, 더 깊은 연구가 필요함을 보여줍니다.

### [MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](https://arxiv.org/abs/2503.15451)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.15451.png)

Vote: 3

Authors: Xiaowei Zhou, Huaijin Pi, Liang Pan, Ziyong Feng, Jingbo Wang, Lixing Xiao, Yueer Zhou, Ke Fan, Sida Peng, Shunlin Lu

- ***What's New***: MotionStreamer는 텍스트 기반의 스트리밍 모션 생성 문제를 해결하기 위한 혁신적인 프레임워크입니다. 이 모델은 연속적인 인과(latent causal space) 공간을 활용하며, 확산 기반( diffusion-based) 오토레그레시브 모델(autoregressive model)을 결합하여 기존 모델이 가진 문제들을 해결합니다. 이를 통해 더욱 실시간 반응이 가능하며, 긴 시간에 걸친 모션 생성에서도 누적 오류를 효과적으로 줄여줍니다.
- ***Technical Details***: MotionStreamer는 텍스트 조건부 스트리밍 모션 생성을 원활히 하기 위해 인과 템포럴 오토인코더(Causal Temporal AutoEncoder)를 사용합니다. 이 수동 인코더는 모션을 인과적 연속 잠재 공간으로 압축하여 온라인 디코딩이 가능하게 하며, 텍스트와 역사적인 모션 정보를 오토레그레시브 모델에 결합하여 다음 모션 잠재를 예측합니다. 더불어 혼합 학습(Mixed training) 전략과 투포워드 학습(Two-forward training) 전략을 통해 테스트 시점 분포를 훈련 시점에 도입하여 오류 누적 문제를 완화했습니다.
- ***Performance Highlights***: MotionStreamer는 HumanML3D 및 BABEL 데이터셋에서 기존 텍스트-모션 생성 및 장시간 모션 합성 작업에 최신 성능을 달성했습니다. 이러한 스트리밍 모션 생성 프레임워크는 온라인 다중 반복 생성, 여러 텍스트와의 장기 모션 생성 및 동적 모션 컴포지션 같은 다양한 응용 프로그램에 적합합니다.

### [UVE: Are MLLMs Unified Evaluators for AI-Generated Videos?](https://arxiv.org/abs/2503.09949)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.09949.png)

Vote: 3

Authors: Haoyuan Guo, Jiacong Wang, Shuhuai Ren, Xu Sun, Rui Zhu, Lu Jiang, Yuanxin Liu

- ***What's New***: UVE는 멀티모달 대형 언어 모델(Multimodal Large Language Models; MLLMs)을 사용하여 AI 생성 비디오(AIGVs)를 평가하는 통합적 접근 방식을 제안합니다. 이는 자동 평가 지표(AIGV)의 통합적 평가 가능성을 탐색하기 위한 새로운 벤치마크, UVE-Bench를 도입합니다.
- ***Technical Details***: UVE-Bench는 최신 비디오 생성 모델(Video Generative Models; VGMs)로 생성된 비디오를 수집하고, 15개의 평가 측면에 대한 인간의 선호도 주석을 제공합니다. 16개의 MLLMs를 이용하여 비디오 평가를 수행하고, 이러한 모델의 설계 선택이 평가 성능에 미치는 영향을 분석합니다.
- ***Performance Highlights***: Qwen2-VL-72B 및 InternVL2.5-78B와 같은 고급 MLLMs는 기존 특수화된 평가 방법을 크게 초월하는 기능을 보여주지만, 인간 평가자와의 큰 성능 격차가 존재합니다. 특히, 비디오의 시간적 역학을 파악해야 하는 세부적인 이해 측면에서 아직 부족한 모습을 보입니다.

### [Improving Autoregressive Image Generation through Coarse-to-Fine Token Prediction](https://arxiv.org/abs/2503.16194)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16194.png)

Vote: 2

Authors: Michael Qizhe Shieh, Kaipeng Zhang, Ziyao Guo

- ***What's New***: 새로운 연구는 대형 코드북(Large Codebooks)을 사용하면서도 오토레그레시브 이미지 생성(Autoregressive Image Generation)의 복잡성을 줄이는 Coarse-to-Fine Token Prediction(CFT)을 제안합니다. 기존의 이미지 생성 방법이 코드북의 확장으로 복잡해지는 문제를 해결하면서도 높은 재구성 품질을 유지하고자 합니다.
- ***Technical Details***: 제안된 프레임워크는 두 단계로 구성됩니다: (1) 오토레그레시브 모델은 시퀀스의 각 토큰에 대한 조잡한 레이블(Clustering)을 순차적으로 예측합니다. k-means 클러스터링을 통해 유사한 코드워드(Codewords)를 그룹화하고 각 클러스터에 조잡한 레이블을 지정합니다. (2) 보조 모델이 주어진 조잡한 레이블에 따라 모든 토큰의 세부 레이블(Fine Labels)을 한 번에 예측합니다. 이 방법은 코드북 사이즈를 유지하면서 오토레그레시브 모델링의 어려움을 줄입니다.
- ***Performance Highlights***: ImageNet 실험 결과, 제안된 방법이 Inception Score에서 평균 59점 향상을 기록하며, 샘플링 속도도 개선되었습니다. FID 점수에서 최대 1점 감소가 있었고, 추가적인 추론 단계에도 불구하고, 오히려 모델은 더 빠른 샘플링 속도를 보여주었습니다. 코드와 모델 가중치는 GitHub(https://github.com/GzyAftermath/CTF)에서 공개될 예정입니다.

### [See-Saw Modality Balance: See Gradient, and Sew Impaired Vision-Language Balance to Mitigate Dominant Modality Bias](https://arxiv.org/abs/2503.13834)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.13834.png)

Vote: 2

Authors: Eunju Lee, MiHyeon Kim, YoungBin Kim, Juhwan Choi, JuneHyoung Kwon

- ***What's New***: 이 연구는 비전-언어(Vision-Language; VL) 모델의 주요 모달리티 편향 문제를 해결하기 위한 새로운 프레임워크인 BALGRAD를 제안합니다. 이 방법은 모달리티 간 기울기를 재조정(inter-modality gradient reweighting)하고, 각 모달리티의 학습 기여도에 따라 KL 발산(KL divergence)의 기울기를 조정하며, 비충돌 방식으로 작업 방향을 정렬하는 inter-task gradient projection을 포함합니다.
- ***Technical Details***: BALGRAD는 모달리티 간 기울기 재조정(inter-modality gradient reweighting)과 작업 간 기울기 투영(inter-task gradient projection)을 통해 주 모달리티 편향을 완화합니다. 이를 통해 서로 다른 기울기 크기로 인한 불균형을 해결하고, 모달리티 간의 기울기 방향을 정렬하여 효과적인 공동 학습을 촉진합니다.
- ***Performance Highlights***: UPMC Food-101, Hateful Memes, 그리고 MM-IMDb 데이터셋에서 BALGRAD는 다른 모달리티에 의존하지 않는, 즉 주 모달리티 편향을 효과적으로 완화하는 높은 성능을 보였습니다. 특히, BALGRAD는 각 데이터셋에서 평균 성능과 ∆Gap 측정에서 가장 작은 편향을 보여주며 모델의 견고성을 입증하였습니다.

### [VideoRFSplat: Direct Scene-Level Text-to-3D Gaussian Splatting Generation with Flexible Pose and Multi-View Joint Modeling](https://arxiv.org/abs/2503.15855)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.15855.png)

Vote: 2

Authors: Byeongjun Park, Hyelin Nam, Byung-Hoon Kim, Hyungjin Chung, Hyojun Go, Changick Kim

- ***What's New***: VideoRFSplat은 비디오 생성 모델을 활용하여 실제 3D Gaussian Splatting(3DGS)을 수치 연산 없이 텍스트로부터 직접 생성하는 새로운 모델입니다. 기존 방법들이 2D 생성 모델의 한계를 초과하려는 시도를 넘어 여러 뷰어 카메라 포즈와 이미지 간의 간섭을 줄이며 텍스트 프롬프트에 맞는 고품질의 3D 장면을 생성합니다.
- ***Technical Details***: VideoRFSplat은 사전 훈련된 비디오 생성 모델과 독립적인 자세 생성 모델을 결합하는 이중 스트림 구조를 제안합니다. 각 모듈이 자신의 타임스텝으로 운영되어 비동기적인 샘플링 전략을 구현하며, 상대적인 모호함을 줄여 교차 모달 일관성을 높입니다. 이를 통해, 카메라 포즈와 다중 시점 이미지의 동시 생성을 더욱 안정되게 합니다.
- ***Performance Highlights***: VideoRFSplat은 RealEstate10K, MVImgNet, DL3DV-10K, ACID와 같은 다수의 대규모 실제 세계 데이터세트에서 학습되었으며, SDS++ 재정비에 의존하지 않고도 선행 방법보다 우수한 성능을 기록했습니다. T3Bench와 같은 벤치마크에서 평가 결과, VideoRFSplat은 모든 평가 지표에서 뛰어난 성능을 보여주었습니다. 이는 효율적인 아키텍처 설계와 개선된 샘플링 전략을 통해 이루어진 결과입니다.

### [Painting with Words: Elevating Detailed Image Captioning with Benchmark and Alignment Learning](https://arxiv.org/abs/2503.07906)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.07906.png)

Vote: 1

Authors: Chunyuan Li, Xianhan Zeng, Fu Li, Haoqi Fan, Qinghao Ye

- ***What's New***: 이 논문은 VLM(Vision-Language Models)의 세밀한 이미지 캡션 작성 능력을 평가하기 위해 DECAPBENCH라는 새로운 평가 벤치마크와 DCSCORE라는 새로운 평가 메트릭을 도입했습니다. DCSCORE는 원시 정보 단위(Primitive Information Units)로 응답을 분해하여 세밀한 이해와 환각(hallucination)을 개별적으로 평가합니다.
- ***Technical Details***: DCSCORE는 이미지 설명 문장들을 가장 작은 자급자족의 단위인 원시 정보 단위로 분해하여 평가를 진행합니다. DECAPBENCH는 이러한 세밀한 요소들을 추가적으로 평가할 수 있는 광범위한 이미지를 포함하고 있으며, FEEDQUILL은 자동화된 피드백 수집 방법을 통해 선호 최적화를 지원합니다. 이러한 방법론은 다중 모드 모델(VLMs)의 성능 향상에 기여합니다.
- ***Performance Highlights***: FEEDQUILL을 적용한 모델은 MMC(Multi-Modal Chat)와 세밀한 이미지 캡셔닝 분야에서 GPT-4o를 능가하는 성능을 보였고, 환각 현상을 최대 40.5%까지 줄였습니다. DECAPBENCH의 측정 결과, DCSCORE가 기존 규칙 기반 및 모델 기반 메트릭보다 더 높은 인간 평가 일치성을 보였습니다.

### [AIMI: Leveraging Future Knowledge and Personalization in Sparse Event Forecasting for Treatment Adherence](https://arxiv.org/abs/2503.16091)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.16091.png)

Vote: 1

Authors: Hassan Ghasemzadeh, Diane J. Cook, Abdullah Mamun

- ***What's New***: AIMI 시스템은 처방된 약물 복용의 예측 및 개입을 개선하기 위해 스마트폰 및 웨어러블 센서를 활용하여 미래 지식(Future Knowledge)와 개인화를 통합한 치료 순응도(Treatment Adherence) 예측 시스템입니다. 이는 웨어러블 센서를 통한 치료 순응도 예측이 덜 확립된 상황에서 새로운 접근 방식을 제공합니다.
- ***Technical Details***: AIMI는 센서 데이터와 과거 약물 복용 기록, 미래 지식을 사용하여 복용 불이행 가능성을 예측합니다. 이를 위해 CNN 및 LSTM 기반 예측 모델을 개발하였으며, 시계열 데이터 처리 및 사용자 맞춤형 학습을 통해 모델의 정확성을 향상시켰습니다. 특히 장소 정보와 미래 처방 시간은 주요 기능으로 사용되었습니다.
- ***Performance Highlights***: 사용자 27명의 데이터로 훈련된 AIMI 시스템의 LSTM 모델은 약물 복용 순응도 예측에서 0.932의 정확도와 0.936의 F-1 점수를 보여주었습니다. 이 결과는 미래 지식을 포함하여 설계된 네트워크가 적절한 데이터 크기로 훈련될 경우 상당한 성능 향상을 가져올 수 있음을 시사합니다.

### [TikZero: Zero-Shot Text-Guided Graphics Program Synthesis](https://arxiv.org/abs/2503.11509)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.11509.png)

Vote: 1

Authors: Eddy Ilg, Jonas Belouadi, Masao Utiyama, Margret Keuper, Simone Paolo Ponzetto, Hideki Tanaka, Steffen Eger, Raj Dabre

- ***What's New***: TikZero는 제로샷 텍스트 기반 그래픽 프로그램 생성(Zero-Shot Text-Guided Graphics Program Synthesis)을 가능하게 하는 새로운 접근법입니다. 이는 이미지 표현을 중간 매개체로 사용하여 그래픽 프로그램 생성과 텍스트 이해를 분리합니다. 이로 인해 캡션이 포함된 그래픽 프로그램의 부족 문제를 해결하며, TikZero는 훨씬 더 큰 데이터에서 학습할 수 있어, GPT-4o 같은 상업용 시스템과 비슷하거나 더 나은 성능을 보여줍니다.
- ***Technical Details***: TikZero는 두 가지 주요 구성 요소로 구성됩니다: 시각적 정보로부터 그래픽 프로그램을 생성하는 역 그래픽 모델(Inverse Graphics Model)과 캡션으로부터 이미지 패치 임베딩을 생성하는 어댑터 네트워크(Adapter Network)입니다. 시각적 텍스트를 기반으로 하는 어댑터 네트워크는 훈련 시 캡션에만 의존해 학습하므로 리소스의 한계를 극복하고 제로샷 텍스트 기반 그래픽 프로그램 생성을 지원합니다.
- ***Performance Highlights***: TikZero는 기존의 최첨단 모델을 능가하며, 특히 이더넷 그래픽 프로그램과 캡션을 결합한 추가 학습 시 주요 상업 시스템인 GPT-4o와 비슷한 성능을 나타냅니다. 캡션-프로그램 쌍을 사용하는 추가 학습에서는 더욱 향상된 성능을 보여줍니다. 새로운 데이터셋인 DaTikZv3는 45만 개 이상의 TikZ 그래픽 프로그램과 17만 개의 캡션 샘플을 포함합니다.

### [Why Personalizing Deep Learning-Based Code Completion Tools Matters](https://arxiv.org/abs/2503.14201)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.14201.png)

Vote: 1

Authors: Gabriele Bavota, Alessandro Giagnorio, Alberto Martin-Lopez

- ***What's New***: 이 논문은 딥러닝(DL)을 기반으로 하는 코드 자동 완성 도구의 성능을 개인화된 데이터로 향상시킬 수 있는지를 연구합니다. 특히, 조직별, 개발자별 맞춤화가 코드 자동 완성의 성능 향상에 어떻게 기여하는지를 대규모 실증 연구를 통해 검증합니다. Apache와 Spring에서 활동 중인 136명의 개발자를 대상으로, T5 및 Code Llama와 같은 다양한 모델을 활용해, 조직 및 개발자 특화 데이터셋을 추가 학습시켜 성능을 비교합니다.
- ***Technical Details***: 연구는 T5와 Code Llama 모델을 사용하여, 최대 7B 파라미터의 대형 모델을 포함한 여러 크기의 모델에서 개인화된 데이터의 효과를 분석합니다. T5 모델은 처음부터 2M 이상의 코드 베이스에서 사전 학습 후, 조직 및 개발자 특화 데이터셋으로 추가 미세 조정을 실시합니다. Code Llama는 공개된 사전 학습 모델을 손쉽게 미세 조정하여 조직 및 개발자 특화 데이터셋으로 비교합니다.
- ***Performance Highlights***: 조직 특화 모델은 7.84%의 정답 예측 증가율을 보이며, Apache 93%, Spring 100% 개발자에게 성능 향상을 제공합니다. 개발자 특화 모델은 76%의 Apache 개발자와 83%의 Spring 개발자에게 성능 향상을 제시하며, 특히 코드를 반복적으로 작성하는 개발자의 경우 더 큰 성능 향상이 관찰되었습니다. 이는 개발자나 조직 수준의 코드를 사용해 모델을 미세 조정함으로써 성능을 크게 향상시킬 수 있음을 나타냅니다. 이 연구는 더 작은 모델을 개인화하여 10배 더 큰 모델과 동일한 성능을 달성할 수 있는 비용 효율성을 강조합니다.

### [LLM-Mediated Guidance of MARL Systems](https://arxiv.org/abs/2503.13553)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.13553.png)

Vote: 0

Authors: Ian Gemp, Philipp D. Siedler

- ***What's New***: 이 논문은 대형 언어 모델(LLM)이 중개하는 다중 에이전트 강화 학습(MARL) 시스템의 가이드라인 제시를 탐구하며, LLM을 통한 중재가 에이전트의 학습 궤적을 바람직한 방향으로 유도하는 방법을 제시합니다.
- ***Technical Details***: 두 가지 유형의 중재자인 자연어(NL) 및 규칙 기반(RB) 컨트롤러를 사용하여 실험을 진행했습니다. NL 컨트롤러는 LLM을 통해 인간의 중재를 시뮬레이션하며, RB 컨트롤러보다 강한 영향을 미쳤습니다. 주로 조기 중재가 훈련 효율성과 성능 향상에 기여했습니다.
- ***Performance Highlights***: NL 및 RB 컨트롤러를 이용한 중재가 중재 없이 수행한 기본 설정보다 더 높은 성능을 보였습니다. 특히 Pharia-1-LLM-7B와 Llama-3.1-8B Instruct 모델을 사용했을 때, 성능 개선이 두드러졌으며, NL 컨트롤러는 자유형 자연어 중재에서 우수한 성능을 보였습니다.

### [Where do Large Vision-Language Models Look at when Answering Questions?](https://arxiv.org/abs/2503.13891)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.13891.png)

Vote: 0

Authors: Ming Li, Longyin Wen, Xiaoying Xing, Ying Wu, Li Fuxin, Fan Chen, Sijie Zhu, Yulei Niu, Chia-Wen Kuo

- ***What's New***: 대규모 시각-언어 모델(Large Vision-Language Models; LVLMs)의 시각적 이해 행동을 해석하기 위한 새로운 방법을 소개합니다. 이 연구는 LVLMs가 시각적 입력에 어느 정도 의존하며, 어떤 이미지 영역이 반응에 기여하는지를 분석합니다.
- ***Technical Details***: 기존의 히트맵 시각화 방법을 확장하여 LVLMs의 자유형식 시각 질문 응답을 해석할 수 있도록 했습니다. 이미지와 생성된 응답 간의 관련성을 반영하는 시각적으로 관련된 토큰을 선택하는 방법을 제안합니다. 또한, 비주얼 인포메이션을 필요로 하는 벤치마크에서 최첨단 LVLMs를 포괄적으로 분석합니다.
- ***Performance Highlights***: 분석 결과, LVLM의 포커스 지역과 응답의 정확성 간에는 관련성이 있으며, 다양한 아키텍처에 따라 시각적 주의가 달라진다는 것, 그리고 LLM의 크기가 시각적 이해에 미치는 영향을 확인했습니다. 다양한 데이터셋에서 삭제 점수와 삽입 점수를 통해 제안 방법의 성능을 평가했으며, 모든 모델과 데이터셋에서 최상의 결과를 보여줍니다.

### [GASP: Unifying Geometric and Semantic Self-Supervised Pre-training for Autonomous Driving](https://arxiv.org/abs/2503.15672)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.15672.png)

Vote: 0

Authors: Lars Hammarstrand, Michael Felsberg, Junsheng Fu, Adam Lilja, Adam Tonderski. Arvid Laveno Ling, Christoffer Petersson, Willem Verbeke, Carl Lindström, William Ljungbergh

- ***What's New***: GASP는 자율주행차를 위한 기하학적 및 의미론적 자기지도 학습 사전 학습(Self-Supervised Pre-training)을 통합하여 다양한 센서 데이터를 활용한 강력한 환경 표현 학습을 가능하게 합니다. 이는 향후의 점유 예측(Occupancy Prediction), 비전 기반 모델(Vision Foundation Model)의 특징 예측, 및 주행 경로(ego-path) 예측을 통해 잉여 데이터 없이 드라이버 보조 시스템(AP)에 기여할 수 있는 획기적인 접근 방법입니다.
- ***Technical Details***: GASP는 과거 센서 데이터를 입력으로 받아 공간적-시간적 연속적인 시공간(Continuous Spacetime)의 점유 필드(Occupancy Field)를 예측하며, 이 과정에서 Lidar와 같은 다양한 센서로부터 감지된 정보를 종합하여 BEV(조감도; Bird’s Eye View) 기능 맵을 생성합니다. 이러한 기능 맵은 DINOv2 모델처럼 사전 학습된 비전 모델(Vision Foundation Model)의 특징을 예측하고 주행 경로를 포함하여 다양한 다운스트림 작업에 활용됩니다. 반사실적 디코더(Implicit Decoder)가 사용되어 각 쿼리 지점에서 점유, 주행 경로, DINOv2 특징을 예측하는 구조를 가지고 있습니다.
- ***Performance Highlights***: GASP는 기존의 UnO 방법론을 능가하는 4D 점유 예측 성능을 보여주었으며, 특히 의미론적 예측 작업에 있어 상당한 성능 향상을 가져왔습니다. GASP를 통한 사전 학습은 다양한 자율주행 작업에서 향상된 일반화 성능을 발휘하였으며, 이는 선행 연구 대비 강화된 학습 전략 결과로 나타났습니다. 이로써 적은 라벨 데이터로도 높은 성능의 모델을 구축할 수 있었습니다.

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
