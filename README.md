# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using GPT-4o.

Thanks to [@AK391](https://github.com/AK391) for great work.

## Daily Papers (2024-12-04)

### [DeMo: Decoupled Momentum Optimization](https://arxiv.org/abs/2411.19870)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2411.19870.png)

Vote: 1

Authors: Jeffrey Quesnelle, Diederik P. Kingma, Bowen Peng

- ***What's New***: 이 연구에서는 대규모 신경망 훈련 시 높은 대역폭의 인터커넥트 없이도 효율적인 학습을 가능하게 하는 새로운 최적화 알고리즘, Decoupled Momentum Optimization (DeMo)를 도입했습니다. DeMo는 모멘텀 업데이트를 분리하여 가속기 간 네트워크 대역폭 요구를 크게 줄이고, 기존의 상용 최적화 알고리즘인 AdamW를 능가하거나 동등한 성능을 달성할 수 있도록 설계되었습니다.
- ***Technical Details***: DeMo는 SGD with Momentum에서 모멘텀을 가속기별로 독립적으로 유지하도록 하기 위해, 그래디언트(Gradient) 전역 동기화를 제거하는 새로운 알고리즘을 기반으로 합니다. 주된 기술적 요소로는 Discrete Cosine Transform (DCT)을 사용하여 모멘텀의 빠르게 변화하는 요소를 추출하고, 이 요소들만 동기화함으로써 통신 오버헤드를 줄입니다. DCT는 프린시플 컴포넌트를 효율적으로 추출할 수 있도록 하며, 이는 모멘텀의 빠르게 변화하는 구성요소를 적절히 분리할 수 있도록 합니다.
- ***Performance Highlights***: DeMo로 훈련된 모델은 100B 토큰에서 기존의 AdamW를 기반으로 한 모델과 비교해 모델 크기별로 유사하거나 더 나은 학습 손실 및 다운스트림 평가 점수를 기록했습니다. 특히 DeMo는 GPU 간 데이터 전송에 필요한 대역폭을 수백 배까지 절감하여 더 적은 통신 요구로 높은 성능을 유지하는데 성공했습니다.

### [Yi-Lightning Technical Report](https://arxiv.org/abs/2412.01253)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.01253.png)

Vote: 18

Authors: Xiaoyi Ren, Chenglin Cai, Yuchi Xu, Feng Hu, Bei Chen, Xinyao Niu, Chujie Zheng, Fan Zhou, Yuxuan Sha, Chengen Huang, Shawn Wang, Xiaohui Hu, Mou Li, Zhiyuan Liu, Daniel Cooper, Heng Ji, Katherine Su, Lihuan Zhang, Ethan Dai, Liying Li, Wen Xie, Albert Wang, 01. AI, Howard Qiu, Jiangcheng Zhu, Peng Liu, Tianhang Zhu, Jun Tian, C. X. Lv, Shiyong Li, Yongzhen Luo, Shijun Zhou, Qichen Hu, Yanpeng Li, Chao Li, Xiang He, Yongke Zhao, Zhaodong Yan, Alan Wake, Zirui Zhang, Xiaobo Chen, Ming Song

- ***What's New***: Yi-Lightning은 우리의 최신 플래그십 대형 언어 모델(LLM)로, Chatbot Arena에서 전체 6위에 오르며, 특히 중국어, 수학, 코딩 및 어려운 문제 영역에서 2위에서 4위를 차지하는 뛰어난 성과를 보였습니다. 이 모델은 향상된 전문가 혼합(Mixture-of-Experts; MoE) 아키텍처를 활용하여 강화된 전문가 세분화와 라우팅 메커니즘을 결합하고 최적화된 KV 캐싱 기술을 포함합니다.
- ***Technical Details***: Yi-Lightning은 세분화된 전문가 세분화 방법론과 향상된 라우팅 전략, 최적화된 키-값(KV) 캐시 감소 기술을 사용하는 전문 혼합 아키텍처를 기반으로 합니다. 훈련 전략에는 사전 훈련, 지도 미세 조정(SFT), 인간 피드백을 활용한 강화 학습(RLHF) 등 다단계 훈련 전략이 포함됩니다. 우리는 안전 문제를 해결하기 위해 RAISE라는 책임 있는 AI 안전 엔진을 구현하여 모델의 전체 라이프사이클에서 안전성을 유지합니다.
- ***Performance Highlights***: Yi-Lightning은 Chatbot Arena에서 사람 간 비교와 투표를 기반으로 한 평가에서 뛰어난 사용자 만족도와 실제 적용에서의 인간 선호도 일치를 보여줍니다. 그러나 전통적인 정적 벤치마크에서의 평가 결과와 실제, 동적 인간 선호도의 차이를 발견했고, 이것이 미래의 AI 시스템 개발에 있어 벤치마크의 유용성을 재평가하도록 촉발했습니다.

### [A dynamic parallel method for performance optimization on hybrid CPUs](https://arxiv.org/abs/2411.19542)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2411.19542.png)

Vote: 5

Authors: Liu Yucheng, Luo Yu, Shen Haihao

- ***What's New***: 이 논문에서는 하이브리드 CPU 상의 성능 최적화를 위한 새로운 동적 병렬 계산 방법을 소개합니다. 특히, llama.cpp 프레임워크 내에서 LLM (Large Language Model) 추론 과정을 최적화하여, 두 개의 하이브리드 Intel CPU에서 메모리 대역폭의 90% 이상을 사용하며 효율성을 크게 향상시킵니다.
- ***Technical Details***: 제안된 방법은 두 주요 구성 요소인 CPU 런타임(runtime)과 스레드 스케줄러(thread scheduler)로 구성됩니다. CPU 런타임은 각 스레드를 물리적 코어에 바인딩하고 커널 실행 중 스레드의 실행 시간을 추적합니다. 스레드 스케줄러는 CPU 런타임에서 CPU 상태를 얻고, 각 코어의 동적 성능에 따라 커널 작업을 서브 태스크로 분할하여 최대한의 추론 성능을 확보합니다. 성능 비율(performance ratio)을 기반으로 작업을 동적으로 분배하여 모든 스레드가 동시에 서브 태스크를 완료하도록 합니다.
- ***Performance Highlights***: 실험 결과, INT8 GEMM(General Matrix Multiply) 시험에서 Ultra-125H에서 65%, Core-12900K에서 85%의 계산 성능 향상을 보였습니다. 또한, INT4 GEMV(General Matrix-Vector Multiplication) 시험에서는 메모리 대역폭이 19% 향상되었으며, MLC (Intel® Memory Latency Checker) 측정값의 90% 이상을 활용할 수 있었습니다. LLM 전체 추론 성능 실험에서는 ACLG(Average Computed Load Growth) 방법보다 최대 30% 향상된 성능을 확인했습니다.

### [SpotLight: Shadow-Guided Object Relighting via Diffusion](https://arxiv.org/abs/2411.18665)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2411.18665.png)

Vote: 1

Authors: Zitian Zhang, Jean-François Lalonde, Louis-Etienne Messier, Mathieu Garon, Frédéric Fortier-Chouinard, Anand Bhattad

- ***What's New***: SPOTLIGHT는 기존의 확산 기반 렌더러(pre-trained diffusion-based neural renderer)를 활용하여 그림자를 통해 객체의 조명을 제어할 수 있는 새로운 방법을 제시합니다. 이 방법은 별도의 추가 훈련 없이도 기존의 확산 모델에서 현실적이고 제어 가능한 재조명을 가능하게 합니다.
- ***Technical Details***: SPOTLIGHT는 주어진 그림자를 기준으로 객체의 조명을 조절하면서도 배경과 조화를 이루도록 설계된 기술입니다. 배경 이미지와 객체의 알베도(albedo) 및 그림자를 사용하여 장면을 조성하고, 클래스-프리 가이던스(classifier-free guidance)를 통해 조명 효과를 증대시킵니다. 중요한 부분은 그림자의 투입을 통해 다양한 기존의 객체 합성 네트워크(object compositing networks)와 확산 모델이 학습한 사전 지식을 활용하여 현실적인 결과를 생성한다는 점입니다.
- ***Performance Highlights***: SPOTLIGHT는 실험 결과와 사용자 연구(user study)를 통해 기존의 조명 제어를 지원하는 확산 기반 모델들을 상회하는 성능을 보였습니다. 특히, ZeroComp 백본을 사용한 SPOTLIGHT는 PSNR 및 SSIM 측면에서 우수한 성능을 나타냈으며, 사용자 연구에서도 가장 현실적인 합성 결과를 얻었습니다.

### [Free Process Rewards without Process Labels](https://arxiv.org/abs/2412.01981)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.01981.png)

Vote: 18

Authors: Wendi Li, Zhiyuan Liu, Hao Peng, Kaiyan Zhang, Bowen Zhou, Huayu Chen, Ganqu Cui, Ning Ding, Lifan Yuan

- ***What's New***: 본 논문은 중간 단계 레이블 없이 프로세스 보상 모델(Process Reward Model; PRM)을 학습할 수 있는 새로운 이론적 및 실험적 접근 방법을 제시합니다. ORM(Outcome Reward Model)을 통해 기존 데이터로 PRM 학습이 가능하다는 점을 보였습니다.
- ***Technical Details***: PRM은 각 중간 단계에서 추론 궤적을 평가하는 모델로, ORM과 달리 다단계 레이블 없이 학습할 수 있는 방법론을 제안합니다. 보상은 정책 모델과 참조 모델의 로그 우도 비(log-likelihood ratio)로 매개변수화되며, CE(Cross-Entropy) 손실을 포함한 여러 학습 목표에 적용됩니다.
- ***Performance Highlights***: MATH 데이터셋에서 기존 MCTS 기반의 강력한 베이스라인을 훨씬 적은 데이터로도 능가하며, CE 손실로 구현된 암시적 PRM은 데이터 효율성이 높아 극단적인 데이터 희소성과 불균형 상황에서도 모델 생성을 지속적으로 개선할 수 있습니다.

### [Collaborative Instance Navigation: Leveraging Agent Self-Dialogue to Minimize User Input](https://arxiv.org/abs/2412.01250)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.01250.png)

Vote: 3

Authors: Edoardo Zorzi, Francesco Taioli, Yiming Wang, Marco Cristani, Alberto Castellini, Alessandro Farinelli, Gianni Franchi

- ***What's New***: Collaborative Instance Navigation (CoIN)는 실시간으로 사용자와 상호 작용하며 목표 인스턴스를 찾는 새로운 태스크입니다. 이는 사용자의 입력을 최소화하면서 탐색 과정을 동적으로 적응시켜 불확실성을 해결하는 접근법을 사용합니다.
- ***Technical Details***: 에이전트-사용자 상호작용과 함께 불확실성 인식(AIUTA)을 활용하며, Vision Language Model(VLM)과 Large Language Model(LLM)이 결합된 Self-Questioner 모듈을 사용하여 객체 인식을 위한 신뢰성 있는 설명을 제공합니다. 이 모듈은 시각적 관찰을 바탕으로 자동 생성한 질문을 통해 인식의 정확성을 높이며, Normalized-Entropy 기술로 불확실성을 평가합니다. Interaction Trigger 모듈은 탐색을 멈출지, 계속할지, 혹은 사용자의 추가 입력을 요구할지를 결정합니다.
- ***Performance Highlights***: AIUTA는 CoIN-Bench에서 기존 최첨단 방식과 비교하여 인스턴스 탐색에서 경쟁력 있는 성능을 보였습니다. 실제 사용자와의 실험에서 유연하게 다양한 사용자 입력을 처리할 수 있는 능력을 입증하였으며, Normalized-Entropy 기반 기술이 불확실성 처리에서 높은 신뢰도를 보여줍니다.

### [A Simple and Provable Scaling Law for the Test-Time Compute of Large Language Models](https://arxiv.org/abs/2411.19477)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2411.19477.png)

Vote: 3

Authors: Yaliang Li, Xuchen Pan, Yanxi Chen, Jingren Zhou, Bolin Ding

- ***What's New***: 이 논문은 대형 언어 모델(Large Language Models; LLMs)의 테스트 시간 계산 효율성을 위한 새로운 두 단계 알고리즘을 제안합니다. 이 알고리즘은 입력 문제에 대해 여러 후보 솔루션을 생성한 후, 토너먼트 형식으로 최종 솔루션을 선택하는 방식을 제안하는 최초의 접근입니다.
- ***Technical Details***: 제안된 알고리즘은 두 단계로 이루어져 있습니다. 첫 번째 단계에서는 입력 문제에 대해 N개의 후보 솔루션을 생성하며, 두 번째 단계에서는 각 후보를 쌍으로 비교하여 최종 솔루션을 결정합니다. 중요한 수학적 분석을 통해 이 알고리즘의 실패 확률은 N과 K 값에 따라 지수적으로 감소한다고 증명되었습니다.
- ***Performance Highlights***: MMLU-Pro 벤치마크에서 Llama3.1과 Qwen2.5 모델을 사용한 실험 결과, 테스트 시간 계산을 확장할수록 알고리즘의 정확도가 향상됨이 확인되었습니다. 특히 '수학'과 '공학' 분야에서 성능 향상이 두드러졌습니다.

### [VISTA: Enhancing Long-Duration and High-Resolution Video Understanding by Video Spatiotemporal Augmentation](https://arxiv.org/abs/2412.00927)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.00927.png)

Vote: 21

Authors: Jie Min, Huan Yang, Weiming Ren, Cong Wei, Wenhu Chen

- ***What's New***: VISTA는 기존 비디오-캡션 데이터셋을 기반으로 장시간 및 고해상도 비디오를 생성하는 비디오 시공간 증강(SpatioTemporal Augmentation) 프레임워크로 비디오 LMMs의 이해 능력을 향상시키기 위해 제안되었습니다. 새로운 합성 비디오와 질문-답변 쌍을 제공하여 이러한 장기 및 고해상도 비디오의 이해를 돕습니다.
- ***Technical Details***: VISTA는 주어진 비디오를 공간 및 시간적으로 결합하여 새로운 비디오 합성 샘플을 생성하고, 기존의 비디오-캡션 데이터셋에서 질문-답변(QA) 데이터를 생성합니다. 이를 통해 비디오-LLM(video-LMM) 모델들은 장시간 및 고해상도 비디오에서의 이해 능력을 훈련받게 됩니다. VISTA-400K 데이터셋과 HRVideoBench라는 고해상도 비디오 이해를 평가하는 벤치마크를 소개합니다.
- ***Performance Highlights***: VISTA-400K 데이터셋으로 다양한 비디오 LMM을 미세 조정한 결과, 평균 3.3% 성능 개선이 나타났으며 HRVideoBench에서는 6.5%의 성능 향상을 기록했습니다. 이는 VISTA의 프레임워크가 비디오 LMM의 성능을 향상시키는 데 매우 효과적임을 보여줍니다. 이 결과는 VISTA가 장기 및 고해상도 비디오 이해 능력을 강화하는 데 효과적임을 나타냅니다.

### [Open-Sora Plan: Open-Source Large Video Generation Model](https://arxiv.org/abs/2412.00131)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.00131.png)

Vote: 25

Authors: Shenghai Yuan, Bin Zhu, Xing Zhou, Junwu Zhang, Shaoling Dong, Lin Chen, Bin Lin, Cen Yan, Xiaoyi Dong, Zongjian Li, Zhang Pan, Li Yuan, Shaodong Wang, Yonghong Tian, Bin She, Yatian Pang, Xinhua Cheng, Yunyang Ge, Xianyi He, Yang Ye, Tanghui Jia, Liuhan Chen, Zhenyu Tang, Zhiheng Hu

- ***What's New***: Open-Sora Plan은 고성능 고해상도 비디오 생성 모델을 제공하기 위한 오픈 소스 프로젝트입니다. Wavelet-Flow Variational Autoencoder, Joint Image-Video Skiparse Denoiser, 조건 제어기와 같은 여러 구성 요소로 비디오 생성의 모든 과정을 다룹니다.
- ***Technical Details***: Open-Sora Plan는 주파수 영역에서 다중 레벨의 웨이블릿 변환을 통해 다중 스케일의 특징을 얻고 이를 피라미드 구조의 합성곱 백본에 주입하는 Wavelet-Flow Variational Autoencoder(WF-VAE)를 제안합니다. 또한, 3D Full Attention 구조로 변경된 Joint Image-Video Skiparse Denoiser와 다양한 조건 컨트롤러를 설계하여 여러 작업을 지원합니다.
- ***Performance Highlights***: Open-Sora Plan는 기존 모델보다 훨씬 적은 메모리 사용과 높은 처리량으로 고해상도 비디오를 높은 품질로 재구성합니다. VBench 및 ChronoMagic Bench와 같은 평가 기준에서 Open-Sora Plan은 타 모델들보다 심미적 품질, 부드러움 및 장면 복원 충실도를 포함한 여러 면에서 뛰어난 성능을 발휘합니다.

### [Steering Rectified Flow Models in the Vector Field for Controlled Image Generation](https://arxiv.org/abs/2412.00100)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.00100.png)

Vote: 15

Authors: Yezhou Yang, Song Wen, Maitreya Patel, Dimitris N. Metaxas

- ***What's New***: FlowChef는 벡터 필드(Vector Field) 속에서 정류된 흐름 모델(Rectified Flow Models; RFMs)을 활용하여 제어된 이미지 생성(Controlled Image Generation)을 위한 통합 프레임워크를 소개합니다. 이 접근법은 추가적인 훈련 없이도, 분류기 가이던스(Classifier Guidance), 선형 역문제(Linear Inverse Problems), 이미지 편집(Image Editing)을 함께 해결할 수 있습니다.
- ***Technical Details***: FlowChef는 연산 집약적인 ODE 솔버 또는 인버전과 같은 추가 비용 없이, 경사 건너뛰기(Gradient Skipping)를 활용하여 벡터 필드를 탐색하며 제어된 이미지 생성을 유도합니다. 이 방법은 레퍼런스 샘플과의 비용 함수를 최소화하며, 모델이 목표 샘플을 향하여 경로를 효율적으로 조정할 수 있도록 합니다.
- ***Performance Highlights***: FlowChef는 다양한 작업에서 기존 방법들보다 성능이 우수하며, 메모리 사용과 시간 요구 사항에서 새로운 최첨단 결과를 달성했습니다. 특히, 플럭스(Flux) 같은 대규모 모델에 대해 선형 역문제를 18초 안에 해결하고, 초대규모 파라미터의 모델에서도 모든 작업을 수행할 수 있습니다.

### [LSceneLLM: Enhancing Large 3D Scene Understanding Using Adaptive Visual Preferences](https://arxiv.org/abs/2412.01292)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.01292.png)

Vote: 7

Authors: Xinyu Sun, Tianhang Xiang, Yinjie Lei, Chuang Gan, Shuailei Ma, Hongyan Zhi, Junyan Li, Peihao Chen, Mingkui Tan

- ***What's New***: LSceneLLM은 대형 3D 씬 이해를 위한 적응형 프레임워크로, 다양한 작업의 시각적 선호도를 활용하여 과제를 자동으로 식별하고, 초점을 맞춘 영역에서 세밀한 세부 정보를 포착하는 'Scene Magnifier Module'을 소개합니다. 이 모듈은 대부분의 기존 3D-VLM(3D Vision-Language Models)에 쉽게 삽입되어 성능을 향상시킬 수 있습니다.
- ***Technical Details***: LSceneLLM은 장면의 거친 이해를 위해 다운샘플링된 포인트 클라우드를 인코딩하는 장면 인코더와, 주의 메커니즘을 사용하여 관심 영역에서 세밀한 시각 토큰을 선택하는 'Dense Token Selector'를 포함한 장면 확대 모듈로 구성되어 있습니다. 관심 영역의 세밀한 시각 정보를 통합하는 적응형 자기 주의 모듈을 통해, 제한된 포인트 클라우드 입력으로 대규모 장면 이해를 가능하게 합니다.
- ***Performance Highlights***: LSceneLLM은 큰 3D 장면 이해 벤치마크와 기존 장면 이해 벤치마크에서 기존 방법을 능가하는 성능을 보였습니다. 특히, XR-Scene이라는 크로스룸 이해 벤치마크에서 평균 132m2의 장면 영역을 처리하며, CIDEr, METEOR, ROUGE 등 다양한 평가 지표에서 우수한 성과를 기록하여, 첨단 성능을 입증했습니다.

### [X-Prompt: Towards Universal In-Context Image Generation in Auto-Regressive Vision Language Foundation Models](https://arxiv.org/abs/2412.01824)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.01824.png)

Vote: 53

Authors: Xiaoyi Dong, Dahua Lin, Zeyi Sun, Yuhang Zang, Jiaqi Wang, Pan Zhang, Tong Wu, Ziyang Chu, Yuanjun Xiong

- ***What's New***: X-Prompt는 대형 비전-언어 모델(Vision Language Models; VL-Models)에서 일반적인 이미지 생성 태스크를 위한 in-context 학습을 도입한 새로운 모델입니다. 이 모델은 보상 길이를 줄이고, 새로운 태스크에 대한 일반화 기능을 향상시킨 압축된 in-context 토큰을 사용하여 범용적인 이미지 생성을 목표로 합니다.
- ***Technical Details***: X-Prompt 모델은 순수하게 auto-regressive 방식을 사용하며, 긴 in-context 토큰 시퀀스를 효과적으로 처리할 수 있도록 하여 범용적인 이미지 생성을 지원합니다. 모델은 text와 image의 예측 태스크를 통합하여 in-context 예제의 정보 압축 기능을 통해 개선된 태스크 인식과 unseen 태스크에 대한 일반화 능력을 강화했습니다.
- ***Performance Highlights***: X-Prompt는 다양한 본 이미지 생성 태스크와 이전에 보지 못한 task에 대한 모델의 견고한 성능을 검증하는 광범위한 실험을 거쳤습니다. 실험은 주로 text-to-image 생성에서 상위 성능을 보여, 복잡한 멀티 오브젝트와 색 특성 테스트에서 특히 두드러진 향상을 보였습니다. 이러한 실험 결과는 X-Prompt가 한층 더 다채로운 태스크에 적응할 수 있는 역량을 향상시켰음을 확인시켜 줍니다.

### [World-consistent Video Diffusion with Explicit 3D Modeling](https://arxiv.org/abs/2412.01821)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.01821.png)

Vote: 4

Authors: Joshua Susskind, Shuangfei Zhai, Qihang Zhang, Jiatao Gu, Kevin Miao, Alexander Toshev, Miguel Angel Bautista

- ***What's New***: 이 논문은 영상 생성 분야에서 3D 일관성을 효율적으로 구현하기 위한 'World-consistent Video Diffusion (WVD)'라는 새로운 프레임워크를 소개합니다. WVD는 각 이미지 픽셀의 전역 3D 좌표를 인코딩하는 XYZ 이미지를 사용하여 명시적인 3D 감독(3D supervision)을 포함하는 것이 특징입니다. 이를 통해 단일 이미지에서 3D 생성, 다중 뷰 스테레오, 카메라 제어 비디오 생성과 같은 다양한 3D 작업을 통합할 수 있습니다.
- ***Technical Details***: WVD는 RGB 및 XYZ 프레임의 결합 분포를 학습하는 확산 변환기(Diffusion Transformer)를 훈련합니다. XYZ 이미지는 3D 변환 정보만 캡처하여 훈련 중 명시적인 3D 감독을 제공하는 반면, 기존 이미지 아키텍처와 호환됩니다. 이를 기반으로 WVD는 조건부 생성 및 다기능 적응을 지원하며, 이를 통해 3D 일관성을 유지하면서도 확장 가능한 3D 일관 비디오 및 이미지 생성을 가능하게 합니다.
- ***Performance Highlights***: WVD는 다양한 벤치마크에서 경쟁력 있는 성능을 보였으며, 이는 단일 사전 훈련된 모델을 통해 3D 일관성을 유지하면서 확장 가능한 3D 비디오 및 이미지 생성의 잠재력을 입증합니다. 또한 카메라 제어 비디오 생성에서 WVD는 실제 비디오의 카메라 경로를 효과적으로 복제할 수 있음을 보여주었습니다.

### [TinyFusion: Diffusion Transformers Learned Shallow](https://arxiv.org/abs/2412.01199)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.01199.png)

Vote: 13

Authors: Xinchao Wang, Kunjun Li, Xinyin Ma, Gongfan Fang

- ***What's New***: TinyFusion은 디퓨전 트랜스포머(Diffusion Transformers)의 심층 레이어를 깊이 프루닝(Depth Pruning)하는 새로운 방법을 제안합니다. 이 방법은 학습 가능한 샘플링 기법을 도입하여 프루닝을 최적화하게 하여, 잘라낸 모델이 미세 조정(Fine-Tuning) 후에도 높은 성능을 유지할 수 있게 설계되었습니다.
- ***Technical Details***: TinyFusion은 레이어 프루닝과 미세 조정을 통합하여 최적화하는 방법으로, 각각의 레이어에 대해 샘플링 마스크를 차등적(Differentiable)으로 샘플링합니다. 중요한 함수는 미세 조정 후 성능 회복 가능성을 최대화하는 것이며, 이를 위해 합동 최적화(Co-Optimized)의 가중치 업데이트가 포함되었습니다. 이 방법은 DiTs, MARs, SiTs와 같은 다양한 아키텍처에 일반화됩니다.
- ***Performance Highlights***: 디트-XL(DiT-XL) 모델 실험에서 TinyFusion을 사용하면 프루닝 후 약 7%의 사전 훈련 비용으로 2배의 속도 향상과 함께 2.86의 FID 점수를 달성합니다. 이는 기존 방법보다 경쟁력 있는 성능을 나타내며, 다양한 아키텍처에서 일반화되고 높은 성능 회복성을 보여줍니다.

### [Motion Prompting: Controlling Video Generation with Motion Trajectories](https://arxiv.org/abs/2412.02700)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.02700.png)

Vote: 5

Authors: Andrew Owens, Carl Doersch, Forrester Cole, Daniel Geng, Tatiana Lopez-Guevara, Yusuf Aytar, Charles Herrmann, Chen Sun, Oliver Wang, Deqing Sun, Michael Rubinstein, Junhwa Hur, Serena Zhang, Tobias Pfaff

- ***What's New***: 이 연구는 비디오 생성에서 텍스트 프롬프트 대신 '모션 프롬프트(Motion Prompts)'를 사용하여 동작 궤적을 제어하는 혁신적인 방법을 제안합니다. 이는 비디오 생성 과정에서 보다 구체적이고 유연한 동작 제어를 가능하게 하여 현실적인 물리적 행동과 상호작용을 탐색할 수 있는 잠재력을 시사합니다.
- ***Technical Details***: 제안된 방법에서는 스파티오-템포럴(Motion Trajectory) 궤적을 특징으로 갖는 모션 프롬프트를 사용하여, 세밀하거나 희박한 동작 궤적을 생성해 정의된 ControlNet 모델을 학습시킵니다. 이는 비디오 디퓨전 모델(Video Diffusion Model) 상에서 훈련되어 객체 및 카메라 제어, 이미지 편집 등의 다양한 응용이 가능합니다.
- ***Performance Highlights***: 제시된 방법은 DAVIS 데이터셋의 검증 세트를 기준으로 QT 성능 평가에서 주요 베이스라인을 상회하는 결과를 보입니다. 또한, 사람의 평가에서도 모션 일치도와 비주얼 품질에서 경쟁 모델 대비 높은 선호도를 보였습니다. 이는 다방면의 동작 제어를 효과적으로 수행할 수 있음을 나타냅니다.

### [o1-Coder: an o1 Replication for Coding](https://arxiv.org/abs/2412.00154)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.00154.png)

Vote: 25

Authors: Yuxiang Zhang, Jitao Sang, Yuqi Yang, Chao Kong, Jinlin Xiao, Shangxi Wu, Jiangming Shu

- ***What's New***: O1-CODER는 OpenAI의 o1 모델을 코딩 작업에 중점을 두어 복제하는 시도입니다. 강화학습(RL)과 몬테카를로 트리 탐색(MCTS)을 통합하여 모델의 시스템-2 사고 능력(System-2 thinking)을 강화합니다. 표준화된 코드 테스트를 위한 테스트 케이스 생성기(TCG)를 훈련하고, MCTS를 사용하여 추론 프로세스로 코드 데이터를 생성하며, 정책 모델을 반복적으로 미세 조정하여 초기에는 의사코드(pseudocode)를 생성하고 그 후 전체 코드를 생성하는 프레임워크가 포함됩니다.
- ***Technical Details***: O1-CODER 프레임워크는 코드 생성에 적용되는 자기 지도 RL을 통해 두 가지 주요 과제를 해결합니다. 첫 번째는 테스트 케이스 생성기(TCG)를 훈련하여 코드 품질을 평가하는 것이며, 이를 통해 RL의 결과 보상을 제공합니다. 두 번째는 의사코드를 통해 추론 과정을 설계하고 정책 모델을 가이드하여 최종 실행 가능한 코드를 생성하는 것입니다. 또한 RL 및 MCTS를 통한 정책 모델의 지속적인 업데이트도 이루어져 모델 개선을 촉진합니다.
- ***Performance Highlights***: 의사코드를 활용한 코드 생성 실험에서, 일반 Chain-of-Thought(CoT)와 pseudocode 기반 CoT 간의 Pass@1 성능은 유사하게 나타났으나, Average Sampling Pass Rate(ASPR)에서는 pseudocode 방법이 더 높은 성공률을 보였습니다. 이는 pseudocode가 올바른 추론 경로를 중요한데 기여하는 것을 시사합니다.

### [HUGSIM: A Real-Time, Photo-Realistic and Closed-Loop Simulator for Autonomous Driving](https://arxiv.org/abs/2412.01718)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.01718.png)

Vote: 2

Authors: Yiyi Liao, Yichong Lu, Longzhong Lin, Dongfeng Bai, Jiabao Wang, Andreas Geiger, Hongyu Zhou, Bingbing Liu, Yue Wang

- ***What's New***: HUGSIM은 자율 주행 알고리즘의 평가를 위한 실시간, 사진 실사적, 폐쇄형 루프 시뮬레이터로 개발되었습니다. 3D Gaussian Splatting을 통해 2D RGB 이미지를 3D 공간으로 전환하여 렌더링 품질을 향상시키고, 폐쇄형 루프 환경을 구축합니다.
- ***Technical Details***: 이 시뮬레이터는 KITTI-360, Waymo, nuScenes, PandaSet 데이터셋에서 추출한 70개 이상의 시퀀스와 400개의 다양한 시나리오를 통해 자율주행 알고리즘을 평가할 수 있는 포괄적인 벤치마크를 제공합니다. 시뮬레이터는 물리적 제약을 적용하여 동적 차량의 경로를 규일화하고, 여러 평면 지상 모델을 통해 차선 왜곡 문제를 해결합니다.
- ***Performance Highlights***: HUGSIM은 기존의 3D Gaussian Splatting 방식에 비해 실시간 성능을 유지하며, 학습된 가우시안의 수를 줄이면서도 품질 높은 지상 렌더링을 구현합니다. 또한 위기 시나리오를 시뮬레이션하여 알고리즘의 평가를 지원합니다.

### [VideoGen-of-Thought: A Collaborative Framework for Multi-Shot Video Generation](https://arxiv.org/abs/2412.02259)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.02259.png)

Vote: 43

Authors: Qifeng Chen, Wenjie Shu, Haojian Huang, Feilong Tang, Harry Yang, Mingzhe Zheng, Xuran Ma, Yongqi Xu, Yatian Pang, Ser-Nam Lim, Yexin Liu

- ***What's New***: VideoGen-of-Thought (VGoT)는 멀티샷 비디오 생성(multi-shot video generation)을 위한 훈련 필요 없는 협업적 프레임워크입니다. 기존 방법들이 논리적 이야기 전개나 비주얼 일관성을 유지하는 데 어려움을 겪는 반면, VGoT는 이러한 문제를 구조화된 모듈식 접근을 통해 효과적으로 해결합니다.
- ***Technical Details***: VGoT는 네 개의 모듈로 구성되어 있습니다: 시나리오 생성 모듈(Script Generation Module)은 LLM(Large Language Model)을 사용하여 간단한 스토리를 각 샷에 대한 자세한 프롬프트로 변환합니다. 키프레임 생성 모듈(Keyframe Generation Module)은 시나리오를 기반으로 일관된 키프레임을 생성합니다. 샷-레벨 비디오 생성 모듈(Shot-Level Video Generation Module)은 시나리오와 키프레임을 결합하여 각 샷의 비디오 잠재 변수를 생성합니다. 마지막으로, 부드러움 모듈(Smooth Module)은 샷 간 부드러운 전환을 보장하여 전체적인 비디오 내러티브의 일관성을 유지합니다.
- ***Performance Highlights***: VGoT는 기존의 비디오 생성 기법보다 높은 품질의 일관된 멀티샷 비디오를 생성하는 데 성공했습니다. 실험 결과는 이야기의 전개와 일관성 측면에서 VGoT가 시각적 품질과 횡적-일관성 측면에서 우수한 성능을 보여줍니다. 이러한 성과는 VGoT가 멀티샷 비디오 생성의 새로운 패러다임을 제시한다는 점을 부각시킵니다.

### [MaskRIS: Semantic Distortion-aware Data Augmentation for Referring Image Segmentation](https://arxiv.org/abs/2411.19067)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2411.19067.png)

Vote: 5

Authors: Minhyun Lee, Hyunjung Shim, Dongyoon Han, Byeongho Heo, Song Park, Seungho Lee

- ***What's New***: MaskRIS는 지칭 이미지 세분화(Referring Image Segmentation; RIS)을 위한 새로운 데이터 증강 및 학습 프레임워크로, 기존 데이터 증강이 RIS에 부적합하다는 점을 해결하고 보다 효율적인 학습을 지원합니다. MaskRIS는 이미지와 텍스트 마스킹을 활용하여 모델의 강인성을 높이며, RefCOCO, RefCOCO+, RefCOCOg 데이터셋에서 새로운 최첨단 성능을 달성했습니다.
- ***Technical Details***: MaskRIS는 입력 마스킹(input masking)을 데이터 증강 기법으로 사용하고, 왜곡 인식 학습(Distortion-aware Contextual Learning; DCL)을 통해 마스킹 전략의 이점을 극대화합니다. 이미지와 텍스트에 각각 마스킹을 도입하여 모델이 불완전한 정보와 다양한 언어적 복잡성을 처리할 수 있도록 돕습니다. DCL은 원본 입력과 마스킹된 입력을 처리하는 두 경로로 구성되어 데이터 다양성을 높이고 예측 일관성을 촉진합니다.
- ***Performance Highlights***: MaskRIS는 기존 RIS 방법을 능가하며, 특히 RefCOCO, RefCOCO+, RefCOCOg 데이터셋에서 새로운 성능 기록을 세웠습니다. 다양한 이미지와 언어적 복잡성에 대한 강력한 성능을 입증했으며, 기존 방법에 비해 증가된 데이터 다양성에 따른 성능 향상과 강인성을 보였습니다.

### [Switti: Designing Scale-Wise Transformers for Text-to-Image Synthesis](https://arxiv.org/abs/2412.01819)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.01819.png)

Vote: 19

Authors: Dmitry Baranchuk, Denis Kuznedelev, Mikhail Khoroshikh, Anton Voronov, Valentin Khrulkov

- ***What's New***: SWITTI는 새로운 규모-구조적(text-to-image) 변환기(transformer)로 텍스트에서 이미지로 변환하는 과정을 개선하였습니다. 기존의 다음-규모 예측 AR 모델(next-scale prediction AR models)을 기반으로 하여 수렴성과 전반적인 성능을 향상시키는 건축적 수정 사항을 제안하였습니다. 이러한 변경을 통해 샘플링 속도를 약 11% 증가시키고 메모리 사용량을 줄일 수 있는 비-AR(counterpart) 모델을 도입하였습니다.
- ***Technical Details***: SWITTI는 다음-규모 예측 AR 모델을 텍스트에서 이미지로의 변환에 사용하며, 모델의 자기-주목 맵(self-attention maps)은 이전 규모에 대한 의존도가 약함을 보여줍니다. 이러한 인사이트에 기초하여, 비-AR 카운터파트를 제안하여 더 빠른 샘플링을 구현하였습니다. 또한, 고해상도에서는 클래스-프리 안내(classifier-free guidance; CFG)가 불필요하다는 점을 발견하고 이를 비활성화하여 샘플링 시간을 추가로 약 20% 가속시켰습니다.
- ***Performance Highlights***: 실험 결과에 따르면 SWITTI는 기존의 텍스트에서 이미지로 변환하는 AR 모델을 능가하며, 최첨단 텍스트에서 이미지로의 확산 모델과 경쟁하면서도 최대 7배 더 빠른 속도를 보여줍니다. SWITTI는 인간 선호 연구와 자동화된 평가에서 두드러진 성능 강점을 보여줬습니다.

### [Art-Free Generative Models: Art Creation Without Graphic Art Knowledge](https://arxiv.org/abs/2412.00176)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.00176.png)

Vote: 6

Authors: Antonio Torralba, Rohit Gandikota, David Bau, Joanna Materzynska, Hui Ren

- ***What's New***: 이 연구는 그래픽 아트 지식을 사용하지 않고도 예술을 생성할 수 있는 새로운 방법인 Art-Free Generative Models를 소개합니다. Art-Free로 명명된 데이터셋에서 훈련된 텍스트-이미지 생성 모델이 소수의 예술작품 예제로도 특정 예술 스타일을 학습하고 재현할 수 있음을 검증하였습니다. 이는 대규모 예술 데이터셋 없이도 예술 스타일을 효과적으로 모방할 수 있음을 보여줍니다.
- ***Technical Details***: Art-Free 모델은 그래픽 아트 요소를 배제하고 자연 이미지를 중심으로 구축된 'Art-Free SAM' 데이터셋에서 훈련되었습니다. Art-Free Diffusion 모델은 LoRA(저순위 어댑터)를 활용하여 소수의 예술 스타일 예제로부터 스타일 정보를 학습하며, '스타일 손실'과 '콘텐츠 손실'을 통해 스타일과 콘텐츠를 분리하여 학습합니다.
- ***Performance Highlights***: Art-Free Diffusion 모델은 훈련 데이터의 예술적 요소를 배제했음에도 불구하고, 유명 예술가 스타일을 성공적으로 모방하여 사용자 평가 연구에서 높은 스타일 충실도를 보였습니다. 특히, 사용자들은 Art-Free Diffusion의 결과를 기존 대규모 아트 데이터셋을 사용한 StyleAligned 방법과 비교하여 더 높은 스타일일치도를 가지고 평가하였습니다.

### [VLsI: Verbalized Layers-to-Interactions from Large to Small Vision Language Models](https://arxiv.org/abs/2412.01822)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.01822.png)

Vote: 10

Authors: Yong Man Ro, Yueh-Hua Wu, Byung-Kwan Lee, Ryo Hachiuma, Yu-Chiang Frank Wang

- ***What's New***: VLsI는 대형 비전 언어 모델에서 소형 비전 언어 모델로의 지식 전달을 자연어 기반 증류(natural language-based distillation) 기법을 통해 효율적으로 수행하는 새로운 비전 언어 모델(VLM) 군입니다. 2B와 7B 모델 크기로 제공되며, 정확성을 유지하면서 효율성을 높이는데 중점을 두고 있습니다.
- ***Technical Details***: VLsI는 층 계통 증류(layer-wise distillation) 과정을 활용하여 중간 'verbalizers'를 도입하여 각 계층의 특징을 자연어 공간으로 변환합니다. 이를 통해 소형 VLM이 대형 VLM의 추론 과정을 유연하게 맞추도록 설계되었습니다. 세 단계로 진행됩니다: 1) 중간 특징을 텍스트 기반 응답으로 처리하여 해석 가능하게 만드는 verbalization 단계, 2) 대형 및 소형 VLM의 계층별 추론 진행을 맞추는 interaction 단계, 3) 작업별 지침-준수 반응성을 강화하는 reinforcement 단계.
- ***Performance Highlights***: VLsI는 GPT-4V보다 2B 모델에서 11.0%, 7B 모델에서 17.4%의 성능 향상을 입증하였으며, 모델 크기 확장이나 구조 변경 없이 이를 달성하였습니다. 오픈 소스 VLM과의 비교 실험에서도 우수한 성능을 나타내며, 미래 연구의 방향성을 제시합니다.

### [AV-Odyssey Bench: Can Your Multimodal LLMs Really Understand Audio-Visual Information?](https://arxiv.org/abs/2412.02611)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.02611.png)

Vote: 17

Authors: Yutong Bai, Shijia Yang, Bohao Li, Kaituo Feng, Zhuoran Yang, Kaixiong Gong, Yibing Wang, Xiangyu Yue, Jiaming Han, Benyou Wang, Mofan Cheng

- ***What's New***: AV-Odyssey Bench는 멀티모달 대형 언어 모델(MLLMs)이 진정으로 오디오-비주얼 정보를 이해할 수 있는지를 평가하기 위해 새롭게 제안된 포괄적인 벤치마크입니다. 기존의 듣기 능력 및 다양한 오디오 속성을 제대로 평가하지 못한 기존 벤치마크의 한계를 극복하고자 설계되었습니다.
- ***Technical Details***: AV-Odyssey Bench는 4,555개의 텍스트, 시각 및 오디오 요소로 구성된 문제들을 포함합니다. 모델이 텍스트, 이미지/영상, 오디오 클립을 통합하여 정확한 답변을 도출하도록 하여 멀티모달 정보 통합 능력을 평가합니다. 또한 DeafTest를 통해 모델의 기본적인 듣기 능력을 평가하며, 문제는 주관적 평가 없이도 평가가 가능하도록 객관식으로 구성되었습니다.
- ***Performance Highlights***: 대부분의 MLLMs는 AV-Odyssey 벤치마크에서 랜덤 성능과 유사한 결과를 보였으며, GPT-4o 오디오 캡션 방식이 가장 높은 34.5%의 성능을 기록했습니다. 이 벤치마크는 기존 모델들이 복잡한 오디오-비주얼 정보 통합 작업에서 여전히 한계를 가지고 있음을 시사하며 차후 연구 방안을 제시합니다.

### [Scaling Image Tokenizers with Grouped Spherical Quantization](https://arxiv.org/abs/2412.02632)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.02632.png)

Vote: 9

Authors: Vincent Tao Hu, Zhen Qin, Yifan Zhang, Rania Briq, Stefan Kesselheim, Björn Ommer, Jiangtao Wang

- ***What's New***: 새로운 그룹형 구면 양자화(Grouped Spherical Quantization; GSQ) 기법을 통해 이미지 토크나이저의 확장성을 향상시켰습니다. GSQ는 초구 초기화 및 조회 정규화를 특징으로 하며, 우수한 복원 품질을 달성하면서도 적은 훈련 반복 횟수를 요구합니다. 이 방법은 고차원 잠재 공간을 저차원 공간으로 재구조화하여 효율적인 확장을 가능하게 합니다.
- ***Technical Details***: GSQ는 잠재적 차원, 코드북 크기 및 압축 비율에 대한 체계적 Scaling을 통해 모델 성능에 미치는 영향을 분석합니다. 고차원 잠재 공간을 효과적으로 관리하기 위해, GSQ는 잠재 공간을 그룹화하여 저차원 공간으로 나누는 방식을 사용합니다. 이는 초구 초기화와 ℓ2 정규화 동안 코드북 항목을 동일하게 유지하여 코드북 사용률을 100% 가깝게 유지합니다.
- ***Performance Highlights***: GSQ-GAN은 16배 다운샘플링에서 0.50의 재구성 FID(rFID) 값을 기록했습니다. 이는 최첨단 방법들보다 우수한 결과로, 20개의 훈련 에포크 내에 달성되었습니다. 또한, 더 커진 배치 사이즈와 학습률로 트레이닝 속도와 모델의 수렴 속도는 더욱 개선되었습니다.

### [WF-VAE: Enhancing Video VAE by Wavelet-Driven Energy Flow for Latent Video Diffusion Model](https://arxiv.org/abs/2411.17459)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2411.17459.png)

Vote: 10

Authors: Shenghai Yuan, Zongjian Li, Yang Ye, Li Yuan, Xinhua Cheng, Liuhan Chen, Bin Lin

- ***What's New***: 이 논문은 다중 레벨의 웨이블릿 변환(Wavelet Transform)을 활용한 WF-VAE를 소개하여, 저주파수 비디오 정보를 잠재 표현 공간에 효율적으로 인코딩하는 새로운 방법을 제안합니다. 또한 Causal Cache라는 손실 없는 블록-와이즈 추론(Block-wise Inference) 메커니즘을 도입하여, 기존 타일링 전략으로 인한 비디오 플리커링 문제를 완전히 해결했습니다.
- ***Technical Details***: WF-VAE는 영상 신호를 다중 레벨의 웨이블릿 변환을 통해 여러 주파수-도메인 성분으로 분해하며, 저주파수 성분을 주 에너지 흐름 경로를 통해 백본을 우회하여 잠재표현으로 흐르도록 설계되었습니다. 이러한 에너지 흐름 경로는 저주파수 정보를 강조하여, 보다 높은 압축률을 높은 주파수 정보에 적용합니다. Causal Cache는 인과적 3D 합성곱의 속성을 활용하여 블록-와이즈 및 직접 추론 결과가 동일하게 유지되도록 보장합니다.
- ***Performance Highlights***: WF-VAE는 다양한 벤치마크 데이터셋(WebVid-10M, Panda-70M)에서 최신 상태의 비디오 리컨스트럭션 성능을 달성하며, 경쟁 모델 대비 낮은 계산 비용을 유지합니다. 특히, WF-VAE-S는 OD-VAE와 같은 인기있는 비디오 VAE를 능가하며, WF-VAE-L은 PSNR, LPIPS 및 FVD 메트릭에서 Allegro를 뛰어넘었습니다. 추가적으로, WF-VAE-L은 IS 및 FVD 측정에서 최고의 결과를 기록하며 비디오 생성 평가에서 우수한 성능을 보였습니다.

### [PhysGame: Uncovering Physical Commonsense Violations in Gameplay Videos](https://arxiv.org/abs/2412.01800)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.01800.png)

Vote: 5

Authors: Ruyang Liu, Ge Zhang, Ian Reid, Meng Cao, Haoze Zhao, Qiang Sun, Jiaheng Liu, Hangyu Guo, Haoran Tang, Xiaodan Liang

- ***What's New***: PhysGame은 게임 플레이 비디오에서 물리적 상식의 위반을 평가하는 선구적인 벤치마크입니다. 이 벤치마크는 물리적 상식 이해에서 비디오 기반 LLMs의 탁월한 능력을 평가하기 위해 설계되었으며, 네 가지 주요 영역(역학, 운동학, 광학, 재료 특성)에 걸친 12가지 세부적인 물리적 상식을 다룹니다.
- ***Technical Details***: PhysGame 벤치마크는 880개의 비디오로 구성되어 있으며, 각각 물리적 상식 위반에 대한 고품질의 다지선다형 질문이 주어집니다. 이 데이터는 주로 Reddit에서 수집되고, YouTube에서 키워드 검색을 통해 보강됩니다. 또한, 물리적 상식 학습을 촉진하기 위해 140,057개의 질의응답 쌍으로 이루어진 PhysInstruct 데이터셋과 34,358개의 학습 쌍으로 구성된 PhysDPO 데이터셋을 제공합니다. 이 두 데이터셋은 물리적 상식의 이해 능력을 향상시키기 위한 다양한 훈련을 지원합니다.
- ***Performance Highlights***: PhysVLM은 물리적 이해 능력에서 최고 수준의 성능을 보여주며, 물리적 상식 이해에서 오픈 소스 모델들이 독점 모델들에 비해 성능이 현저히 뒤처진다는 기존의 연구 결과에 도전합니다. 특히 Video-MME와 VCG 등의 일반 비디오 이해 벤치마크에서도 우수한 성능을 입증했습니다.

### [VLSBench: Unveiling Visual Leakage in Multimodal Safety](https://arxiv.org/abs/2411.19939)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2411.19939.png)

Vote: 7

Authors: Xuhao Hu, Jing Shao, Xuanjing Huang, Hao Li, Dongrui Liu

- ***What's New***: VLSBench는 멀티모달 대형 언어 모델(Multimodal Large Language Models; MLLMs)에서 시각적 안전 정보 누출(Visual Safety Information Leakage; VSIL)의 문제를 해결하기 위해 설계된 새로운 멀티모달 시각 누출 없는 안전 벤치마크입니다. 이 벤치마크는 이미지-텍스트 쌍에서 시각적 안전 정보가 텍스트 쿼리에 드러나는 것을 방지하며, 2.4K 이미지-텍스트 쌍을 포함합니다.
- ***Technical Details***: VLSBench는 GPT-4o와 같은 모델을 활용하여 유해한 텍스트 쿼리를 수정하고, 이미지 설명에서 Start-Diffusion-3.5-Large 모델을 사용하여 고품질 이미지를 생성하며, 유해한 이미지-텍스트 쌍을 걸러냅니다. 각 샘플은 시각적 안전 정보가 누출되지 않도록 특별히 디자인되었습니다. 실험에서, 이 벤치마크는 LLaVA, Qwen2-VL, Llama3.2-Vision, GPT-4o를 포함한 여러 MLLM들에게 중요한 도전을 제기합니다.
- ***Performance Highlights***: VLSBench는 현재의 오픈소스 및 독점 MLLMs에게 상당한 도전을 제기하며, 멀티모달 정렬 방법들은 VSIL 없는 VLSBench 벤치마크에서 텍스트 정렬 방법보다 더 나은 성능을 보여줍니다. 특히, 멀티모달 안전성이 강력한 모델들도 50% 미만의 안전률을 기록하여, 전반적으로 강력한 멀티모달 정렬 방법이 필요하다는 것을 시사합니다.

### [Exploring the Abilities of Large Language Models to Solve Proportional Analogies via Knowledge-Enhanced Prompting](https://arxiv.org/abs/2412.00869)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.00869.png)

Vote: 2

Authors: Aman Chadha, Ponnurangam Kumaraguru, Vinija Jain, Ruwan Wickramarachchi, Amitava Das, Sreeram Vennam, Thilini Wijesiriwardene, Amit Sheth

- ***What's New***: 이 연구는 새로운 15K 다지선다형 질문 응답(MCQA) 데이터셋을 도입하여, 비율적 유추(Proportional Analogies) 문제 해결에서 현대의 대형 언어 모델들(LLMs)의 성능을 평가합니다. 특히, 3가지 유형의 지식(Exemplar, Structured, Targeted)을 프롬프트에 추가하여 유추 문제 해결 능력을 분석한 것이 특징입니다.
- ***Technical Details***: 본 연구는 15K 데이터셋에서 평가하며, Zero-shot, Few-shot, Structured Knowledge Prompting(SKP), Targeted Knowledge Prompting(TKP) 등의 네 가지 프롬프트 기법을 활용합니다. Zero-shot은 추가 지식 없이 문제만을 제공하며, Few-shot은 예제 지식을, SKP는 WordNet, ConceptNet, Wikidata에서 구조화된 지식을 이용합니다. TKP는 문제 용어 쌍의 특정 의미론적 관계를 통해 타겟 지식을 제공합니다.
- ***Performance Highlights***: Nine GenAI 모델들을 평가한 결과, GPT-3.5-Turbo 모델이 TKP에서 55%의 정확도로 최고 성능을 보였습니다. 그러나 Zero-shot 프롬프트와 비교했을 때, SKP를 활용한 경우 성능이 하락하는 경향을 보였습니다. 이는 다양한 지식을 추가하더라도 비율적 유추 문제 해결에 항상 도움이 되지 않음을 시사합니다.

### [Efficient Track Anything](https://arxiv.org/abs/2411.18933)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2411.18933.png)

Vote: 12

Authors: Saksham Suri, Balakrishnan Varadarajan, Chong Zhou, Forrest Iandola, Bilge Soran, Ramya Akula, Raghuraman Krishnamoorthi, Yunyang Xiong, Chenchen Zhu, Xiaoyu Xiang, Vikas Chandra, Zechun Liu, Lemeng Wu

- ***What's New***: Efficient Track Anything(EfficientTAM) 모델은 기존 Segment Anything Model 2(SAM 2)와 비교해 모바일 기기에서도 효과적으로 작동할 수 있도록 설계된 경량화된 영상 객체 추적 및 세분화 모델입니다. 이 모델은 비선형 비계층적 Vision Transformer(ViT)를 이미지 인코더로 채택해 복잡성을 줄이고, 효율적인 메모리 모듈을 도입하여 비디오 객체 세분화에서 높은 성능을 유지하면서도 빠른 속도를 자랑합니다.
- ***Technical Details***: EfficientTAM은 ViT-Small/ViT-Tiny와 같은 경량의 비계층적 이미지 인코더를 사용하여 프레임 특성을 추출하며, 강한 메모리 공간 토큰의 지역성을 활용한 효율적인 크로스 어텐션(cross-attention)을 통해 메모리 모듈의 연산과 메모리 비용을 줄였습니다. 이를 통해 비디오 객체 세분화와 트랙킹 작업에서 성능-효율성 간의 우수한 균형을 이뤄냈습니다.
- ***Performance Highlights***: EfficientTAM은 SAM 2과 비교하여 비디오 객체 세분화에서 약 2배 빠른 속도를 보여주며, 약 2.4배 적은 파라미터 수를 가집니다. A100 GPU에서의 성능 측정 결과와 아이폰 15 Pro Max와 같은 모바일 장치에서 초당 10 프레임의 비디오 세분화를 수행할 수 있는 품질을 보여, 모바일 환경에 적합한 효율적인 대용량 비디오 처리의 가능성을 제시합니다.

### [AMO Sampler: Enhancing Text Rendering with Overshooting](https://arxiv.org/abs/2411.19415)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2411.19415.png)

Vote: 2

Authors: Bo Liu, Hongliang Fei, Qiang Liu, Xixi Hu, Keyang Xu

- ***What's New***: AMO Sampler는 기존의 텍스트-이미지 생성 모델에서 특히 텍스트 렌더링의 정확성을 극대화하면서도 추가적인 학습 없이 구현될 수 있는 혁신적인 방법입니다. Attention Modulated Overshooting(AMO) 샘플러는 오일러 샘플러에 비해 적은 계산 비용으로 텍스트 렌더링을 크게 개선합니다.
- ***Technical Details***: AMO 샘플러는 Rectified Flow(RF) 모델에서 overshooting과 노이즈를 번갈아가며 적용하여 확률론적 샘플링을 도입합니다. 이는 오일러 샘플러의 누적 오차를 보완하고 텍스트 생성 품질을 향상시킵니다. 또한, 각 이미지 패치의 attention score에 따라 overshooting의 강도를 조절하는 기법을 제안하여, 각 패치가 텍스트 내용과 얼마나 관련이 있는지를 기반으로 overshooting을 조절합니다.
- ***Performance Highlights***: AMO 샘플러는 SD3와 Flux와 같은 최첨단 RF 기반 모델에서 텍스트 렌더링 정확도를 각각 32.3% 및 35.9% 향상시키며, 전체 이미지 품질을 유지하면서도 생성 정확도를 크게 개선하였습니다. 이는 오일러 샘플러에 비해 더 나은 텍스트 렌더링 성능을 보여줍니다. 인퍼런스 비용을 증가시키지 않고도 이러한 성능 개선을 달성하였습니다.

### [Long Video Diffusion Generation with Segmented Cross-Attention and Content-Rich Video Data Curation](https://arxiv.org/abs/2412.01316)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.01316.png)

Vote: 7

Authors: Wenhao Huang, Xin Yan, Qiuyue Wang, Yuan Zhou, Huan Yang, Yuxuan Cai

- ***What's New***: 이 논문에서는 Presto라는 새로운 비디오 확산 모델(Video Diffusion Model)을 소개합니다. Presto는 15초 동안의 장시간 비디오를 생성하며, 장기적인 일관성과 풍부한 콘텐츠를 유지합니다. Segmented Cross-Attention (SCA)이라는 방법을 제안하여, 숨겨진 상태를 시간 차원에서 세그먼트로 나누고 각 세그먼트를 해당하는 서브 캡션(Sub-Caption)과 연결합니다. 이 접근법은 추가 파라미터가 필요 없으며, 기존의 DiT 기반 아키텍처에 쉽게 통합될 수 있습니다.
- ***Technical Details***: Presto를 위해 고품질의 장시간 비디오 생성을 도울 데이터셋인 LongTake-HD를 구축하였으며, 261,000개의 콘텐츠가 풍부한 비디오로 구성되어 있습니다. 또한, 비디오 생성 모델의 성능 향상을 위해 Segmented Cross-Attention 메커니즘을 적용하였습니다. 이는 비디오를 여러 시나리오로 나누고 진행적인 서브 캡션을 생성하여 텍스트 정보의 풍부함과 비디오의 장기간 일관성을 동시에 달성합니다.
- ***Performance Highlights***: Presto는 VBench에서 Semantic Score 78.5%와 Dynamic Degree 100%를 기록하여 기존의 최첨단 비디오 생성 방법들을 능가하였습니다. 또한, 사용자가 실시한 연구에서는 Presto가 시나리오 다양성, 시나리오 일관성, 텍스트-비디오 정렬에서 우수성을 나타냈습니다.

### [MALT: Improving Reasoning with Multi-Agent LLM Training](https://arxiv.org/abs/2412.01928)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.01928.png)

Vote: 19

Authors: Rocktim Jyoti Das, Christian Schroeder de Witt, Ivan Laptev, Sumeet Ramesh Motwani, Ronald Clark, Chandler Smith, Fabio Pizzati, Markian Rybchuk, Philip H. S. Torr

- ***What's New***: MALT는 대형 언어 모델(Large Language Models; LLMs)의 협업 역량을 강화하기 위해 다중 에이전트 학습(Multi-Agent Learning)을 통한 새로운 추론 방법을 제안합니다. 이 연구는 LLM들이 문제 해결에 있어 서로 다른 역할(생성자, 검증자, 개선자)을 수행하며 문제를 점진적으로 해결하도록 훈련하는 초기 시도를 보여줍니다.
- ***Technical Details***: MALT 접근 방식은 문제 해결을 위해 생성자(generator), 검증자(verifier), 개선자(refiner)로 이뤄진 이질적인 다중 에이전트 설정을 사용합니다. 각 에이전트는 역할에 맞는 특화된 데이터를 통해 학습하며, 신뢰도 기반 보상과 경로 확장을 이용한 합성 데이터 생성 및 신용 할당 전략을 채택하여 모델의 전문성을 향상시킵니다.
- ***Performance Highlights***: MALT를 Llama 3.1 8B 모델에 적용한 결과, MATH, GSM8k, CSQA 등의 데이터셋에서 각각 상대적 성능 향상률이 14.14%, 7.12%, 9.40%로 나타났습니다. 이는 단일 모델 대비 다중 에이전트 협력 시스템의 성능 향상을 보여주며, 수리 및 상식 추론 문제에서 효과적인 협력을 이룬 결과입니다.

### [TAPTRv3: Spatial and Temporal Context Foster Robust Tracking of Any Point in Long Video](https://arxiv.org/abs/2411.18671)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2411.18671.png)

Vote: 16

Authors: Tianhe Ren, Jinyuan Qu, Zhaoyang Zeng, Shilong Liu, Hongyang Li, Lei Zhang

- ***What's New***: TAPTRv3는 기존 TAPTRv2를 개선하여 긴 동영상에서도 더 강력한 포인트 추적을 가능하게 합니다. TAPTRv3는 공간 및 시간적 맥락을 활용하여 더 나은 특징 쿼리링을 수행하며, 이를 통해 추적의 견고성을 증가시킵니다.
- ***Technical Details***: TAPTRv3는 Context-aware Cross-Attention (CCA)와 Visibility-aware Long-Temporal Attention (VLTA) 기법을 소개합니다. CCA는 주변 공간 맥락을 레버리지하여 이미지 특징을 쿼리할 때 주의 점수의 품질을 향상시키며, VLTA는 과거 프레임들의 가시성을 고려하여 모든 프레임에 대해 시간적 주의를 수행하여 특징 드리프팅 문제를 효율적으로 해결합니다.
- ***Performance Highlights***: TAPTRv3는 대부분의 도전적인 데이터셋에서 TAPTRv2보다 큰 차이를 보이며, 가장 뛰어난 성능을 기록합니다. 대규모 추가 내부 데이터를 사용하여 훈련된 방법들과 비교해도 여전히 경쟁력 있는 성능을 보여줍니다.

### [MATATA: a weak-supervised MAthematical Tool-Assisted reasoning for Tabular Applications](https://arxiv.org/abs/2411.18915)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2411.18915.png)

Vote: 8

Authors: Gregory Senay, Luis Martí, Vishnou Vinayagame

- ***What's New***: MATATA는 새로운 수단으로서 대형 모델이나 외부 데이터, 광범위한 프롬프트 엔지니어링에 의존하지 않고 표 형식의 데이터 문제를 다루기 위한 합리적이고 계획적인 도구를 사용하여 훈련된 SLM(Small Language Models) 기반의 수학적 추론 방법입니다. 이는 데이터 프라이버시가 중요한 비즈니스 문맥에서 특히 유리합니다.
- ***Technical Details***: MATATA는 도구 증강 형식의 SLM을 활용하여 약한 지도 학습(Weak Supervision)과 자체 생성된 올바른 추론 경로를 통해 툴을 미세 조정하며, 이를 통해 다중 단계 추론(Multi-step Reasoning)을 수행합니다. 이 프로세스는 계획(Planning), 전문화된 도구를 통한 다중 단계 추론 경로 생성으로 이루어져 표 형식의 비즈니스 문서에서도 효율적으로 작동합니다.
- ***Performance Highlights***: MATATA는 FinQA 및 TAT-QA에서 최첨단의 성능을 보여줍니다. MATATA로 훈련된 모델은 GPT-4 기반의 프레임워크와 비교하여도 경쟁력을 가지며, TabMWP에서도 비비긴 모델들보다 성능이 우수합니다. 이 실험 결과는 작은 언어 모델(SLM)이면서도 효과적이고 확장 가능한 성능을 보여주며, 대형 모델 대비 비용 효율적임을 증명합니다.

### [FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait](https://arxiv.org/abs/2412.01064)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.01064.png)

Vote: 18

Authors: Dongchan Min, Gyoungsu Chae, Taekyung Ki

- ***What's New***: FLOAT는 오디오 기반의 회화적인 움직임 생성에서 모션 잠재 공간을 활용해 시각적으로 일관된 동영상 생성을 가능하게 하는 새로운 오디오 구동 말하는 초상화 생성 모델입니다. 특히, 음성에서 추출한 감정 기반의 움직임 제어를 통해 자연스럽고 감정 표현이 풍부한 움직임을 생성할 수 있습니다.
- ***Technical Details***: FLOAT는 flow matching (흐름 매칭) 생성 모델에 기반하여 모션 잠재 공간 내에서 말하는 움직임을 모델링합니다. 이를 위해 Transformer 기반의 vector field predictor를 도입하여 시간적 일관성을 유지하면서 효율적인 모션 샘플링을 가능하게 했습니다. 이를 통해 오디오와 관련된 개인의 감정 표현을 자연스럽게 반영하는 데 도움을 줍니다.
- ***Performance Highlights***: 다양한 실험을 통해 FLOAT가 최신 기술의 오디오 구동 말하는 초상화 생성 방법들에 비해 시각적 품질, 모션의 성실도, 그리고 효율성 면에서 뛰어난 성능을 보여주었습니다. 특히, 감정 인식 기반의 모션 제어를 통해 보다 자연스러운 표현의 말하는 영상을 생성할 수 있음을 입증했습니다.

### [Towards Cross-Lingual Audio Abuse Detection in Low-Resource Settings with Few-Shot Learning](https://arxiv.org/abs/2412.01408)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.01408.png)

Vote: 1

Authors: Aditya Narayan Sankaran, Reza Farahbaksh, Noel Crespi

- ***What's New***: 이 연구는 저자들이 제안한 모델-아그노스틱 메타-러닝(Model-Agnostic Meta-Learning; MAML) 기반 몇 개의 샷(Few-Shot) 크로스-언어 오디오 남용 감지 방법론을 통해 저자들이 직접 녹음한 10개의 인도 언어로 구성된 오디오 데이터를 사용하여 소량 자원 환경에서의 악성 오디오 콘텐츠 탐지 문제를 해결하려 한 것입니다.
- ***Technical Details***: 연구는 Wav2Vec과 Whisper와 같은 사전 학습된 오디오 표현을 활용하여 ADIMA 데이터셋을 사용하여 오디오 남용 언어를 분류합니다. 두 가지 특징 정규화(L2 노멀라이제이션, Temporal Mean)를 통해 이 특징의 효과를 평가하였습니다. 실험은 핑-샷(Few-Shot) 환경에서 크로스-언어 학습 및 모델의 일반화 능력을 평가하며 일반화 성능을 개선하기 위해 MAML 프레임워크를 사용하였습니다.
- ***Performance Highlights***: Whisper 모델은 L2 노멀라이제이션과 함께 사용했을 때 100 샷 환경에서 최고의 정확도(78.98%부터 85.22%)를 달성하였으며, 특징 시각화 연구를 통해 언어 유사성이 저자들의 크로스-언어 남용 감지를 어떻게 개선할 수 있는지 분석했습니다. 이는 저자들이 제안한 방법론이 저자나 연구자들에게 제공하는 실질적 혜택과 미래 연구 방향을 제시합니다.

### [Truth or Mirage? Towards End-to-End Factuality Evaluation with LLM-OASIS](https://arxiv.org/abs/2411.19655)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2411.19655.png)

Vote: 12

Authors: Federico Martelli, Roberto Navigli, Simone Tedeschi, Alessandro Scirè, Karim Ghonim, Andrei Stefan Bejgu

- ***What's New***: LLM-OASIS는 Wikipedia로부터 정보를 추출하고 사실을 수정하여 대규모 데이터셋을 생성함으로써 종단 간 사실 평가를 위한 첫 번째 대규모 자원입니다. Wikipedia의 주장 세트를 기반으로 사실적 텍스트와 비사실적 텍스트 쌍을 생성함으로써 사실성을 평가할 수 있는 환경을 제공합니다.
- ***Technical Details***: LLM-OASIS는 Wikipedia 페이지에서 발췌한 주장을 기반으로 비사실적인 정보를 포함하는 텍스트를 생성하는 파이프라인으로 구성됩니다. 이 시스템은 클레임 추출(Claim Extraction), 클레임 위조(Claim Falsification), 사실 및 비사실적 텍스트 생성(Factual and Unfactual Text Generation)을 통하여 구조화된 데이터를 생성합니다. 이러한 프로세스는 파이썬을 이용한 다양한 자연어 처리 기법 및 기계 학습 모델링을 통해 이루어집니다.
- ***Performance Highlights***: GPT-4o는 제안된 종단 간 사실 평가 작업에서 약 60%의 정확도를 달성했으며, 이는 최첨단 LLM에게도 LLM-OASIS 리소스가 상당한 도전을 제기함을 보여줍니다. 대체 모델 중 일부는 보다 전문화된 작업에 대해 경쟁력 있는 성능을 보였으며, 특히 80K Wikipedia 페이지에서 언어 및 도메인을 다양화하여 모든 문서가 자연 언어로 표현된 사실성을 평가할 수 있는 환경을 조성했습니다.

### [INCLUDE: Evaluating Multilingual Language Understanding with Regional Knowledge](https://arxiv.org/abs/2411.19799)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2411.19799.png)

Vote: 5

Authors: Micol Altomare, Alfonso Amayuelas, Ran Tamir, Antoine Bosselut, Imanol Schlag, Daniil Dzenhaliou, Joel Niklaus, Danylo Boiko, Rishabh Maheshwary, Shivalika Singh, Johan Samir Obando Ceron, Marjana Prifti Skenduli, Christopher Klamm, Jekaterina Novikova, Yiyang Nan, Esther Ploeger, Jebish Purbey, Negar Foroutan, Zeming Chen, Thenuka Ovin Weerasinghe, Jenny Chim, Aditya Kumar Dalmia, Sara Hooker, Viraat Aryabumi, Eldar Khalilov, Anna Sotnikova, Azril Hafizi Amirudin, Dominik Krzemiński, Abraham Diress, Marzieh Fadaee, Sharad Duwal, Drishti Sharma, Ayush Kumar Tarun, Daniel Fernando Erazo Florez, Fajri Koto, Mohamed A. Haggag, Syrielle Montariol, Sree Harsha Nelaturu, Gal Cohen, Shayekh Bin Islam, Roshan Santhosh, Swati Rajwal, Angelika Romanou, Selvan Sunitha Ravi, Michael Chang, Debjit Paul, Bardia Soltani Moakhar, Azmine Toushik Wasi, Serhan Yilmaz, Sara Rydell, Arshia Soltani Moakhar, Joseph Marvin Imperial, Mike Zhang, Snegha A, Gabriel Adriano de Melo, Fabian Farestam, Perttu Isotalo, Börje F. Karlsson, Maral Jabbarishiviari

- ***What's New***: INCLUDE 벤치마크는 44개의 언어로 구성되어 지역적 맥락에서 멀티링궐 대형 언어 모델(Large Language Model; LLM)의 언어 이해 능력을 평가하기 위한 방대한 고유 데이터셋을 소개합니다. 총 197,243개의 다중 선택 질문을 포함한 이 자원은 각국의 시험 자료를 통해 지역적 및 문화적 지식을 평가합니다.
- ***Technical Details***: INCLUDE는 44개의 언어 및 15개의 스크립트로, 1,926개의 다양한 시험에서 197,243개의 질문-답변(QA) 쌍을 데이터로 수집했습니다. 데이터 수집은 각국의 원주민 스피커와의 협력을 통해 지역 시험 자료를 얻어 이루어졌으며, 번역 후 생성되는 번역어를 피하고 각 언어에 포함된 문화적 뉘앙스를 포착하여 다언어 모델의 성능을 평가합니다.
- ***Performance Highlights***: GPT-4o 모델은 평균 정확도 77.1%를 획득하며 모든 도메인에서 가장 높은 성능을 보였습니다. 추가로 5-shot 설정과 zero-shot 체인-오브-생각(Chain-of-Thought; CoT) 설정 모두에서 적당한 성능 향상이 나타났습니다. 절반 이상의 언어에서 모델들은 지역적 지식을 요하는 질문에서 고전하는 경향을 보였습니다.

### [Generating a Low-code Complete Workflow via Task Decomposition and RAG](https://arxiv.org/abs/2412.00239)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.00239.png)

Vote: 2

Authors: Orlando Marquez Ayala, Patrice Béchard

- ***What's New***: 이 논문은 Task Decomposition과 Retrieval-Augmented Generation (RAG) 두 가지 기법을 Generative AI(GenAI) 시스템의 디자인 패턴으로 공식화하였습니다. 이 기법들은 복잡한 AI 기반의 소프트웨어 시스템에 통합되기 위한 재사용 가능한 솔루션을 제공합니다. GenAI 기반 워크플로우 생성 응용 프로그램을 구축하는 사례 연구를 통해 이러한 패턴의 산업적 적용 가능성을 제시합니다.
- ***Technical Details***: Task Decomposition은 AI 도메인에서 문제를 여러 하위 작업으로 나누어 해결하는 Divide and Conquer 접근 방식의 응용입니다. 이 기법은 복잡한 ML 작업을 보다 쉽게 해결할 수 있는 하위 작업으로 분해합니다. RAG는 환경 데이터와 상호작용하여 모델의 가중치에 저장된 정보에 제한을 받지 않고 모델의 출력을 조정하는 기법입니다. 이 두 기법은 데이터 수집, 모델 교육, 모델 평가, 배포 등 AI 개발 주기 전반에 걸쳐 기술적 영향을 미칩니다.
- ***Performance Highlights***: 제안된 GenAI 워크플로우 생성 시스템은 모듈화, 확장성 및 보안성에서 뛰어난 성과를 보여주었습니다. Task Decomposition과 RAG를 통해 데이터 레이블링 시간 단축과 시스템의 효율적 테스트가 가능해졌습니다. 그러나 이로 인해 모델 훈련의 복잡성이 증가하고 여러 AI 컴포넌트의 배포 부담이 증가했습니다. 전체적인 시스템 정확성과 성능 향상을 위해서는 하위 작업의 개별적인 분석과 테스트가 필수적임을 보여줍니다.

### [Improving speaker verification robustness with synthetic emotional utterances](https://arxiv.org/abs/2412.00319)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.00319.png)

Vote: 2

Authors: Aman Chadha, Andreas Stolcke, Nikhil Kumar Koditala, Minho Jin, Chelsea Jui-Ting Ju, Ruirui Li

- ***What's New***: 이 논문에서는 감정적 발화를 사용하는 새로운 데이터 증강 기법을 통해 연사 인증(Speaker Verification; SV) 시스템의 강인성을 개선하는 방법을 제안합니다. CycleGAN을 활용하여 특정 연사의 고유 목소리를 유지하면서 감정적 음성을 합성하여 데이터 세트를 보강하고자 하였습니다. 이는 감정적 음성을 처리하는 SV 모델의 성능을 개선함으로써 오류율(EER)을 최대 3.64% 감소시켰습니다.
- ***Technical Details***: 우리의 SV 모델은 멀티 레이어 LSTM 네트워크로 구성되어 있으며, 40차원의 Mel-spectrogram을 입력으로 받아 드 벡터(d-vector)를 출력하여, 이는 연사의 음성 특징을 캡슐화합니다. 이 모델은 일반화된 엔드투엔드(GE2E) 손실을 사용하여 훈련되었습니다. CycleGAN 프레임워크는 WORLD Vocoder를 사용하여 음성 특징을 추출하며, 스펙트럼 변환을 위한 MFCCs와 운율 변환을 위한 F0 특징을 별도로 처리하며, 감정적 톤의 오디오 특징을 변환하는 생성자와 변환된 데이터를 식별하는 판별자로 구성되어 있습니다.
- ***Performance Highlights***: CycleGAN을 활용하여 합성한 감정적 발화를 훈련 데이터에 포함한 경우, 감정적 발화를 처리할 때 SV 모델의 오류율(EER)이 1.08%에서 3.64%까지 상대적으로 감소되었음을 확인했습니다. 이는 연속적인 감정 변화에 잘 대응하지 못하는 기존 모델의 한계점을 극복하여, SV 시스템의 강인성을 증대시킵니다.

### [SOLAMI: Social Vision-Language-Action Modeling for Immersive Interaction with 3D Autonomous Characters](https://arxiv.org/abs/2412.00174)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.00174.png)

Vote: 17

Authors: Weiye Xiao, Tianxiang Ren, Zhiqian Lin, Zhongang Cai, Jianping Jiang, Ziwei Liu, Huaizhong Zhang, Zhengyu Lin, Lei Yang, Yang Gao

- ***What's New***: SOLAMI는 3D 자율 캐릭터와 몰입형 상호작용을 위한 최초의 종단간 사회적 비전-언어-행동 모델(VLA 모델)입니다. 사용자와의 다중모달 입력을 기반으로 3D 캐릭터의 사회적 상호작용을 구동하는 다중모달 응답(연설 및 모션)을 생성합니다.
- ***Technical Details***: SOLAMI는 두 가지 주요 구성요소로 작동합니다. 사용자로부터 입력된 연설과 모션을 이산화한 후, 디코더 전용 대형 언어 모델(Large Language Model; LLM) 백본이 이 입력을 바탕으로 캐릭터의 연설과 모션 토큰을 예측합니다. 훈련은 모션과 텍스트, 연설과 텍스트 사이의 모달 정렬과 다중턴 대화에 대한 지시 조정으로 구성됩니다. 합성된 다중모달 데이터 세트인 SynMSI를 사용하여 대규모 다중모달 데이터 구축 및 학습이 이루어집니다.
- ***Performance Highlights***: SOLAMI는 모션 품질 및 추론 지연 시간에서 DLP 방법을 포함한 다른 방법들에 비해 우수한 성능을 보여줍니다. 모션 품질은 다중 모달 데이터를 활용한 신체 언어 및 언어의 세부 사항을 정확하게 캡처하여, 자연스러운 상호작용 모션을 생성합니다. 음성의 경우, 캐릭터의 표정과 맞는 음성을 합성할 수 있는 능력을 보여줍니다. SOLAMI 아키텍처의 종단간 접근은 모듈식 파이프라인 접근보다 더 낮은 지연 시간을 가지며, 실제 인간 소통 과정과 자연스럽게 정렬된 방식으로 작동합니다.

### [VisOnlyQA: Large Vision Language Models Still Struggle with Visual Perception of Geometric Information](https://arxiv.org/abs/2412.00947)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.00947.png)

Vote: 7

Authors: Yusen Zhang, Sarkar Snigdha Sarathi Das, Rui Zhang, Ranran Haoran Zhang, Ryo Kamoi

- ***What's New***: VisOnlyQA는 대형 시각 언어 모델(Large Vision Language Models; LVLMs)의 시각적 인식 능력을 평가하기 위한 새로운 데이터셋을 소개합니다. 이 데이터셋은 과학적 도표에서 기하학적 및 수치적 정보에 대한 질문들의 시각적 인식 능력을 독립적으로 분석할 수 있도록 설계되었습니다.
- ***Technical Details***: VisOnlyQA 데이터셋은 4가지 유형의 도형에 관한 12개의 작업에 걸쳐 1,200개의 선택형 질문을 포함하고 있습니다. 또한 70,000개의 인공적인 학습 데이터를 제공합니다. 이 데이터셋은 LVLMs의 시각적 인식 능력을 평가할 수 있도록 설계되었으며, 다른 능력에 의존하지 않고 세밀한 시각 정보를 평가하는 것이 가능합니다.
- ***Performance Highlights***: 20개의 LVLM을 평가한 결과, GPT-4o와 Gemini 1.5 Pro를 포함한 최신 모델들이 VisOnlyQA에서 낮은 성능을 보이며, 인간보다 훨씬 성능이 떨어지는 결과를 보였습니다. 몇몇 작업에서는 세부 조정(Fine-tuning)을 통해 성능 향상의 가능성을 보였으나, 작업에 따라 제한적인 개선만 관찰되었습니다. 강화된 언어 모델이 시각적 인식을 개선하는 데 기여할 수 있다는 점이 확인되었습니다.

### [GRAPE: Generalizing Robot Policy via Preference Alignment](https://arxiv.org/abs/2411.19309)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2411.19309.png)

Vote: 32

Authors: Kaiyuan Zheng, Mingyu Ding, Zhaorun Chen, Chaoqi Wang, Yi Li, Dieter Fox, Huaxiu Yao, Joel Jang, Zijian Zhang

- ***What's New***: GRAPE는 로봇 정책의 일반화를 향상시키기 위해 선호도 정렬(Preference Alignment)을 도입한 새로운 방법입니다. 이를 통해 성공과 실패 시범 모두를 통해 보상을 모델링하여 다양한 작업으로의 일반화 성능을 강화합니다. 또한 복잡한 작업을 독립적인 단계들로 분할하고, 대규모 비전-언어 모델이 제안한 공간-시간적 제약을 통해 자동으로 선호 모델링을 가이드합니다.
- ***Technical Details***: GRAPE는 경로 단위로 비전-언어-행동(VLA) 모델을 정렬하고, 성공과 실패 시범에서 보상을 암묵적으로 모델링하여 일반화 성능을 향상시킵니다. 이를 위해 대형 비전 기반 모델을 사용하여 각 단계에 중요한 주요 지점을 식별하고, 사용자 지정 정렬 목표에 따라 각 단계에 대한 비용 함수 시리즈를 얻습니다. GRAPE는 시퀀싱 메커니즘에서 다단계 비용을 기반으로 정책을 점진적으로 조정하는 선호 최적화 방식을 채택합니다.
- ***Performance Highlights***: 실험 결과, GRAPE는 기존 VLA 모델들의 성능을 크게 향상시켜, 도메인 내 작업과 보지 못한 작업에서 각각 51.79%와 60.36%의 성공률 증가를 보여주었습니다. 또한, GRAPE는 안전성 및 효율성 등의 다양한 목표에 정렬될 수 있으며, 이에 따라 충돌률을 44.31% 줄이고 롤아웃 단계 길이를 11.15% 줄입니다. 이는 다양한 작업 환경에서 GRAPE의 높은 적응성과 효과를 보여줍니다.

### [OmniCreator: Self-Supervised Unified Generation with Universal Editing](https://arxiv.org/abs/2412.02114)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.02114.png)

Vote: 12

Authors: Lan Wang, Haodong Chen, Harry Yang, Ser-Nam Lim

- ***What's New***: OmniCreator는 텍스트 기반의 통합 이미지 및 비디오 생성 및 편집을 한 곳에서 수행할 수 있는 새로운 프레임워크입니다. 이는 기존의 특정 편집 타입이나 추가적인 제어에 의존하지 않고도 텍스트와 영상 간의 의미적 대응 관계를 학습하여 보편적인 편집 효과를 제공합니다.
- ***Technical Details***: OmniCreator는 CLIP의 시각 및 텍스트 임베딩을 비디오 입력에 맞추기 위해 어댑터를 CLIP 이미지 인코더에 도입하여 시간적 역학을 캡처하며, 이는 3D 컨볼루션 레이어로 구현됩니다. 또, 쿼리 트랜스포머를 사용하여 비디오와 텍스트 임베딩을 효과적으로 정렬하며, LoRA(Low-rank Adaption)을 공간 및 시간 레이어에 적용하여 계산 복잡성을 줄입니다. 이 구성 요소들은 텍스트와 비디오 간의 의미적 관계를 내재화하여, 전역적 시각 의미를 보존하면서 로컬 텍스트 의미를 통해 통제된 편집을 도입합니다.
- ***Performance Highlights***: OmniCreator는 OmniBench-99 데이터셋에서 경쟁 모델들보다 뛰어난 성능을 보이며, 이는 다양한 편집 타입과 시나리오에서의 성능을 입증합니다. 자동화된 평가 및 사용자인식과 함께 구조적 일관성, 편집 정확도, 전반적인 품질에서 높은 점수를 기록하며, 제어 가능 생성 분야를 진일보시킬 잠재력을 보여줍니다.

### [OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation](https://arxiv.org/abs/2412.02592)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.02592.png)

Vote: 8

Authors: Qintong Zhang, Zichen Wen, Ka-Ho Chow, Ying Li, Wentao Zhang, Bin Wang, Conghui He, Junyuan Zhang, Linke Ouyang

- ***What's New***: 이 연구는 OCR(Optical Character Recognition)이 RAG(Retrieval-Augmented Generation) 시스템에 미치는 영향을 처음으로 벤치마킹한 연구로, OHRBench라는 벤치마크를 통해 평가합니다. OHRBench는 6개의 실제 RAG 애플리케이션 도메인에서 추출한 350개의 PDF 문서와 문서 내 멀티모달 요소에서 비롯된 질문과 답을 포함합니다.
- ***Technical Details***: OHRBench는 법률, 금융, 교과서, 매뉴얼, 신문, 학계 등 6개 분야의 복잡한 PDF 문서와 해당 문서의 멀티모달 요소로부터 생성된 Q&A 쌍을 포함합니다. OCR 잡음에는 예측 오류로 인한 'Semantic Noise'와 문서 요소의 비균일한 표현에서 기인한 'Formatting Noise'가 있습니다. OHRBench는 다양한 수준의 잡음을 반영한 'Perturbed Structured Data'를 생성하여 RAG 시스템이 OCR 잡음에 어떻게 영향을 받는지 탐구할 수 있도록 합니다.
- ***Performance Highlights***: 현재의 OCR 솔루션은 고품질 RAG 지식 베이스 구축에 적합하지 않으며, 가장 우수한 OCR 솔루션조차 지식 베이스 구축에서 최소 7.5%의 성능 저하를 보입니다. 정보-언어 모델을 OCR 없이 사용해 결합했을 때, OCR 텍스트와 이미지 입력을 병합하면 성능이 최대 24.5% 향상되어 실제 텍스트 성능에 근접함을 보여줍니다. 이러한 결과는 VLM이 향후 RAG 시스템에 적용 가능성을 시사합니다.

### [VideoLights: Feature Refinement and Cross-Task Alignment Transformer for Joint Video Highlight Detection and Moment Retrieval](https://arxiv.org/abs/2412.01558)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.01558.png)

Vote: 2

Authors: Nabeel Mohammed, Md Rizwan Parvez, Shafin Rahman, Dhiman Paul

- ***What's New***: VideoLights는 비디오 분석을 위한 최신 기능 정제 및 태스크 연결 정보 교환 모델입니다. 이 모델은 비디오 하이라이트 검출(Video Highlight Detection; HD)과 순간 검색(Moment Retrieval; MR)을 동시에 수행하며, 비디오-텍스트 정렬 및 정제를 개선합니다. 강력한 특징 검출을 위해 LVLMs(대형 시각-언어 모델)를 활용하여 동시 학습 효율성을 높입니다.
- ***Technical Details***: VideoLights는 영상-텍스트 특징을 효과적으로 연결 및 정렬하기 위해 두 가지 주요 모듈을 사용합니다: (i) 특징 정제 및 정렬 모듈(Feature Refinement and Alignment; FRA)과 (ii) 양방향 교차 모달 융합 네트워크(Bi-Directional Cross-Modal Fusion; Bi-CMF). FRA는 텍스트와 비디오 간의 로컬 및 글로벌 특징 정렬을 제공하며, Bi-CMF는 강력한 질의 인지적 클립 표현을 생성합니다. 신규 손실 기능도 도입하여 오류를 적응적으로 교정하고, 대형 시각-언어 모델(BLIP-2)을 활용한 지능적 사전 학습을 통해 모델 성능을 강화합니다.
- ***Performance Highlights***: QVHighlights, TVSum, 그리고 Charades-STA 벤치마크에서 VideoLights는 기존 모델들 대비 평균 1% 이상의 성능 향상을 보였습니다. VideoLights는 QVHighlights에서 0.7%의 R@0.5 성능 향상을, Charades-STA에서 평균 2% 이상의 개선을 달성했으며, TVSum에서 평균 3.3% 이상의 성능 향상을 보여주었습니다. 이는 VideoLights의 교차-태스크와 교차-모달 상호작용 학습이 성능을 높이는 데 중요한 역할을 했음을 나타냅니다.

### [The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning](https://arxiv.org/abs/2412.00568)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2412.00568.png)

Vote: 13

Authors: Daniel Fortunato, Ruben Ohana, Suryanarayana Maddu, Keiya Hirashima, Marsha Berger, Fruzsina J. Agocs, Bruno Régaldo-Saint Blancard, Miles Cranmer, Drummond B. Fielding, Miguel Beneitez, Romain Watteaux, François Rozet, Stefan S. Nixon, Jared A. Goldberg, Jonah Miller, Stuart B. Dalziel, Shirley Ho, Blakesley Burkhart, Jeff Shen, Rudy Morel, Payel Mukhopadhyay, Liam H. Parker, Lucas Meyer, Yan-Fei Jiang, Rich R. Kerswell, Michael McCabe

- ***What's New***: The Well은 다양한 물리적 시스템의 대규모 시뮬레이션을 제공하는 데이터셋입니다. 이는 기계 학습 기반의 대리 모델링(surrogate modeling)을 위해 설계된 15TB 규모의 데이터로, 생물학적 시스템, 유체 동역학, 음향 산란, 초신성 폭발 등의 다양한 물리적 현상을 포함합니다. 이러한 데이터를 기계 학습의 벤치마크로 활용할 수 있으며, PyTorch 인터페이스를 제공하여 모델 훈련과 평가를 쉽게 할 수 있습니다.
- ***Technical Details***: 이 데이터셋은 16개의 서로 다른 데이터셋으로 구성되며, 각 데이터는 HDF5 형식으로 저장되어 있습니다. 데이터의 구성은 스칼라 및 벡터 값 필드를 포함하며, 각 필드는 공간적 및 시간적 차원에 맞춰 저장되어 있습니다. 모델은 주어진 시뮬레이션의 초기 조건이나 경계 조건을 기반으로 다음 시점의 상태를 예측하는 방식으로 훈련됩니다. 또한 Fourier Neural Operator, 요인화된 FNO, U-net 및 ConvNextU-net 등의 모델이 사용되었습니다.
- ***Performance Highlights***: 모델의 성능은 다양한 시뮬레이션 환경에서 Variance Scaled Root Mean Squared Error (VRMSE) 측정으로 평가되었습니다. 특정 데이터셋에서 CNextU-net이 돋보이는 성능을 보여주며 우세를 보였으나, 일부 데이터셋에서는 공간적 처리 영역을 선호하는 경향이 관찰되었습니다. 이는 다양한 물리적 조건에서 하나의 모델이 모두 잘 작동하기 어려움을 나타냅니다. 실험 결과는 12시간 동안 단일 NVIDIA H100 GPU에서 진행되었습니다.

### [Critical Tokens Matter: Token-Level Contrastive Estimation Enhence LLM's Reasoning Capability](https://arxiv.org/abs/2411.19943)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2411.19943.png)

Vote: 33

Authors: Chufan Shi, Yujiu Yang, Zhaopeng Tu, Jiahao Xu, Xing Wang, Siheng Li, Ruilin Luo, Tian Liang, Zicheng Lin

- ***What's New***: 이 논문에서는 '중요 토큰(Critical Tokens)'이 LLM의 추론 능력에 미치는 영향을 탐구하고, 새로운 cDPO라는 접근 방식을 제안합니다. 이 방법은 각 레벨에서 중요 토큰을 자동으로 식별하고, 이를 피드백 및 보상으로 활용하여 모델의 성능을 향상시킵니다.
- ***Technical Details***: 새로운 cDPO 접근 방식은 대조 추정법(Contrastive Estimation)을 사용하여 부정확한 추론 경로에 존재하는 중요 토큰을 식별합니다. 이 대조 추정법을 통해 긍정적 및 부정적인 모델의 생성 가능성을 비교하여 중요 토큰을 강조할 수 있습니다. 중요 토큰의 존재는 부정적인 추론 경로를 결과적으로 수정하는 데 기여하게 됩니다. 이후 중요한 토큰 정보를 활용하여 토큰 수준의 보상 신호를 통한 선호도 최적화(Preference Optimization) 프로세스를 수행합니다.
- ***Performance Highlights***: cDPO 방법은 GSM8K 및 MATH500 벤치마크에서 실험적으로 우수한 성능을 보이며, 여러 베이스라인 방법을 상회합니다. GSM8K에서는 평균 점수 77.2%, MATH500에서는 33.4%의 성과를 기록하여, Llama-3 및 DeepSeek 모델에서 최고의 성과를 나타냅니다. cDPO는 중요한 토큰을 식별하고 이에 기반한 피드백을 활용하여 모델의 추론 능력을 효과적으로 향상시킵니다.

### [GATE OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation](https://arxiv.org/abs/2411.18499)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2411.18499.png)

Vote: 17

Authors: Xiaojun Chang, Hao Zhang, Shuo Liu, Ziyao Guo, Tianhua Li, Wenqi Shao, Pengfei Zhou, Zhaopan Xu, Kaipeng Zhang, Yefei He, Xiaopeng Peng, Yuqi Lin, Chuanhao Li, Yu Qiao, Lirui Zhao, Yuxuan Xie, Jiajun Song, Yue Yang

- ***What's New***: GATE OpenING(OpenING)은 개방형 다중 모달 생성(Open-ended Multimodal Generation)을 평가하기 위한 종합적인 벤치마크로, 56개의 실제 과제를 포함한 5,400개의 고품질 인스턴스로 구성되어 있습니다. OpenING은 여행 가이드, 디자인, 브레인스토밍과 같은 다양한 일상 시나리오를 다루어 다채로운 통합 생성 방법을 도전하는 플랫폼을 제공합니다. 이것은 기존 벤치마크의 데이터 크기 및 다양성의 한계를 극복하고 보다 다양한 데이터 집합을 제공합니다.
- ***Technical Details***: OpenING은 인간이 주석을 단 5,400개의 인스턴스로 구성되며, 이는 23개의 메타 주제와 56개의 구체적인 과제로 나뉩니다. 이 벤치마크는 다양한 주제와 작업 요구를 충족하기 위한 것이며, 각 인스턴스는 꼼꼼하게 설계된 질문으로 다양한 데이터를 제공합니다. 또한 OpenING은 OpenLEAF와 InterleavedBench에 비해 더 포괄적인 데이터와 과제 범위를 제공합니다. 뿐만 아니라, IntJudge라는 새로운 심사 모델을 도입하여 개방형 다중 모달 생성 방법을 평가하는데, 인간의 평가와 일치율이 82.42%로 뛰어난 성능을 보여줍니다.
- ***Performance Highlights***: OpenING에 대한 광범위한 실험 결과는 현재의 인터리브드 생성 메소드가 개선의 여지가 많음을 보여줍니다. 대표적인 통합 파이프라인인 GPT-4o+DALL-E-3는 두드러진 성능을 보여주는 반면, 기존의 엔드투엔드 모델인 MiniGPT-5와 GILL는 상대적으로 낮은 성능을 보였습니다. IntJudge를 이용한 평가는 인간의 평가와 좋은 일치율을 보였으며, 특히 GPT 기반 평가보다 11.34% 개선된 성능을 얻었습니다. 또한, 텍스트 대 이미지의 영향에서 텍스트 생성이 상대적으로 강한 성능을 보임을 관찰할 수 있었으며, 자연 이미지가 여전히 생성된 이미지보다 더 선호됨을 확인했습니다.

### [Training Noise Token Pruning](https://arxiv.org/abs/2411.18092)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2411.18092.png)

Vote: 1

Authors: Mingxing Rao, Daniel Moyer, Bohan Jiang

- ***What's New***: 이번 연구에서는 비전 트랜스포머를 위한 새로운 훈련 노이즈 토큰(Training Noise Token; TNT) 가지치기 기법을 제안합니다. 이 방법은 알고리즘적으로 토큰 드롭핑 대상을 지속적인 가산 노이즈로 완화하여, 훈련 중 매끄러운 최적화를 제공하면서도 배포 설정에서 이산 드롭핑에 따른 계산적 이득을 유지합니다. Rate-Distortion 문헌과의 이론적 연결을 제공하고, TNT가 기존의 가지치기 방법보다 이점을 갖는다는 것을 입증합니다.
- ***Technical Details***: TNT 가지치기 기법은 정보 병목 문제의 제한적 사례로 토큰 가지치기를 다시 규정합니다. 채널 제약으로 토큰 드롭핑 비율을 보고, 왜곡 지표로 정확도 페널티를 봄으로써 압축 문헌의 아이디어를 적용합니다. 이러한 조건의 완화는 연속적인 최적화 기대 사항을 제공하며, 더욱 쉽게 최적화할 수 있습니다. 이 접근 방법은 ImageNet-1K 벤치마크에서 ViT와 DeiT 아키텍처에서 최근 가지치기 방법보다 우수한 성능을 보여주었습니다.
- ***Performance Highlights***: TNT는 보유 토큰 비율이 낮을수록(토큰의 큰 부분을 제거할 때) 높은 정확성을 유지하면서 계산 부하를 크게 줄이는 데 탁월한 성능을 보입니다. ImageNet 데이터셋에서의 평가 결과는 비전 트랜스포머의 효율성을 향상시키는 데 있어 TNT의 잠재적 영향을 강조합니다.

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
