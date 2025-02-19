# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using GPT-4o.

Thanks to [@AK391](https://github.com/AK391) for great work.

## Daily Papers (2025-02-19)

### [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11089.png)

Vote: 71

Authors: Zhenda Xie, Lean Wang, Yuqing Wang, Wangding Zeng, Ming Zhang, Jingyang Yuan, Zhengyan Zhang, Y. X. Wei, Liang Zhao, Damai Dai, Junyu Luo, Zhiping Xiao, Chong Ruan, Wenfeng Liang, Huazuo Gao

- ***What's New***: 이 논문에서는 Native Sparse Attention (NSA)라는 새로운 주의 메커니즘을 소개합니다. 이는 하드웨어 친화적 최적화와 알고리즘 혁신을 통합하여 긴 문맥을 효율적으로 처리할 수 있도록 설계되었습니다. NSA는 동적 계층적 희소 전략을 사용하여 전역 문맥 인식 및 로컬 정밀도를 순차적으로 보존합니다.
- ***Technical Details***: NSA에서는 군집화된 토큰 압축(compression)과 선택(selection)을 결합하여 세 가지 주의 경로를 통해 입력 시퀀스를 처리합니다: 압축된 조대 토큰, 선택적으로 유지되는 미세 토큰, 그리고 로컬 문맥을 위한 슬라이딩 윈도우(sliding window)가 그 세 가지입니다. 이는 하드웨어 효율적인 블록 단위의 희소 주의를 위한 최적의 커널 설계를 통해 이루어집니다.
- ***Performance Highlights***: NSA는 64k-길이의 시퀀스에서 Full Attention에 비해 여러 단계에서 상당한 속도 향상을 달성했습니다. 실제 실험 결과로는, 디코딩, 순전파(forward propagation), 역전파(backward propagation)에서 6에서 11배의 속도 향상이 관찰되었습니다. 이는 Full Attention에 비해 모델 성능을 유지하거나 초과하는 결과를 보여주었습니다.

### [Learning Getting-Up Policies for Real-World Humanoid Robots](https://arxiv.org/abs/2502.12152)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.12152.png)

Vote: 32

Authors: Xialin He, Saurabh Gupta, Zixuan Chen, Runpei Dong

- ***What's New***: 이 연구는 휴머노이드 로봇이 다양한 초기 자세와 다양한 지형에서 일어서는 기술을 습득하게 하는 새로운 학습 프레임워크를 제안하였습니다. 이는 실제 크기의 휴머노이드 로봇에 학습된 일어서는 정책을 성공적으로 적용한 첫 번째 사례입니다.
- ***Technical Details***: HUMANUP 시스템은 두 단계의 강화학습(제1단계: Discovery Policy, 제2단계: Deployable Policy)을 통해 다양한 초기 상태와 지형에서 휴머노이드 로봇이 안정적으로 일어설 수 있는 컨트롤러를 학습합니다. 첫 번째 단계에서는 주어진 미션을 적은 제약 속에서 해결하도록 하며, 두 번째 단계에서는 실제 사용에 적합한 매끄럽고 느린 동작으로 정교하게 다듬습니다. 두 단계 학습 과정은 Sim2Real 학습 커리큘럼을 통해 진행되며, 각 단계는 충돌 메쉬 완화, 자세 무작위화, 제어 규제와 같은 커리큘럼 요소를 포함합니다.
- ***Performance Highlights***: HUMANUP는 G1 휴머노이드 로봇이 평평한 표면뿐만 아니라 미끄러운 지형이나 경사진 환경에서도 안정적으로 일어설 수 있도록 하였습니다. 실험 결과, HUMANUP는 기본 G1 컨트롤러 대비 78.3%의 더 높은 성공률을 기록했으며, 다양한 지형과 초기 자세에 대한 일반화 성능을 증명했습니다.

### [SWE-Lancer: Can Frontier LLMs Earn $1 Million from Real-World Freelance Software Engineering?](https://arxiv.org/abs/2502.12115)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.12115.png)

Vote: 27

Authors: Samuel Miserendino, Tejal Patwardhan, Michele Wang, Johannes Heidecke

- ***What's New***: SWE-Lancer는 Upwork에서 수집된 1,488개의 프리랜서 소프트웨어 엔지니어링 작업(Tasks)으로 구성된 벤치마크입니다. 이는 대규모 언어 모델(Language Models; LLMs)이 실세계 문제를 해결할 수 있는 능력을 평가하고 그 경제적 가치를 측정하기 위해 설계되었습니다.
- ***Technical Details***: SWE-Lancer는 개별 기여자(Individual Contributor; IC) SWE 작업과 SWE 관리자(Manager) 작업으로 나뉩니다. IC SWE 작업은 엔드투엔드(End-to-End; E2E) 테스트를 통해 평가되며, SWE 관리 작업에서는 모델이 다양한 구현 제안 중 최적의 솔루션을 선택해야 합니다. 전체 데이터셋은 오픈 소스 Docker 이미지로 제공되며 세부 사항은 공개된 GitHub 리포지토리를 통해 확인할 수 있습니다.
- ***Performance Highlights***: Claude 3.5 Sonnet 모델은 IC SWE 작업에서 26.2%, SWE 관리 작업에서 44.9%의 성능을 기록하며, 이는 최대 50만 달러의 SWE-Lancer Diamond 평가 세트 중 208,050달러를 획득했습니다. 동일한 데이터를 기반으로 한 모든 모델은 100만 달러의 풀 셋에 대한 가능 지급액보다 낮은 성과를 보였습니다.

### [ReLearn: Unlearning via Learning for Large Language Models](https://arxiv.org/abs/2502.11190)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11190.png)

Vote: 19

Authors: Mengru Wang, Ningyuan Zhao, Shumin Deng, Huajun Chen, Ningyu Zhang, Nay Oo, Liming Yang, Bryan Hooi, Sendong Zhao, Haoming Xu

- ***What's New***: ReLearn은 대형 언어 모델(Large Language Model; LLM)의 비허가된 지식을 효과적으로 제거하기 위해 데이터 증강과 정교한 파인 튜닝을 활용하는 새로운 언러닝 파이프라인과 관련 평가 프레임워크를 소개합니다. 이 프레임워크는 지식 수준의 보존을 측정하기 위한 Knowledge Forgetting Rate (KFR) 및 Knowledge Retention Rate (KRR)와 생성 품질을 평가하기 위한 Linguistic Score (LS)를 도입합니다. ReLearn은 목표 지식을 잊어버리면서도 고품질 출력을 유지할 수 있습니다.
- ***Technical Details***: ReLearn은 민감한 정보를 새로운 허가된 지식으로 대체하여 모델의 언어적 능력을 보존하는 데이터 증강과 파인 튜닝 과정을 포함합니다. 평가 프레임워크는 관찰 엔티티와 의미적 일관성을 평가하는 Entity Coverage Score (ECS)와 Natural Language Inference (NLI)를 통해 이루어집니다. LLS는 Perplexity (PPL), Brunet's Index (BI), Honoré's Statistic (HS)를 사용하여 언어적 품질을 측정합니다.
- ***Performance Highlights***: KnowUnDo 및 TOFU와 같은 벤치마크 데이터셋에서 ReLearn은 KFR 0.85, KRR 0.74를 유지하며 좋은 성능을 보였습니다. 반면 Gradient Ascent (GA)와 Negative Preference Optimization (NPO)는 반복적이고 비일관적인 출력을 초래하며 낮은 Fluency와 Relevance를 보였습니다. 그러나 ReLearn은 이러한 영역에서 모델의 언어적 질을 잘 보존하였습니다.

### [CRANE: Reasoning with constrained LLM generation](https://arxiv.org/abs/2502.09061)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.09061.png)

Vote: 17

Authors: Shubham Ugare, Gagandeep Singh, Sasa Misailovic, Debangshu Banerjee, Tarun Suresh

- ***What's New***: CRANE는 제약된 생성(Constrained LLM Generation)을 활용하여 LLM의 추론 능력을 보존하면서 구문적, 의미적 정확성을 유지하는 새로운 방법론을 제안합니다. 기존의 엄격한 구문 강제화가 LLM의 추론 능력을 저하시킬 수 있다는 이론적 설명을 제공하고, 이를 개선하기 위해 출력 구문에 추가 규칙을 보강하여 LLM의 표현력을 유지할 수 있음을 이론적으로 입증합니다.
- ***Technical Details***: CRANE는 추론을 위한 비제약적 생성(Unconstrained Generation)과 구조적으로 올바른 출력을 위한 제약적 생성(Constrained Generation)을 효과적으로 번갈아 가며 수행하는 방법론입니다. 다양한 오픈소스 LLM과 벤치마크에서 이 메커니즘을 통해 높은 정확성을 달성합니다. 특히, LT(n) 스텝의 튜링 머신(Turing Machine)을 시뮬레이션하며 출력 구문에 추가 규칙을 삽입하는 구문 강화 방법론을 제안하였습니다.
- ***Performance Highlights***: CRANE는 여러 오픈소스 LLM에서 실험한 결과, GSM-symbolic 및 FOLIO와 같은 도전적인 상징적 추론 벤치마크에서 기존 최첨단(SoTA) 제약 디코딩 전략 및 비제약 디코딩에 비해 최대 10%까지 정확성을 개선했습니다. 이는 제약적 생성과 비제약적 생성의 강점을 효과적으로 균형 있게 통합했음을 보여줍니다.

### [IHEval: Evaluating Language Models on Following the Instruction Hierarchy](https://arxiv.org/abs/2502.08745)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.08745.png)

Vote: 15

Authors: Haodong Wang, Xianfeng Tang, Meng Jiang, Qingyu Yin, Zhihan Zhang, Haoming Jiang, Zixuan Zhang, Yichuan Li, Bing Yin, Yifan Gao, Zheng Li, Zhaoxuan Tan, Xin Liu, Shiyang Li

- ***What's New***: IHEval는 새로운 벤치마크로, 다양한 우선순위의 지시(instruction)들이 상충하거나 일치하는 경우를 포함하여 지시 계층(instruction hierarchy)을 따르는 모델의 능력을 평가합니다. 총 3,538개의 예제와 9개의 과제로 구성되어 있습니다.
- ***Technical Details***: IHEval는 시스템 메시지(system messages), 사용자 메시지(user messages), 대화 히스토리(conversation history), 도구 출력(tool outputs)이라는 네 가지 유형의 입력을 포함하여 포괄적인 입력 계층구조를 다루며, 각 설정에 대한 효율적이고 재현 가능한 평가를 위해 프로그래머블 평가를 지원합니다.
- ***Performance Highlights***: ILMS는 상충되는 지시를 마주할 때 고수준의 지시를 우선시하는 데 어려움을 겪으며, 오픈 소스 모델들은 이러한 충돌을 해결하는 데 있어서 50% 이하의 정확도를 기록했습니다. 특히, GPT-4o는 가장 경쟁력 있는 성능을 보였으나 여전히 충돌이 해결되지 않는 경우 성능이 크게 저하되었습니다.

### [HermesFlow: Seamlessly Closing the Gap in Multimodal Understanding and Generation](https://arxiv.org/abs/2502.12148)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.12148.png)

Vote: 15

Authors: Bin Cui, Wentao Zhang, Minghao Xu, Xinchen Zhang, Ling Yang, Chenming Shang, Ye Tian

- ***What's New***: HermesFlow는 다중 모드 대형 언어 모델(Multimodal Large Language Models; MLLMs)의 이해와 생성 능력 간의 큰 차이를 발견하고, 이 격차를 해소하기 위해 설계된 프레임워크입니다. 이해 데이터를 생성 데이터와 결합하여 Homologous Preference 데이터를 사용할 수 있습니다. 이러한 데이터를 통해 Pair-DPO와 self-play iterative 최적화를 수행하여 다중 모드 이해와 생성 능력을 효과적으로 정렬합니다.
- ***Technical Details***: HermesFlow는 Homologous Input 데이터를 받아들여 이해와 생성 선호도 데이터를 수집합니다. 이해 선호도 데이터 수집을 위해 이미지 캡셔닝 작업을 수행하고, 생성 선호도 데이터 수집을 위해 입력 프롬프트에서 다수의 이미지를 생성하며, self-VQA 평가 방식을 사용하여 선택합니다. Pair-DPO는 같은 의미 영역 내에서 이해와 생성 선호도 데이터를 동시에 최적화하는 훈련 알고리즘으로, 다양한 이해와 생성 작업에 대해 한 번에 개선을 제공합니다. iterative 최적화를 통해 모델 자체 개선을 달성할 수 있습니다.
- ***Performance Highlights***: HermesFlow는 다양한 멀티모달 이해 벤치마크에서 기존의 MLLMs보다 더 나은 성능을 달성했습니다. 특히 이해와 생성 능력 간의 차이를 성공적으로 줄였습니다. 기준 모델인 Show-o와 비교하여 이해 능력뿐 아니라 생성 능력에서도 높은 성과를 보였습니다. 더불어 사용자의 광범위한 연구에서 시각적 생성의 질에 대해 긍정적인 평가를 받았습니다.

### [How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training](https://arxiv.org/abs/2502.11196)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11196.png)

Vote: 14

Authors: Shumin Deng, Huajun Chen, Hui Jin, Ningyu Zhang, Yixin Ou, Yunzhi Yao, Jiacheng Sun, Zhenguo Li

- ***What's New***: 이 논문은 LLMs의 새로운 지식 획득 메커니즘을 지식 회로(knowledge circuits) 관점에서 탐구하여, 지속적인 사전 훈련 중 지식 회로의 진화 과정을 분석합니다. 새로 획득하는 지식이 기존 지식과의 연관성에 영향을 받는다는 점, 지식 회로의 진화가 형성에서 최적화로의 뚜렷한 단계 변화를 보이며, 깊은 레이어에서 얕은 레이어로의 패턴을 따른다는 주요 결과를 도출했습니다.
- ***Technical Details***: 지식 회로의 진화를 분석하기 위해, 모델의 행동을 전체적으로 대변하는 계산 서브그래프(circuit)를 발견하여 각 태스크에 대한 모델의 성능을 분석합니다. 이 논문의 접근 방식은 Deep-to-Shallow 패턴을 밝히기 위해 중요도 점수를 기반으로 각 엣지를 평가하는 EAP-IG 기법을 사용하여 지속적인 사전 훈련 동안 지식 회로의 변화를 추적합니다.
- ***Performance Highlights***: 지식 회로의 성능 분석 결과, 관련된 새로운 지식은 완전히 새로운 지식보다 더 효율적으로 통합됩니다. 또한, 지속적인 사전 훈련 중 성능 향상은 주로 기존 구조의 효율성을 최적화한 결과임을 나타냅니다. 이를 통해 지식 회로의 동적 변화를 지속적으로 모니터링하고, 사전 훈련 전략을 개선할 수 있는 방향을 제시합니다.

### [SURGE: On the Potential of Large Language Models as General-Purpose Surrogate Code Executors](https://arxiv.org/abs/2502.11167)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11167.png)

Vote: 11

Authors: Siqiao Huang, Bohan Lyu, Zichen Liang

- ***What's New***: 이 논문에서는 LLMs(Large Language Models)가 코드 실행 결과를 예측할 수 있는 일반 목적의 대체 코드 실행기(Surrogate Code Executors)로서의 잠재력을 조사하기 위해 SURGE라는 포괄적인 벤치마크를 소개합니다. SURGE는 다중 언어 프로그래밍 과제, 대회 수준의 문제, 과학적 계산 등 8개의 주요 측면을 포함하고 있습니다.
- ***Technical Details***: SURGE는 여러 오픈 소스 및 상용 LLMs를 평가하여 모델 크기와 학습 데이터의 규모가 대체 실행 정확성에 미치는 영향을 분석합니다. 데이터셋은 다양한 프로그램 실행 시나리오를 평가하기 위해 설계되었으며, 체인 오브 띠아이(Chain-of-Thought)와 같은 방법을 통해 몇 가지 학습 전략과도 결합하여 평가합니다.
- ***Performance Highlights***: 모델의 성능은 주어진 실행 시간이 짧을수록 높은 예측 정확성을 보이며, Claude-3.5-Sonnet과 같은 상용 모델은 다른 모델에 비해 전반적으로 더 나은 성능을 보여주었습니다. 그러나 초급 구조적인 C++ 문법 이해에는 어려움이 있었습니다.

### [I Think, Therefore I Diffuse: Enabling Multimodal In-Context Reasoning in Diffusion Models](https://arxiv.org/abs/2502.10458)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.10458.png)

Vote: 11

Authors: Kfir Aberman, Dan Xu, Zhenxing Mi, Guocheng Qian, Sergey Tulyakov, Hanrong Ye, Kuan-Chieh Wang, Runtao Liu

- **What's New**: ThinkDiff라는 새로운 정렬 패러다임이 소개되었습니다. 이는 VLMs(Vision-Language Models)의 능력을 확장하여 텍스트-이미지 확산 모델에 멀티모달 인컨텍스트(reasoning in-context) 추론을 가능하게 합니다. VLMs를 확산 디코더에 직접 맞추지 않고, 대형 언어 모델(LLM) 디코더에 정렬하여 훈련합니다. 이렇게 함으로써, 복잡한 훈련과 데이터셋이 없이도 이해, 추론, 그리고 조합 능력을 효과적으로 발휘할 수 있게 됩니다.
- **Technical Details**: ThinkDiff는 VLM을 LLM 디코더와 정렬하는 프락시(Proxy) 작업을 통해 멀티모달 기능을 확산 모델로 전송합니다. 이 모델은 이미지와 텍스트 입력에서 멀티모달 이해와 추론을 강화하기 위해 VLM의 깊은 토큰 특징을 활용합니다. 또한, 정렬 네트워크와 RMSNorm 레이어를 사용하여 학습 중의 수렴 문제를 해결합니다. 두 가지 변형이 있으며, 각각 LVLM(Large Vision-Language Model)과 CLIP 이미지 인코더를 사용합니다.
- **Performance Highlights**: ThinkDiff는 CoBSAT 벤치마크에서 멀티모달 추론 생성에서 정확도를 19.2%에서 46.3%로 크게 향상시켰습니다. 이 모델은 4개의 A100 GPU에서 단 5시간 동안의 훈련으로만 가능합니다. 또한 여러 이미지와 텍스트를 논리적으로 일관된 이미지로 구성하는 데 뛰어난 성능을 보여줍니다.

### [Diffusion-Sharpening: Fine-tuning Diffusion Models with Denoising Trajectory Sharpening](https://arxiv.org/abs/2502.12146)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.12146.png)

Vote: 11

Authors: Bin Cui, Mengdi Wang, Yunhai Tong, Xinchen Zhang, Ling Yang, Ye Tian

- ***What's New***: Diffusion-Sharpening 방법은 새로운 fine-tuning 접근법으로, 샘플링 경로 최적화를 통해 diffusion 모델의 성능을 향상시킵니다. 이 방법은 기존의 RL 기반 fine-tuning 방법들이 시간 단계에 초점을 맞춰 전체적인 샘플링 경로 최적화가 부족했던 부분을 보완하며, 샘플링 경로의 최적화를 통해 효율적인 모델 alignement를 가능하게 합니다.
- ***Technical Details***: Diffusion-Sharpening 프레임워크는 (i) 간섭 경로에서 다중 경로의 샘플링을 통해 최고의 경로를 찾고, (ii) 경로 적분을 통해 보상을 계산하며, (iii) 최적 경로로 향상되도록 모델을 훈련 시킵니다. 두 가지 구현 방법을 제안합니다: SFT-Diffusion-Sharpening은 사전 이미지-텍스트 데이터셋을 사용한 감독 학습 기반의 fine-tuning을 통해 임의의 보상 모델을 활용하여 최적화를 가능하게 하고, RLHF-Diffusion-Sharpening은 온라인 방법으로 양과 음의 샘플을 생성해 DPO 손실을 통해 자기 인도 학습을 수행합니다.
- ***Performance Highlights***: Diffusion-Sharpening은 다양한 평가 지표에서 RL 기반 fine-tuning 방법과 샘플링 경로 최적화 방법을 능가하며, 텍스트 정렬, 조합 능력 및 인간 선호도 측면에서 최고의 성능을 보여 주었습니다. RLHF-Diffusion-Sharpening은 탁월한 학습 및 추론 효율성을 보이며, 다양한 보상 모델에 대한 뛰어난 적응력과 일반화 능력을 보여줍니다. 이를 통해 모델의 텍스트-이미지 정렬, 조합 능력 및 전반적인 품질이 향상됩니다.

### [The Mirage of Model Editing: Revisiting Evaluation in the Wild](https://arxiv.org/abs/2502.11177)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11177.png)

Vote: 7

Authors: Xinyu Ma, Wanli Yang, Qi Cao, Jiajun Tan, Fei Sun, Xueqi Cheng, Dawei Yin, Huawei Shen

- ***What's New***: 이 논문에서는 대규모 언어 모델(LLMs)의 모델 편집 성능을 더욱 현실적으로 평가하기 위해 QAEdit라는 새로운 벤치마크와 평가 프레임워크를 소개합니다. 기존 연구에서 인공적인 평가 방법론의 한계를 지적하며, 실제 시나리오에서의 효과적인 모델 편집을 위해 엄격한 평가 기준을 확립하였습니다.
- ***Technical Details***: QAEdit는 Natural Questions, TriviaQA, SimpleQA와 같은 인기있는 QA 데이터셋을 바탕으로 설계되었으며, 실제 세계의 QA 작업으로부터 답을 주입하는 편집 방법을 평가할 수 있도록 구성되었습니다. 다양한 QA 시나리오를 포괄하는 19,249개의 샘플을 포함하고, 모델 편집의 강력한 분석을 위해 입력, 생성 전략, 출력 트렁케이션, 평가 지표를 상호 비교합니다.
- ***Performance Highlights***: 실험 결과, 현재의 모델 편집 방법들은 QAEdit에서 평균 38.5%의 성공률을 보여 기존 연구에서 보고한 결과인 약 96%에 크게 미치지 못했습니다. 실제 시나리오에서의 순차적 편집 실험에서는 1,000개의 수정만을 해도 심각하게 실패하여 성공률이 약 10%까지 떨어짐을 확인하였습니다.

### [Intuitive physics understanding emerges from self-supervised pretraining on natural videos](https://arxiv.org/abs/2502.11831)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11831.png)

Vote: 7

Authors: Yann LeCun, Nicolas Ballas, Mahmoud Assran, Michael Rabbat, Emmanuel Dupoux, Quentin Garrido, Adrien Bardes, Laurent Najman

- ***What's New***: 이 연구에서는 자연 비디오의 셀프-슈퍼바이즈드 프리트레이닝(self-supervised pretraining)을 통해 직관적인 물리(physics) 이해가 어떻게 발생하는지 조사하고 있습니다. 특히, 비디오의 마스킹 된 영역을 예측하도록 훈련된 일반 목적의 딥 뉴럴 네트워크(deep neural network) 모델들이 객체 영속성(object permanence)과 형태 일관성(shape consistency)과 같은 다양한 직관적인 물리 특성을 이해한다고 발견했습니다.
- ***Technical Details***: V-JEPA(Joint Embedding Predictive Architecture) 아키텍처는 예측 부호화(predictive coding) 가설과 일치하는 방법으로, 비디오에서 마스킹된 부분을 재구성하여 비디오 프레임을 나타내는 법을 학습합니다. 이는 기대 위반(violation-of-expectation) 프레임워크를 사용하여 직관적인 물리 이해를 조사하며, 특정 작업이나 적응 없이도 가능합니다.
- ***Performance Highlights***: V-JEPA는 IntPhys 벤치마크에서 98%, 그리고 InfLevel 벤치마크에서 62%의 제로샷(zero-shot) 정확도를 달성했습니다. 이는 현재의 멀티모달 대형 언어 모델(multimodal large language models)이 거의 기회 수준의 성능을 보이는 것과 대조적입니다. V-JEPA는 교육 데이터, 프리트레이닝 프레딕션 목표, 모델 크기와 같은 요소에 영향을 받지만, 가장 단순한 버전조차도 직관적 물리 이해를 획득할 수 있음을 시사합니다.

### [SAFE-SQL: Self-Augmented In-Context Learning with Fine-grained Example Selection for Text-to-SQL](https://arxiv.org/abs/2502.11438)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11438.png)

Vote: 6

Authors: Byeongjeong Kim, Ingeol Baek, Hwanhee Lee, Jimin Lee

- ***What's New***: SAFE-SQL은 텍스트-투-SQL(Text-to-SQL) 작업에서 고품질의 예시를 생성하고 필터링하여 성능을 향상시키는 새로운 프레임워크입니다. 이 방법은 사전 학습된 대형 언어 모델(LLM)의 생성 능력을 활용하여 자동으로 텍스트-투-SQL 예시를 생성, 필터링하여 사용하며, 추가적인 학습 없이도 정확도를 높입니다.
- ***Technical Details***: SAFE-SQL은 스키마 링크(Schema Linking), 예시 생성(Example Generation), 임베딩 유사성(Embedding Similarity), 키워드 및 구조적 정렬(Structural Alignment), 그리고 추론 경로 유효성(Relevance Path Validity)을 기반으로 예시를 평가하는 구조화된 필터링 메커니즘을 갖추고 있습니다. 이를 통해 데이터베이스 스키마와 자연어 질문 간의 구조적 유사성을 보장하여 정확한 SQL 쿼리를 생성합니다.
- ***Performance Highlights***: SAFE-SQL은 Spider 데이터셋을 기반으로 한 테스트에서 특히 어려운 시나리오에서의 성능 향상을 보였으며, 다른 전통적인 방법들과 비교해 더 높은 정확도를 기록했습니다. 특히, 포괄적인 추론 경로를 통해 복잡한 쿼리 생성 능력을 향상시킴으로써 높은 성능을 입증했습니다.

### [Explorer: Scaling Exploration-driven Web Trajectory Synthesis for Multimodal Web Agents](https://arxiv.org/abs/2502.11357)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11357.png)

Vote: 6

Authors: Corby Rosset, Yadong Lu, Boyu Gou, Yu Su, Vardaan Pahuja, Spencer Whitehead, Arindam Mitra, Ahmed Awadallah

- ***What's New***: Explorer는 자동화된 웹 탐색을 통해 대규모의 웹 궤적 데이터셋을 생성하는 새로운 프레임워크를 소개합니다. 이 데이터셋은 94,000개 이상의 멀티모달 웹 궤적을 포함하며, 49,000개의 고유 URL과 720,000개의 스크린샷, 3,300만 개의 웹 요소를 포함합니다. Explorer를 통해 생성된 데이터셋은 다양한 도메인과 과제를 다룹니다.
- ***Technical Details***: Explorer는 다중 에이전트 파이프라인을 활용하여 웹 환경을 체계적으로 탐색하고, 다양한 실제 과제를 수집합니다. 이 파이프라인은 초기 추상 과제 제안에서 출발하여, 웹 탐색을 통해 특정 과제로 점진적으로 세분화하는 과정을 거칩니다. 각각의 궤적은 스크린샷, HTML, 접근성 트리와 같은 다양한 아티팩트로 주석이 달려 있어 웹 에이전트 교육에 대한 종합적인 지식을 제공합니다.
- ***Performance Highlights***: Explorer를 통해 교육된 모델은 Mind2Web-Live, Multimodal-Mind2Web, MiniWob++와 같은 웹 에이전트 벤치마크에서 뛰어난 성능을 보였습니다. 특히 기존의 웹 에이전트 기준선들을 크게 상회하는 성적을 기록하며, 데이터 규모의 중요성을 강조합니다. 교육 데이터의 크기를 늘려가며 전반적인 성능이 향상되는 것을 확인할 수 있습니다.

### [MagicArticulate: Make Your 3D Models Articulation-Ready](https://arxiv.org/abs/2502.12135)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.12135.png)

Vote: 5

Authors: Chaoyue Song, Xiu Li, Jiashi Feng, Guosheng Lin, Xiaoyang Guo, Fayao Liu, Jun Hao Liew, Zhongcong Xu, Fan Yang, Yiwen Chen, Jianfeng Zhang

- ***What's New***: MagicArticulate는 정적 3D 모델을 실제적인 애니메이션을 지원하는 가동 준비 상태로 자동 변환시키는 효과적인 프레임워크입니다. 이 연구에서 소개하는 주요 기여는 세 가지입니다. 첫째, Objaverse-XL에서 수집된 33,000개 이상의 고품질 가동 주석이 포함된 대규모 벤치마크인 Articulation-XL을 소개합니다. 둘째, 다양한 3D 모델 내의 뼈나 관절의 수를 자연스럽게 처리할 수 있도록 자동 회귀 변환기(auto-regressive transformer)를 활용한 새로운 골격 생성 방법을 제안합니다. 셋째, 볼륨 측지 거리(volumetric geodesic distance) 우선권을 통합하여 기능적 확산 프로세스를 통해 스키닝 가중치(skinning weights)를 예측합니다.
- ***Technical Details***: MagicArticulate의 골격 생성은 입력 3D 메시가 주어졌을 때, 자동 회귀(sequence modeling) 문제로 과제를 재구성하여 실행됩니다. 또한, 스키닝 가중치 예측은 메시 표면의 기능적 확산 방식으로 예측되며, 볼륨 측지 거리 우선권을 포함하여 메시와 관절 사이의 복잡한 토폴로지를 효과적으로 처리합니다. 이러한 설계는 대규모 데이터셋에서의 우수한 확장성을 보여주며, 다양한 객체 범주에 잘 일반화됩니다.
- ***Performance Highlights***: Articulation-XL과 ModelsResource에서의 광범위한 실험 결과, MagicArticulate는 다양한 객체 범주에 걸쳐 기존 방법들을 능가하는 높은 품질의 가동을 달성하였음을 보여줍니다. MagicArticulate의 방법은 아티스트가 생성한 리소스와 유사하면서도 더욱 정확한 결과를 제공하였으며, 특히 애플리케이션에서 리얼리스틱한 포즈 조작을 지원하는 자연스러운 애니메이션을 생성할 수 있도록 합니다.

### [video-SALMONN-o1: Reasoning-enhanced Audio-visual Large Language Model](https://arxiv.org/abs/2502.11775)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11775.png)

Vote: 5

Authors: Changli Tang, Zejun MA, Jimin Zhuang, Wei Li, Chao Zhang, Yudong Yang, Yixuan Li, Guangzhi Sun

- ***What's New***: video-SALMONN-o1은 비디오 이해 작업을 위한 최초의 오픈소스 추론 강화 오디오-비주얼 대형 언어 모델(LLM)입니다. 이번 모델은 일반 비디오 이해에서의 추론 능력을 개선하기 위해 추론 집약적인 데이터세트와 대조적인 단계 선택을 이용한 프로세스 직접 선호 최적화(pDPO) 방법을 도입하였습니다.
- ***Technical Details***: video-SALMONN-o1은 시그리프(SigLIP) 비주얼 인코더와 큐웬(Qwen) 2 백본 LLM을 기반으로 제작되었습니다. 제안된 pDPO 방법은 다중 모달 입력에 맞춘 대조 단계 선택을 통해 각 단계의 보상 모델링을 구현합니다. 자사의 합성 비디오 감지를 위한 RivaBench 벤치마크와 같은 새로운 벤치마크를 소개하여 성능을 평가합니다.
- ***Performance Highlights***: video-SALMONN-o1은 LLaVA-OneVision 기반 라인과 비교하여 다수의 비디오 추론 벤치마크에서 3-8%의 절대 정확도 향상을 달성했습니다. pDPO는 RivaBench에서 감독 모델보다 6-8%의 향상을 기록하였고, 이 모델은 제로샷 합성 비디오 감지 능력을 입증했습니다.

### [Cuckoo: An IE Free Rider Hatched by Massive Nutrition in LLM's Nest](https://arxiv.org/abs/2502.11275)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11275.png)

Vote: 5

Authors: Zilong Wang, Jingbo Shang, Letian Peng, Feng Yao

- ***What's New***: 이 논문에서는 대규모 언어 모델(LLM)의 방대한 영양소를 활용하여 정보 추출(Information Extraction; IE)모델을 발전시킨 Cuckoo를 소개합니다. Cuckoo는 기존의 IE 모델과 달리 LLM의 훈련 데이터에서 추출된 정보를 기반으로 발전하며, 수작업이 아닌 자동화된 데이터를 지속적으로 학습하여 발전할 수 있습니다.
- ***Technical Details***: Cuckoo는 LLM의 사전 훈련 및 사후 훈련 데이터를 변환해서 총 102.6M의 추출형 데이터를 수집, 다음 토큰 추출(Next Tokens Extraction; NTE) 패러다임을 제안합니다. 이는 주어진 문맥에서 이미 존재하는 토큰을 추출하는 방식으로, BIO 태그 체계를 사용하여 다중 태그를 효율적으로 추출합니다. 이러한 방식은 파라미터 효율성, 추론 효율성, 적응성에서 이점을 가집니다. RoBERTa 모델을 활용하여 대규모 NTE 데이터를 기반으로 지속적으로 훈련했습니다.
- ***Performance Highlights***: Cuckoo는 기존의 여러 IE 사전 훈련 데이터 세트를 기반으로 한 모델과 비교해 월등한 성능을 보이며, Named Entity Recognition (NER)과 같은 기본 정보 추출 작업에서 강력한 성능을 발휘합니다. 또한, Cuckoo는 LLM의 포스트 훈련 데이터와 함께 진화할 수 있으며, 이는 IE 태깅에서 in-context 학습 능력을 갖추고 있음을 보여줍니다. Rainbow Cuckoo라는 확장형 모델은 다양한 포스트 훈련 리소스를 통합하여 더욱 향상된 성능을 제공합니다.

### [Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation](https://arxiv.org/abs/2502.08826)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.08826.png)

Vote: 4

Authors: Mohammad Mahdi Abootorabi, Mahdi Dehghani, Mahdieh Soleymani Baghshah, Amirhosein Zobeiri, Omid Ghahroodi, Ehsaneddin Asgari, Mohammadali Mohammadkhani, Bardia Mohammadi

- ***What's New***: 이 논문은 멀티모달 리트리벌-증강 생성(Multimodal Retrieval-Augmented Generation; RAG)의 발전을 포괄적으로 분석한 최초의 종합적인 설문조사를 제공하며, 다양한 모달리티를 통합하여 대형 언어 모델의 환각 문제와 업데이트되지 않은 데이터베이스에 대한 의존성을 해결하는 방식을 소개합니다.
- ***Technical Details***: 멀티모달 RAG 시스템은 문서, 이미지, 오디오, 비디오 등의 다양한 포맷을 효과적으로 결합하기 위해 혁신적인 검색, 융합, 증강 및 생성 전략을 사용합니다. 수학적 공식화, 데이터셋, 벤치마크, 평가 방법론뿐만 아니라 다양한 멀티모달 RAG 시나리오를 분석합니다.
- ***Performance Highlights***: 멀티모달 RAG는 전통적인 단일모드 RAG와 차별화되는 고유한 도전 과제를 가지고 있으며, 특히 교차모드 정렬과 추론에서의 새로운 도전 과제가 있습니다. 논문에서는 100개 이상의 연구를 검토하여 최근의 혁신 및 프론티어를 탐구하며, 향후 연구의 방향성을 제시합니다.

### [Building A Proof-Oriented Programmer That Is 64% Better Than GPT-4o Under Data Scarsity](https://arxiv.org/abs/2502.11901)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11901.png)

Vote: 4

Authors: Dylan Zhang, Justin Wang, Tianran Sun

- ***What's New***: 이 연구는 데이터 부족 환경에서 증명 지향 프로그래머(proof-oriented programmer)를 64% 더 향상시킨 PoPilot 모델을 소개합니다. 기존의 LLM들은 증명 지향 프로그래밍 언어(F* 등)를 위한 충분한 데이터가 부족하고, 프로젝트 수준의 대규모 증명 데이터가 부족하여 언어 모델의 적응력이 제한됩니다. 이 연구는 이러한 문제를 해결하기 위해 F* 언어의 기본적인 증명 문제를 생성하고, 다양한 코딩 데이터를 통합하며, 기존 레포지토리 내에 새로운 증명 및 수리 데이터를 생성하는 방법을 제안합니다.
- ***Technical Details***: PoPilot 모델은 14B 파라미터를 가지며, 데이터 부족 문제를 해결하기 위해 다양한 코드 데이터와 F* 언어의 기본적인 증명 문제를 합성하고, 기존 레포지토리에서 새로운 증명 및 수리 데이터를 생성합니다. 이 모델은 함수 수준과 레포지토리 수준의 코드를 합성하고 수리하는 능력을 가지고 있으며, 증명 생성 및 수리 작업에서도 마찬가지로 뛰어난 성능을 보입니다.
- ***Performance Highlights***: PoPilot 모델은 GPT-4o보다 프로젝트 수준의 증명 지향 프로그래밍에서 64% 향상된 성능을 보이며, GPT-4o의 출력물을 수리하여 54% 향상된 성능을 보여줍니다. 이는 PoPilot이 기존의 대형 LLM들과 비교하여 더욱 효율적이고 강력한 증명 프로그래밍 도구임을 보여줍니다.

### [Dyve: Thinking Fast and Slow for Dynamic Process Verification](https://arxiv.org/abs/2502.11157)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11157.png)

Vote: 4

Authors: Zhijian Xu, Jianyuan Zhong, Qiang Xu, Xiangyu Wen, Zeju Li

- ***What's New***: 이 연구에서는 Dyve라는 새로운 동적 프로세스 검증기(Dynamic Process Verifier)를 소개합니다. 이는 Kahneman의 시스템 이론(Systems Theory)에 영감을 받아, 빠른 사고와 느린 사고를 결합해 대형 언어 모델(LLMs)의 추론 오류를 감지하는 기능을 강화합니다. Dyve는 즉각적인 토큰 수준 확인을 위한 시스템 1과 복잡한 분석을 위한 시스템 2를 적응적으로 적용합니다.
- ***Technical Details***: Dyve는 단계별 합의 필터링된 프로세스 감독 기법(step-wise consensus-filtered process supervision technique)을 도입해 고품질의 감독 신호를 확보합니다. 이 기법은 몬테카를로 추정(Monte Carlo estimation), LLM-as-a-Judge, 및 전문화된 추론 모델을 결합합니다. Dyve는 1,200,000개의 노이즈가 섞인 롤아웃 데이터에서 약 117,000개의 고품질 교육 사례를 추출합니다. 이러한 교육 데이터를 통해 학습한 DeepSeek-R1-Distill-Qwen-14B 모델은 빠른 시스템 1 확인과 포괄적인 시스템 2 검정을 수행할 수 있습니다.
- ***Performance Highlights***: Dyve는 ProcessBench 및 MATH 데이터셋에서 기존의 프로세스 기반 검증기를 능가하는 성능을 보여주었습니다. 또한, Proposer LLM과 결합하여 Best-of-N 설정에서 우수한 성능을 발휘하며, 특히 OlympiadBench 및 OmniMATH와 같은 복잡한 데이터셋에서도 뛰어난 일반화 능력을 보였습니다. 예측 정확도는 Proposer LLM과 결합했을 때 최대 95.5%에 도달했습니다.

### [Talk Structurally, Act Hierarchically: A Collaborative Framework for LLM Multi-Agent Systems](https://arxiv.org/abs/2502.11098)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11098.png)

Vote: 4

Authors: Wei-Yao Wang, Shingo Takamatsu, Zhao Wang, Sota Moriyama, Briti Gangopadhyay

- ***What's New***: 이 논문에서는 LLM 기반의 멀티 에이전트 시스템(LLM-MA)들을 위한 새로운 협력 프레임워크인 TalkHier를 소개합니다. 이 시스템은 구조화된 의사소통 프로토콜과 계층적 정제 시스템을 도입하여 복잡한 작업에서 에이전트들이 상호 협력할 수 있도록 합니다.
- ***Technical Details***: TalkHier는 전체 시스템을 직렬적 통신 및 정제 프로세스로 운영하며, 이러한 프레임워크를 통해 각 에이전트의 개별적 메모리를 보존하고, 에이전트가 독립적으로 과거의 상호작용 및 지식을 유지할 수 있게 합니다. 이러한 에이전트-별 메모리는 독립성과 지속성을 지원하여 일관성 있고 정보를 반영한 의사 결정을 가능하게 합니다.
- ***Performance Highlights***: TalkHier는 다양한 벤치마크 및 실험에서 우수한 성능을 기록했습니다. MMLU 벤치마크에서는 평균 88.38%의 정확도로, 오픈소스 멀티 에이전트 모델 AgentVerse(83.66%) 대비 5.64% 더 높은 성능을 보였습니다. 또한, WikiQA 벤치마크에서는 ROUGE-1 0.3461, BERTScore 0.6079로, 오픈 도메인 질문 응답에서 뛰어난 성능을 입증했습니다.

### [PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning](https://arxiv.org/abs/2502.12054)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.12054.png)

Vote: 4

Authors: Yuxuan Dong, Jun Liu, Xinyu Zhang, Yanrui Wu, Basura Fernando, Jiaxing Huang, Chengyou Jia, Lingling Zhang, Mike Zheng Shou

- ***What's New***: PhysReason은 물리학 기반의 추론 능력을 평가하기 위한 포괄적인 벤치마크로, 기존 평가의 빈틈을 메우고자 설계되었습니다. 물리학 정리와 제약을 필요로 하는 1,200개의 다양한 문제를 포함하고 있으며, 쉬움, 중간, 어려움 세 가지 난이도로 나누어져 있습니다. 이 벤치마크는 LLMs(대규모 언어 모델)의 물리 기반 추론 능력을 체계적으로 평가하기 위해 고안되었습니다.
- ***Technical Details***: PhysReason은 25%의 지식 기반 문제와 75%의 추론 기반 문제로 구성된 1,200개의 문제를 포함하고 있습니다. 특히, 복잡한 추론은 평균 8.1단계가 소요되며, 난이도 높은 문제는 15.6단계에 이릅니다. 우리는 모델의 성능을 철저히 평가하기 위해 물리적 솔루션 자동 점수화 프레임워크(Physics Solution Auto Scoring Framework; PSAS)를 제안하며, 이는 답변 수준과 단계 수준의 평가를 포함합니다.
- ***Performance Highlights***: Deepseek-R1, Gemini-2.0-Flash-Thinking 등 최상의 모델은 답변 수준 평가에서 60% 미만의 성능을 보였으며, 문제 난이도가 높아질수록 성능이 현저히 떨어졌습니다. 이는 현재 LLMs가 복잡한 물리 기반 추론에서는 큰 한계를 보이며, 추후 연구가 필요함을 시사합니다.

### [System Message Generation for User Preferences using Open-Source Models](https://arxiv.org/abs/2502.11330)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11330.png)

Vote: 4

Authors: Minbyul Jeong, Minsoo Khang, Dawoon Jung, Teakgyu Hong, Jungho Cho

- ***What's New***: 이 논문은 SYSGEN이라는 새로운 파이프라인을 도입하여 시스템 메시지(System Messages)를 공개 소스 모델(Open-source Models)을 활용해 생성합니다. 이는 시스템 메시지가 없는 기존의 감독된 미세 조정 데이터셋(Supervised Fine-Tuning Dataset)을 보다 잘 정렬된 보조 응답을 만들어 개선합니다.
- ***Technical Details***: SYSGEN 파이프라인은 시스템 메시지를 8개의 주요 기능으로 문장 수준에서 분류하여 생성합니다. 그런 다음 태그가 잘못 지정된 문구를 제거하고, 태그의 키 기능을 검증함으로써 더욱 자연스러운 시스템 메시지를 형성합니다. 이로써 더 잘 정렬된 보조 응답을 생성하여 사용자 지침(User Instructions)과의 일치를 높입니다.
- ***Performance Highlights***: SYSGEN 데이터로 학습된 다양한 공개 소스 모델들은 Multifacet 벤치마크에서 모델의 응답이 시스템 메시지와 사용자 지침에 더 잘 맞도록 개선되었으며, Open LLM Leaderboard 2와 같은 미지의 벤치마크에서도 성능 저하를 최소화합니다.

### [Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning](https://arxiv.org/abs/2502.10550)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.10550.png)

Vote: 4

Authors: Alexey K. Kovalev, Nikita Kachaev, Egor Cherepanov, Aleksandr I. Panov

- ***What's New***: 본 연구는 MIKASA라는 새로운 벤치마크를 도입하여 강화학습(Reinforcement Learning; RL) 에이전트의 메모리 활용 능력을 평가합니다. MIKASA는 로봇 테이블탑 조작 시나리오에서 에이전트가 부분 가시성 아래서 역할을 수행할 때 필요한 메모리 의존 기술을 측정하기 위한 32개의 과제를 담고 있습니다.
- ***Technical Details***: MIKASA는 메모리 집중 RL 작업을 위한 새로운 분류 프레임워크를 제안하고, 다양한 시나리오에서 메모리 강화 에이전트를 체계적으로 평가할 수 있는 통합 벤치마크인 MIKASA-Base를 수집했습니다. MIKASA-Robo라는 벤치마크는 12개의 카테고리에 걸친 32개의 메모리 집중 로봇 테이블탑 조작 작업을 담고 있으며, 각 작업의 난이도와 구성 모드를 다양하게 조절할 수 있습니다.
- ***Performance Highlights***: 클래식 PPO 알고리즘을 사용한 실험에서 메모리 강화된 LSTM 백본이 메모리 집중 작업에 대해 더 높은 성공률을 보였지만, 사전 지식 없이 학습하는 데 한계가 있었습니다. 새로운 환경에서의 평가는 기존의 메모리 메커니즘이 복잡한 작업에서 도전에 직면할 때 성능이 악화되는 것을 보여주며, 메모리 강화 RL 에이전트의 발전을 위한 효과적인 벤치마크로서 MIKASA-Robo의 유효성을 입증합니다.

### [One Example Shown, Many Concepts Known! Counterexample-Driven Conceptual Reasoning in Mathematical LLMs](https://arxiv.org/abs/2502.10454)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.10454.png)

Vote: 3

Authors: Ying Shen, Zhikun Xu, Jiayi Kuang, Yangning Li, Xiaoyu Tan, Yinghui Li, Yi Yu, Hai-Tao Zheng, Haojing Huang, Philip S. Yu, Xinnian Liang, Chao Qu, Wenlian Lu

- ***What's New***: 이 논문에서는 수학적 대형 언어 모델(Mathematical Large Language Models; LLMs)이 수학적 추론을 증명하는데 현재 사용되는 두 가지 주요 방법론이 한계를 지니고 있음을 주장합니다. 인간의 수학 교육에서 흔히 사용되는 '반례에 의한 증명(proof by counterexamples)'을 모방하여 LLMs의 수학적 개념 추론 능력을 향상시키기 위해 COUNTERMATH라는 새로운 벤치마크를 제안합니다.
- ***Technical Details***: COUNTERMATH는 대학 수준의 수학적 명제에 반례(counterexamples)를 제시하여 LLMs의 수학적 개념 이해도를 평가할 수 있도록 설계되었습니다. 총 1,216개의 명제-이유(rationale) 쌍이 수학 교과서에서 수집되었으며, 이상 조건 하에서 특정 명제를 반증하는 데 초점을 맞추고 있습니다. 또한, 데이터 엔지니어링 프레임워크를 개발하여 LLMs의 추가적인 훈련 데이터를 자동으로 수집합니다.
- ***Performance Highlights***: COUNTERMATH 벤치마크에서 주류 수학 LLMs의 성능 평가가 이루어졌으며, OpenAI o1처럼 최신 LLMs가 반례 기반의 증명 능력이 부족하다는 것이 밝혀졌습니다. 특히 위상수학(topology)과 실해석학(real analysis) 분야에서의 낮은 성능은 향후 연구 방향을 제시합니다. 제한된 1,025개의 훈련 샘플만을 사용하여 미세 조정(fine-tuning)한 모델은 강력한 성능을 나타내며, 반례 기반 학습이 수학적 추론 개선에 효과적이라는 것을 입증합니다.

### [Can a Single Model Master Both Multi-turn Conversations and Tool Use? CALM: A Unified Conversational Agentic Language Model](https://arxiv.org/abs/2502.08820)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.08820.png)

Vote: 3

Authors: Oussama Elachqar, Jeremiah Greer, Gokhan Tur, Dilek Hakkani-Tür, William Zeng, Ze Yang, Emre Can Acikgoz, Akul Datta, Emmanouil Koukoumidis

- ***What's New***: CoALM은 대화 지향적 언어 모델인 CoALM-IT를 통해 통합된 접근 방식을 제안하여 대화적 및 에이전트적 기능을 통합하였습니다. CoALM은 멀티턴 대화를 다루면서 복잡한 API 사용법을 함께 통합함으로써, GPT-4o 같은 기존의 도메인 특화 모델들을 능가하는 성능을 보입니다.
- ***Technical Details***: CoALM-IT는 멀티턴 ReAct 추론과 복잡한 API 사용을 결합한 멀티태스크 데이터셋입니다. 이를 기반으로 CoALM 가족인 CoALM 8B, 70B, 405B를 훈련하였으며, 각각의 모델은 Llama 3.1 시리즈를 기반으로 합니다. 특히, 새로운 CoALM-IT 데이터셋은 ReAct 스타일의 다단계 추론을 활용하여 멀티턴 TOD 시나리오에서 대화 상태 추적 및 복잡한 함수 호출을 포함합니다.
- ***Performance Highlights***: CoALM 모델들은 MultiWOZ 2.4에서 CoALM 70B가 69.4%의 성공률과 43.8%의 DST 정확도를 기록하여, GPT-4o보다 나은 성능을 보였습니다. 또한, BFCL V3에서는 CoALM 405B가 63.34%의 전체 정확도를 기록하여 GPT-4o를 능가하였으며, 이는 공개 소스 모델 중 문자 그대로 최고 수준의 성능을 나타냅니다.

### [ILIAS: Instance-Level Image retrieval At Scale](https://arxiv.org/abs/2502.11748)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11748.png)

Vote: 3

Authors: Nikos Efthymiadis, Zakaria Laskar, Ondřej Chum, Giorgos Tolias, Jiří Matas, Nikolaos-Antonios Ypsilantis, Anna Manko, Giorgos Kordopatis-Zilos, Pavel Šuma, Vladan Stojnić

- ***What's New***: 이번 연구에서는 대규모 인스턴스 수준 이미지 검색을 위한 ILIAS라는 새로운 시험 데이터셋을 도입합니다. 이는 현재 및 미래의 모델과 검색 기법이 특정 개체를 인식하는 능력을 평가하기 위해 고안되었습니다. ILIAS는 무려 1,000개의 개체 인스턴스에 대한 쿼리 이미지와 양성 이미지를 포함하고 있으며, 이는 YFCC100M에서 수백만 개의 이미지를 사용하는 대규모 검색을 지원합니다.
- ***Technical Details***: ILIAS 데이터셋은 2014년 이후에 등장한 것으로 확인된 쿼리 개체만 포함하여 추가적인 주석 작업 없이도 오탐(false negative)을 피합니다. 검색 성능 평가는 mAP(mean Average Precision)@1k 지표를 사용하며, 인스턴스 수준 이미지 검색의 실제 문제를 해결하는 데 초점을 맞추고 있습니다. 또한, 해당 데이터셋은 시각-언어 모델(Vision-Language Models; VLM)과 같은 다양한 기법에 대한 포괄적인 벤치마킹을 허용합니다.
- ***Performance Highlights***: 벤치마크를 통해 관찰된 바에 따르면, 특정 도메인을 대상으로 훈련된 모델들은 해당 도메인 내에서는 뛰어난 성능을 보이지만, ILIAS에서는 성과가 떨어집니다. 반면, 다중 도메인 학습과 클래스 감독(Class Supervision)을 활용한 선형 적응 레이어 학습은 성능을 향상시킵니다. 특히, 지역 서술자(local descriptors)가 뚜렷한 배경 클러터가 있을 때 중요한 역할을 하며, 텍스트에서 이미지로의 성능이 이미지에서 이미지로의 성능에 근접함을 확인할 수 있었습니다.

### [EQ-VAE: Equivariance Regularized Latent Space for Improved Generative Image Modeling](https://arxiv.org/abs/2502.09509)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.09509.png)

Vote: 3

Authors: Spyros Gidaris, Ioannis Kakogeorgiou, Nikos Komodakis, Theodoros Kouzelis

- ***What's New***: EQ-VAE는 기존 오토인코더(autoencoder)의 라텐트 공간(latent space)에 동등 변환(equivariance)을 적용하여 생성적 이미지 모델링 성능을 향상시키는 새로운 정규화 접근법을 제안합니다. 이 방법은 사전 학습된 오토인코더를 EQ-VAE로 미세 조정하여 기존 최첨단 생성 모델의 성능을 향상시키며, 특히 DiT-XL/2 모델에서는 5번의 SD-VAE 미세 조정만으로 7배의 속도 향상을 달성하였습니다.
- ***Technical Details***: EQ-VAE는 선형 변환에서 라텐트 공간의 동등 변환을 장려하는 단순한 정규화 접근법을 사용하여 기존 오토인코더 구조의 변화를 필요로 하지 않습니다. 이 방법은 이미지에 가해진 공간 변환과 라텐트 표현에 가해진 변환 간의 불일치를 최소화하며, 이는 특정 변환에 대해 변환 후 라텐트 재구성이 입력 이미지의 해당 변환과 일치하도록 합니다. EQ-VAE는 연속 및 이산 오토인코더 모두에 호환 가능합니다.
- ***Performance Highlights***: EQ-VAE를 적용한 DiT 모델은 기존의 SD-VAE와 비교하여 생성적 성능에서 상당한 향상을 보였으며, 특히 DiT-XL/2는 7M 이터레이션(iteration)에서 9.6 GFID를 달성했던 모델이 EQ-VAE를 통해 단 1.5M 이터레이션에서 8.8 GFID를 기록하는 성과를 냈습니다. 또한 REPA와의 결합을 통해 기존의 REPA보다 4배 빠른 수렴을 달성했습니다.

### [Towards Data-Efficient Pretraining for Atomic Property Prediction](https://arxiv.org/abs/2502.11085)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11085.png)

Vote: 2

Authors: Yasir Ghunaim, Bernard Ghanem, Hasan Abed Al Kader Hammoud

- ***What's New***: 본 논문에서는 원자 특성 예측(Atomic Property Prediction) 분야에서 데이터와 자원의 확장이 반드시 필요한 것이 아니라, 전략적인 데이터 선택이 성능을 동일하거나 더 낮은 자원 소비로도 달성할 수 있음을 보여줍니다. 이를 위해 화학적 유사성 지수(Chemical Similarity Index; CSI)라는 새로운 지표를 도입하여, 프리트레이닝 데이터셋과 다운스트림 과제 간의 정렬 정도를 측정할 수 있습니다.
- ***Technical Details***: CSI는 컴퓨터 비전의 Fréchet Inception Distance(FID)에서 영감을 받아 설계된 분자 그래프 용어로, 상위 프리트레이닝 데이터셋과 하위 결과 예측에서의 데이터셋간의 유사성을 측정합니다. 이를 통해 적절한 프리트레이닝 데이터셋을 소규모로 선택하여도 광범위한 데이터셋에 기반한 모델의 성능을 초과할 수 있음을 증명했습니다. 특히, 모델의 학습에 있어서 정량적 측면에서 원하는 목표에 부합하는 데이터셋을 선택하는 방식이 예측 성능을 더욱 향상시킵니다.
- ***Performance Highlights***: ANI-1x 데이터셋을 이용한 프리트레이닝은 24배 적은 컴퓨팅 자원을 사용했음에도 불구하고, 대규모 데이터셋과 경쟁할 수 있는 성능을 보였습니다. 다운스트림 작업에서 ANI-1x가 포함된 혼합 데이터셋보다 높은 성능을 보였으며, 예측 성능이 CSI에 있어 낮은 값을 가질수록 향상됨을 보여줍니다.

### [Large Language Models and Mathematical Reasoning Failures](https://arxiv.org/abs/2502.11574)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11574.png)

Vote: 2

Authors: Johan Boye, Birger Moell

- ***What's New***: 이 연구는 대형 언어 모델(LLMs)의 수학적 추론 능력을 고등학교 수준의 50개 새로 구축된 단어 문제로 평가합니다. 기존 연구들이 정답률에만 초점을 맞춘 것과 달리, 이 연구는 최종 답변 뿐만 아니라 솔루션 단계도 철저히 분석하여 추론 실패를 식별합니다.
- ***Technical Details***: 연구진은 Mixtral, Llama, Gemini, GPT-4o, OpenAI의 o1 모델을 포함한 8개의 최신 모델을 평가하였습니다. 문제들은 자연어로 제시되며, 고등학교 수준의 수학적 지식이 필요합니다. 문제 유형에는 공간적 추론, 전략적 계획, 산술 문제 등이 포함됩니다.
- ***Performance Highlights***: 새로운 모델인 o3-mini와 deepseek-r1은 더 높은 정확도를 보였지만, 모든 모델이 논리적 결함에도 올바른 답변을 제공하거나, 잘못된 논리로 인해 정답을 이루지 못하는 등 여전히 공간상 추론과 전략적 계획에서 오류를 보였습니다. o1 모델은 50개의 문제 중 37개를 정확히 해결하며 가장 우수한 성과를 보였습니다.

### [Diffusion Models without Classifier-free Guidance](https://arxiv.org/abs/2502.12154)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.12154.png)

Vote: 2

Authors: Baining Guo, Zhicong Tang, Dong Chen, Jianmin Bao

- ***What's New***: 이 논문에서는 새로운 모델 가이던스(Model-guidance; MG) 방법을 도입하여, 기존의 Classifier-free guidance(CFG)를 제거한 확산 모델(Diffusion Models) 훈련을 제안합니다. 이 방법은 기존 데이터 분포 모델링을 뛰어넘어 사후 확률(Postterior Probability)을 통합하여 확산 모델의 성능을 향상시킵니다.
- ***Technical Details***: MG는 확산 모델의 조건부와 비조건부 모델을 분리하여 학습할 필요없이, 모델 자체를 암묵적 분류기로 변환하여 포스터리어 확률의 점수를 직접 학습합니다. 더불어, MG는 간단한 한 줄의 코드 수정만으로 기존 모델에 플러그 앤 플레이 할 수 있으며, 훈련과 추론 속도를 동시에 개선합니다. 이 방법론은 또한 이미지네트(Imagenet) 256×256 벤치마크에서 최첨단 성능을 달성하였습니다.
- ***Performance Highlights***: MG는 이미지네트 256×256에서 FID 점수 1.34를 기록하며, 기존 CFG보다 우수한 결과를 냅니다. 또한, MG는 훈련 속도를 6.5배 가속하며, 추론 속도를 두 배 이상 향상시킵니다. 이러한 결과는 확산 모델의 세팅과 데이터셋에 잘 확장될 수 있음을 보여줍니다.

### [Data Valuation using Neural Networks for Efficient Instruction Fine-Tuning](https://arxiv.org/abs/2502.09969)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.09969.png)

Vote: 1

Authors: Ishika Agarwal, Dilek Hakkani-Tür

- ***What's New***: 이 논문에서는 대형 언어 모델을 사용한 데이터 영향을 평가하는데 있어 NN-CIFT(Neural Networks for effiCient Instruction Fine-Tuning)라는 소형 신경망을 활용하여 기존의 비효율적인 데이터 영향을 효율적으로 추정하는 방법을 제안합니다. 이로 인해 최대 99%의 비용 절감 효과를 보이며, 전체 언어 모델 크기의 0.0027% 수준에서 영향을 추정할 수 있게 되었습니다.
- ***Technical Details***: NN-CIFT는 기존의 LLM이 아닌 소형 신경망 InfluenceNetwork를 사용하여 데이터 영향을 효율적으로 계산합니다. InfluenceNetwork는 2개의 레이어로 구성되며, 각 레이어는 100개의 뉴런을 가지고 있습니다. 이를 통해 점대점(pairwise) 방식의 영향 함수에 대해 훈련하여 데이터 선별 과정을 보다 저렴하게 진행할 수 있습니다. 데이터 선별 알고리즘으로는 하위 모듈러(Submodular) 함수를 사용하여 정보량이 풍부한 소규모 데이터 세트를 선택합니다.
- ***Performance Highlights***: NN-CIFT를 사용한 데이터 유효성 검사 비용은 기존 방법에 비해 77-99% 감소했으며, 성능 감소 없이 비슷한 수준의 성능을 유지합니다. NN-CIFT가 기존의 영향 함수에 비해 평균 성능 차이가 고작 1.40%에 불과하여, 시간과 자원을 절약하며 추가적인 데이터에 대한 재훈련 없이도 일관된 성능을 보여줍니다.

### [Show Me the Work: Fact-Checkers' Requirements for Explainable Automated Fact-Checking](https://arxiv.org/abs/2502.09083)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.09083.png)

Vote: 1

Authors: Irina Shklovski, Isabelle Augenstein, Greta Warren

- ***What's New***: 이 논문은 자동화된 팩트체킹 시스템의 설명 가능성 요구를 이해하기 위해 팩트체커들이 작업 과정을 평가하고, AI 도구의 사용 방식을 조사하며, 설명이 어떻게 팩트체커의 의사결정을 지원할 수 있는지에 대한 통찰을 제공하는 연구를 진행했습니다.
- ***Technical Details***: 기술적인 접근으로는 자연어 처리(Natural Language Processing; NLP)와 인간-컴퓨터 상호작용(Human-Computer Interaction; HCI) 연구자 간의 협업을 통해 세미 구조화된 인터뷰를 설계하여 10명의 팩트체킹 전문가들과 대화를 나누었습니다. 이는 팩트체킹의 각 단계에서 필요한 설명을 식별하는 데에 중점을 두고 있습니다.
- ***Performance Highlights***: 팩트체커들은 AI 도구에 대한 신뢰가 제한적이지만, 이를 통해 시간 소모적인 작업을 줄일 수 있는 잠재력에 대한 긍정적인 견해를 가지고 있습니다. 그럼에도 불구하고 다중 도구 사용 및 AI 도구의 설명 부족은 주요 과제로 남아 있습니다.

### [Better Embeddings with Coupled Adam](https://arxiv.org/abs/2502.08441)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.08441.png)

Vote: 1

Authors: Tobias Stollenwerk, Felix Stollenwerk

- ***What's New***: 본 연구는 Adam 최적화 알고리즘이 단어 임베딩의 비이상성 문제를 유발하는 중요한 요인임을 밝혀내고, 이를 보완하기 위해 수정된 최적화 알고리즘인 Coupled Adam을 제안하고 있습니다.
- ***Technical Details***: Coupled Adam은 원래의 Adam 알고리즘에서 임베딩 파라미터를 특별히 조정하여 비이상성 문제를 완화하도록 설계되었습니다. 이 알고리즘은 임베딩 벡터의 평균값이 기원에서 이동하게 만드는 i-의존적 두 번째 모멘트를 동일하게 만드는 방식으로 장치되었습니다.
- ***Performance Highlights***: Coupled Adam을 활용하면 대규모 데이터셋상에서 모델의 상류 성능과 하류 성능 모두 개선되는 것을 확인할 수 있습니다. 또한 임베딩의 품질이 현저히 향상되며, 이는 임베딩 벡터의 아이소트로피 수치가 0.90 이상으로 증가하는 등 기존 방법에 비해 크게 개선되는 결과를 나타냅니다.

### [ExaGPT: Example-Based Machine-Generated Text Detection for Human Interpretability](https://arxiv.org/abs/2502.11336)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11336.png)

Vote: 0

Authors: Ryuto Koike, Naoaki Okazaki, Preslav Nakov, Masahiro Kaneko, Ayana Niwa

- ***What's New***: ExaGPT는 인간 고유의 의사결정 과정을 토대로한 해석 가능한 탐지 방법을 제안합니다. 이 방법은 텍스트가 인간 작성물과 더 유사한지, 대형 언어 모델(LLM)이 생성한 것과 더 유사한지 확인하여 텍스트의 출처를 판별합니다. 이로써 텍스트의 각 부분에 대한 유사한 예제를 제공하여 해석 가능성을 높입니다.
- ***Technical Details***: ExaGPT는 대상 텍스트의 각 n-gram 간에 비슷한 유사 span 예제를 데이터 저장소에서 검색합니다. 탐지에서 각 span에 대한 최적의 분할을 찾기 위해 동적 프로그래밍(Dynamic Programming)을 적용합니다. 주기는 유사도 점수로 필요한 분할 경계를 결정하여 텍스트를 보다 이해하기 쉽게 만듭니다.
- ***Performance Highlights***: ExaGPT는 학교 과제 등의 도메인과 ChatGPT, GPT-4와 같은 생성자 간의 다양한 실험에서 최대 40.9포인트 이상의 정확도를 보여, 기존 탐지기보다 뛰어난 성능을 나타냈습니다. 1%의 낮은 false positive rate에서 높은 탐지 정확도를 유지하여 실무적 관점에서 효율적인 탐지기임을 입증했습니다.

### [Language Complexity Measurement as a Noisy Zero-Shot Proxy for Evaluating LLM Performance](https://arxiv.org/abs/2502.11578)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.11578.png)

Vote: 0

Authors: Johan Boye, Birger Moell

- ***What's New***: 이 연구는 대형 언어 모델(LLMs)의 성과를 평가하기 위해 언어 복잡성 측정, 특히 LIX 가독성 지수와 평균 종속 거리(ADD)를 사용하는 새로운 접근법을 제안합니다. LIX는 주어진 언어의 복잡성을 평가하는 데 사용되며, ADD는 문장의 구조적 복잡성을 평가합니다.
- ***Technical Details***: 본 연구에서는 여섯 가지 최신 LLMs, 즉 Gemini-1.5-Pro, Gemini-2.0-flash, Llama 70b, Llama 70b 3.3 (Meta), GPT-4o-mini, 그리고 o1-mini (OpenAI)을 평가했습니다. 각 모델은 동일한 프롬프트를 통해 LIX 지수와 의존 구문 분석을 수행하였고, 이러한 결과를 Stanza 라이브러리를 사용해 생성된 그라운드 트루스와 비교했습니다.
- ***Performance Highlights***: o1-mini 모델은 LIX 계산 및 의존 구문 분석에서 가장 높은 정확도를 달성하며, LIX에서 가장 낮은 오류(7.4)와 ADD에서 가장 작은 차이를 보였습니다. LIX 오류와 MMLU 벤치마크 성능 간의 피어슨 상관계수(r)는 -0.875로 매우 강한 음의 상관관계를 보여, 모델의 언어 복잡성 평가 능력이 LLM의 전반적인 능력을 평가하는 유용한 프록시가 될 수 있음을 시사합니다.

### [Sailor2: Sailing in South-East Asia with Inclusive Multilingual LLMs](https://arxiv.org/abs/2502.12982)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.12982.png)

Vote: 0

Authors: Cunxiao Du, Sittipong Sripaisarnmongkol, Wei Lu, Shiqi Chen, Qunshu Lin, Min Lin, Zili Wang, Qian Liu, Zichen Liu, Tongyao Zhu, Penghui Yang, Anh Dao, Fan Zhou, Fajri Koto, Wannaphong Phatthiyaphaibun, Kridtaphad Sae-Khow, Phakphum Artkaew, Haonan Wang, Changyu Chen, Ziqi Jin, Taechawat Konkaew, Matichon Maneegard, Mike Zhang, Man Tsung Yeung, Yongchi Zhao, Longxu Dou, Narong Borijindargoon, Zheng-Xin Yong, Chao Du, Xin Mao, Nirattisai Thongchim, Hoang H. Tran, Kunat Pipatanakul, Min Si Thu, Xinyi Wan, Quan Nguyen, Hynek Kydlíček, Tianyu Pang, Xiachong Feng, Jiaheng Liu, Zeyi Liu

- ***What's New***: Sailor2는 동남아시아 언어를 지원하는 선구적인 멀티모달 LLM(Large Multimodal Models)입니다. 1B, 8B, 20B 크기로 제공되며, 13개의 SEA 언어를 지원하면서 중국어와 영어에 대한 능력을 유지합니다. 추가로, 다국어 모델 구축을 위한 데이터 큐레이션, 지속적 사전 훈련, 맞춤형 모델 등을 포함한 포괄적인 방법론을 제공합니다. Sailor2-20B는 SEA 언어에서 GPT-4o와 경쟁할 만큼 뛰어난 성능을 보여줍니다.
- ***Technical Details***: Sailor2는 Qwen2.5에 기반하여 500B 토큰(400B SEA 전용, 100B 리플레이 토큰)으로 지속적 사전 훈련됩니다. 데이터 큐레이션은 4.8M의 고품질 예시, 6개의 레이어로 필터링을 통해 이루어지며, Supervised Fine-tuning, 모델 확장, 모델 병렬 최적화 등의 기술이 활용됩니다. 이 과정에서 ZB-2P파이프라인 병렬 프로세스와 대형 어휘 최적화 기술이 적용됩니다.
- ***Performance Highlights***: Sailor2-20B 모델은 SEA 언어에서 GPT-4o와의 대결에서 50-50 승률을 달성했습니다. 수많은 언어에서 두드러진 성과를 보이며, 특히 저자원 언어들 사이에서 월등한 성능을 드러냅니다. 이러한 성과는 Sailor2가 SEA 언어 번역과 문화 이해에서 뛰어난 역량을 가지고 있음을 보여줍니다.

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
