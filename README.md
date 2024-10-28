# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using GPT-4o.

Thanks to [@AK391](https://github.com/AK391) for great work.

## Daily Papers (2024-10-28)

### [Are LLMs Better than Reported? Detecting Label Errors and Mitigating Their Effect on Model Performance](https://arxiv.org/abs/2410.18889)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.18889.png)

Vote: 11

Authors: Omer Nahum, Orgad Keller, Idan Szpektor, Nitay Calderon, Roi Reichart

- ***What's New***: 이 연구는 LLM (Large Language Models)의 레이블 오류 탐지와 이러한 오류가 모델 성능에 미치는 영향을 다룹니다. LLM이 기존 데이터셋의 레이블 오류를 감지하는 데 효과적일 수 있으며, 이를 수정하면 모델 성능 향상에 크게 기여할 수 있음을 보여줍니다.
- ***Technical Details***: 본 연구는 LLM을 심판(LLM-as-a-judge)으로 활용하여 다수의 LLM 앙상블을 통해 잠재적으로 잘못 레이블된 사례를 탐지하는 간단한 방법을 제안합니다. 이 접근법은 데이터의 레이블을 재학습하거나 필터링하여 개선하려는 목적으로 사용할 수 있습니다.
- ***Performance Highlights***: 실험에서 LLM은 6%에서 21% 사이의 레이블 오류를 성공적으로 탐지했으며, 높은 정확도로 모델의 성능을 최대 15%까지 개선할 수 있음을 보여주었습니다. 이는 기존 데이터셋에서 많은 예측 오류가 실제 오류가 아닌 레이블 오류임을 시사합니다.

### [Read-ME: Refactorizing LLMs as Router-Decoupled Mixture of Experts with System Co-Design](https://arxiv.org/abs/2410.19123)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.19123.png)

Vote: 12

Authors: Babak Ehteshami Bejnordi, Ruisi Cai, Yeonju Ro, Geon-Woo Kim, Zhangyang Wang, Aditya Akella, Peihao Wang

- ***What's New***: Read-ME는 대형 언어 모델(LLM's)을 더 작은 전문가 혼합 모델(Mixture of Experts; MoE)로 변환하는 새로운 프레임워크를 제안했습니다. 이 방법은 기존에 학습된 밀집 모델을 재활용, 부작용 없이 시스템 친화적인 설계로 변환하여 최신 MoE 모델에서 발생하는 비효율성과 고비용 문제를 해결합니다.
- ***Technical Details***: Read-ME는 활성화 희소성(Activation Sparsity)을 이용해 전문가를 추출하며, 레이어별 라우터 디자인의 중복성을 분석하여 이를 전문가 뒷단과 분리된 사전 게이트(pre-gating) 라우터로 대치합니다. 이 구조는 최적의 배치 및 캐싱을 가능하게 하여 리소스 제약 환경에서 효율적이면서 확장 가능한 LLM 추론을 지원합니다.
- ***Performance Highlights***: Read-ME는 MMLU에서 최대 10.1%의 성능 향상을 기록하였고, 평균 종단간 지연 시간을 최대 6.1% 개선하였습니다. 코드들은 https://github.com/VITA-Group/READ-ME. 에서 확인할 수 있습니다.

### [Mapping the Media Landscape: Predicting Factual Reporting and Political Bias Through Web Interactions](https://arxiv.org/abs/2410.17655)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.17655.png)

Vote: 4

Authors: Petr Motlicek, Dairazalia Sánchez-Cortés, Esaú Villatoro-Tello, Sergio Burdisso

- ***What's New***: 이 연구는 긴 시간 축에 걸쳐 뉴스 미디어 소스의 프로파일링을 개선하는 방법을 제시합니다. 웹 상의 하이퍼링크 상호작용을 기반으로 뉴스 미디어의 사실성 보고(Factual Reporting)과 정치 편향(Political Bias)을 보다 정교하게 예측합니다.
- ***Technical Details***: 거대 뉴스 미디어 하이퍼링크 그래프를 이용하며, 강화 학습(Reinforcement Learning) 전략을 통해 소스 미디어의 정확성을 평가합니다. 정확한 이웃 소스와의 상호작용을 통해 뉴스 미디어의 편향성을 추론할 수 있음을 보여줍니다. 데이터셋은 MBFC와 CLEF 2023 CheckThat! Lab를 사용하며, P-프로퍼티(P-property), I-프로퍼티(I-property) 등 여러 전략을 적용하여 정치 편향 및 사실성 보고를 예측합니다.
- ***Performance Highlights***: MBFC 데이터셋에서 I-프로퍼티 전략이 77.77%의 F1 점수를 기록하며 가장 높은 성능을 보였습니다. CLEF 2023 CheckThat! Lab에서는 정치 편향 예측에서 최고 성능을 기록해, F1 점수 0.784와 MAE 0.214를 달성했습니다. 이는 다른 참가 모델들의 성능을 능가합니다.

### [Continuous Speech Synthesis using per-token Latent Diffusion](https://arxiv.org/abs/2410.16048)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.16048.png)

Vote: 23

Authors: David Haws, Hagai Aronowitz, Ron Hoory, Nimrod Shabtay, Avihu Dekel, Slava Shechtman, Arnon Turetzky

- ***What's New***: 이 논문에서는 새로운 연속형 음성 합성 방법론인 SALAD (Speech synthesis with Autoregressive LAtent Diffusion)를 소개합니다. 이는 토큰별로 잠재적 확산 모델(per-token latent diffusion model)을 사용하여 제로-샷 방식의 텍스트-음성 변환(zero-shot text-to-speech)을 수행하며, 연속적인 표현에서 작동합니다.
- ***Technical Details***: SALAD는 연속적인 음향 특성을 생성하기 위한 세 가지 변형(S2A-AR, S2A-NAR, T2A)을 제안합니다. S2A-AR은 시맨틱 토큰에서 다음 토큰을 예측하여 음향 특징을 생성하고, S2A-NAR은 MaskGIT 일정에 따라 음향 특징을 생성합니다. T2A는 텍스트에서 직접 시맨틱 및 음향 토큰을 병렬로 예측하는 모델입니다. 이와 병행하여, SALAD는 VAE(Variational Autoencoder) 잠재 토큰을 연속적으로 예측하고, 확산 헤드(diffusion head)를 사용하여 다중모드 분포를 모델링합니다.
- ***Performance Highlights***: 성과 측면에서 SALAD의 T2A 모델은 원본 오디오와 비교할 때 뛰어난 직관성 점수를 달성하였으며, 음질 및 화자 유사성도 원본 오디오와 동등한 수준입니다. 연속 모델은 특히 자모 음성 명료성에서 높은 점수를 기록하였으며, 이는 특정 텍스트를 정확하게 합성해야 할 때 가장 신뢰할 수 있는 모델임을 시사합니다. 반면, 화자 유사성 측면에서는 이산 T2A 및 S2A-NAR 모델이 더 높은 점수를 기록했습니다.

### [Dynamic 3D Gaussian Tracking for Graph-Based Neural Dynamics Modeling](https://arxiv.org/abs/2410.18912)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.18912.png)

Vote: 3

Authors: Kaifeng Zhang, Yunzhu Li, Mingtong Zhang

- ***What's New***: 이번 연구에서는 다중 뷰 RGB 비디오에서 직접 객체의 동적인 3D 상태를 추적하고 예측할 수 있는 새로운 프레임워크를 제시했습니다. 이는 로봇의 행동 궤적을 명시적으로 고려하여 장면의 동역학을 학습하도록 설계되어 있으며, 3D Gaussian Splatting을 사용해 객체의 동적 변화를 모델링합니다.
- ***Technical Details***: 이 프레임워크는 다이나믹 3D Gaussian Splatting을 통해 얻은 3D Gaussian 입자를 기반으로 그래프 신경망(Graph Neural Network)을 훈련시킵니다. 이는 움직임 제어 입자에서 예측된 3D 변형을 보간하여 객체의 미래 상태를 렌더링하고, 행동 조건부 비디오 예측(Action-Conditioned Video Prediction)을 수행할 수 있습니다.
- ***Performance Highlights***: 로프, 옷, 인형과 같은 다양한 변형 가능한 물체에서 실험한 결과, 제안된 방법은 밀도 있기 3D Gaussian 추적에서 뛰어난 성능을 보였으며, occlusion 상황에서도 정밀한 일치가 유지되었습니다. 예측된 다이나믹스 모델은 물체의 물리적 특성을 사실적으로 시뮬레이션하며, 기존의 파라미터 최적화 기반 시스템 식별 접근법보다 더 정확한 운동 예측 성능을 보였습니다.

### [Analysing the Residual Stream of Language Models Under Knowledge Conflicts](https://arxiv.org/abs/2410.16090)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.16090.png)

Vote: 5

Authors: Xuanli He, Aryo Pradipta Gema, Hongru Wang, Yu Zhao, Xiaotang Du, Alessio Devoto, Pasquale Minervini, Giwon Hong, Kam-Fai Wong

- ***What's New***: 이 논문에서는 대형 언어 모델(LLMs)이 지식 충돌 상황에서의 잔여 스트림(residual stream)을 분석하는 방법을 제시합니다. 이는 외부로부터 제공되는 컨텍스트 지식과 모델 내부의 매개변수 지식(parametric knowledge) 간의 충돌을 탐지할 수 있는 방법을 통해 LLMs 내부에서 지식 갈등을 처리하는 방식을 이해하려는 시도입니다.
- ***Technical Details***: LLMs의 중간 레이어(예: Llama3-8B의 13th 레이어)에 있는 잔여 스트림의 패턴을 통해 지식 갈등의 신호를 감지할 수 있다는 사실을 밝혀냈습니다. 이러한 신호를 활용하여 로지스틱 회귀(Logistic Regression) 모델로 90%의 정확도로 갈등을 탐지할 수 있으며, 입력이나 모델 매개변수를 수정하지 않고도 가능했습니다. 또한, 모델이 컨텍스트 지식을 활용할 때와 매개변수 지식을 활용할 때 잔여 스트림이 다르게 표현된다는 것을 관찰했습니다.
- ***Performance Highlights***: 잔여 스트림에서 지식의 출처에 따른 다른 패턴을 보임으로써 LLMs가 지식 갈등 상태에서 어떤 출처의 지식을 사용할지를 미리 예측할 수 있습니다. 이는 의도치 않은 응답 생성 문제를 사전에 완화할 수 있는 가능성을 보여줍니다. 이에 따라 LLMs의 성능을 효율적으로 제어하는 데 기여할 수 있습니다.

### [Counting Ability of Large Language Models and Impact of Tokenization](https://arxiv.org/abs/2410.19730)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.19730.png)

Vote: 5

Authors: Chenyu You, Juntai Cao, Xiang Zhang

- ***What's New***: 이 논문은 대형 언어 모델(Large Language Models; LLMs)에서 Tokenization이 Numeric Reasoning, 특히 Counting Task에 영향을 미치는 방식을 조사합니다. 이전 연구에서는 LLM의 일반적인 토큰화 방법이 Model’s Reasoning Ability를 손상시킬 수 있음을 이론적으로 분석했으나, 이를 뒷받침하는 실험적인 증명은 부족했습니다. 이 논문은 이러한 Gap을 채우고자 실험 결과와 새로운 토큰화 전략 제안을 포함합니다.
- ***Technical Details***: 일반 LLM은 Byte-Pair Encoding(BPE) 토크나이저를 사용하여 여러 문자를 하나의 토큰으로 그룹화합니다. 연구팀은 LLM이 Counting을 수행할 때 Tokenization 전략의 중요성을 강조하고, Model-Agnostic Approach를 통해 LLM의 Counting 능력을 실험적으로 분석하였습니다. 여러 Counting Experiment를 통해 CoT (Chain of Thought)가 Tokenization 선택에 따라 다양한 성능 변화를 가져옴을 보여줍니다.
- ***Performance Highlights***: CoT를 적용한 LLM에서 Tokenization이 올바르게 선택될 경우 Counting 정확도가 큰 폭으로 향상되었습니다. 잘못된 Tokenization에서는 최대 80%의 정확도 손실이 발생했으며, 정확한 Tokenization에서는 실험 결과 정확도가 크게 향상되어 이론적 상한에 도달할 수 있게 되었습니다. 특히 Tokenization 유형에 따라 Counting 오류의 양상이 크게 달라지는 것을 발견하였습니다.

### [Reflection-Bench: probing AI intelligence with reflection](https://arxiv.org/abs/2410.16270)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.16270.png)

Vote: 3

Authors: Haiquan Zhao, Yan Teng, Lingyu Li, Yingchun Wang, Chunbo Li, Yixu Wang, Shuqi Kong

- ***What's New***: Reflection-Bench는 AI 시스템의 지능을 반영(reflection)의 과정으로 평가하는 처음이자 포괄적인 벤치마크입니다. 이 벤치마크는 LLM(Large Language Models)의 반영 능력을 평가하기 위해 지각(perception), 기억(memory), 믿음 업데이트(belief updating), 의사결정(Decision-Making), 예측(Prediction), 반사적 사고(Counterfactual Thinking) 및 메타 반영(meta-reflection)과 같은 핵심 인지 기능을 포함한 7개의 과제로 구성되어 있습니다.
- ***Technical Details***: Reflection-Bench는 인지 과학의 확립된 패러다임을 기반으로 하여 6개의 패러다임과 7개의 과제를 설계했습니다. 이 과제들은 LLM의 평가를 위해 설계되었으며, 'oddball paradigm', 'n-back task', 'probabilistic reversal learning task', 'Wisconsin card sorting test', 'weather prediction task', 'double-choice Iowa gambling task', 및 'meta-bandit task'를 포함합니다. 반영 벤치마크는 다양한 측면을 포괄적으로 평가하며, 더 높은 수준의 AI 모델을 수용할 수 있도록 난이도를 조정할 수 있습니다.
- ***Performance Highlights***: 13개의 다양한 LLM을 반영 벤치마크로 평가한 결과, o1-preview 모델이 가장 높은 점수를 기록하며 우수한 성능을 보였으나 모든 모델에서 메타 반영 능력의 부재를 관찰했습니다. 또한, o1-preview는 상대적으로 낮은 MMN 반응 점수를 받아 의외의 정보에 대한 즉각적인 인식이 부족한 것으로 나타났습니다. 이는 현재 LLM의 인간 수준 반영 능력이 크게 부족함을 나타냅니다.

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
