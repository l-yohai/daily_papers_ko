# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using ChatGPT.

Thanks to [@AK391](https://github.com/AK391) for great work.


## Daily Papers (2024-04-19)

### [Reka Core, Flash, and Edge: A Series of Powerful Multimodal Language Models](https://arxiv.org/abs/2404.12387)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2404.12387.png)

Vote: 5

Authors: Nishant Relan, Ren Chen, Matthew Henderson, Cyprien de Masson d'Autume, Isaac Ong, Eugenie Lamprecht, Donovan Ong, Piotr Padlewski, Eric Chen, +, Max Bain, Yazheng Yang, Qi Liu, Hai Pham, Lei Li, Che Zheng, Deyu Fu, Samuel Phua, Mikel Artetxe, Kaloyan Aleksiev, Dani Yogatama, Yi Tay, Aitor Ormazabal

- 렉카(Reka)사가 처음부터 직접 훈련시킨 강력한 다중 모드 언어 모델인 렉카 코어(Reka Core), 렉카 플래시(Reka Flash), 그리고 렉카 엣지(Reka Edge)를 소개합니다.
- 이 모델들은 텍스트, 이미지, 비디오, 오디오 입력을 처리하고 이해할 수 있는 능력을 갖추고 있습니다.
- 기술 보고서에서는 일부 모델의 훈련 세부사항을 논의하고 종합적인 평가 결과를 제공합니다.
- 렉카 엣지와 렉카 플래시는 최신 기술을 능가할 뿐만 아니라, 훨씬 큰 모델들보다 우수한 성능을 보이며 해당 계산 클래스에서 뛰어난 가치를 제공합니다.
- 가장 능력이 뛰어나고 큰 모델인 렉카 코어는 자동 평가 및 블라인드 인간 평가에서 최고의 프론티어 모델에 근접합니다.
- 이미지 질문 응답 벤치마크에서는 GPT4-V와 경쟁적인 성능을 보이며, 다중 모드 채팅에서는 블라인드 제3자 인간 평가 설정에서 선호도가 두 번째로 높은 모델로 평가되었습니다.
- 텍스트 벤치마크에서는 다른 프론티어 모델들과 경쟁적인 성능을 보여주며, 인간 평가에서 GPT4-0613을 능가합니다.
- 비디오 질문 응답에서는 젬나이 울트라(Gemini Ultra)를 능가하는 성능을 보입니다.
- 모델은 http://chat.reka.ai에서 제공되며, 선택되지 않은 질적 예시들은 http://showcase.reka.ai에서 확인할 수 있습니다.

### [Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing](https://arxiv.org/abs/2404.12253)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2404.12253.png)

Vote: 3

Authors: Linfeng Song, Baolin Peng, Ye Tian, Dian Yu, Lifeng Jin, Dong Yu, Haitao Mi

- 대규모 언어 모델(LLMs)은 복잡한 추론과 계획을 포함하는 시나리오에서 여전히 어려움을 겪고 있으며, 최근 연구에서는 고품질 데이터를 사용한 미세조정과 고급 프롬프팅 기술을 제안하였습니다.
- 데이터의 가용성과 품질에 의해 제한되는 기존 접근법들을 넘어서 자기 개선과 자기 학습이 더 효율적인 대안으로 부상하고 있습니다.
- 본 논문에서는 LLM의 자기 개선을 위해 몬테카를로 트리 검색(MCTS)을 통합한 AlphaLLM을 소개하며, 이는 추가적인 주석 없이 LLM의 능력을 향상시키는 자기 개선 루프를 구축합니다.
- AlphaGo의 성공에서 영감을 받은 AlphaLLM은 데이터 부족, 언어 작업의 방대한 검색 공간, 그리고 언어 작업의 주관적인 피드백 등의 독특한 도전을 다룹니다.
- AlphaLLM은 프롬프트 합성 구성 요소, 언어 작업에 맞춘 효율적인 MCTS 접근 방식, 그리고 정확한 피드백을 제공하는 세 가지 비평 모델로 구성되어 있습니다.
- 수학적 추론 작업에서의 실험 결과는 추가 주석 없이도 AlphaLLM이 LLM의 성능을 크게 향상시킬 수 있음을 보여주며, LLM의 자기개선 가능성을 시사합니다.

### [BLINK: Multimodal Large Language Models Can See but Not Perceive](https://arxiv.org/abs/2404.12390)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2404.12390.png)

Vote: 3

Authors: Haoyu Wang, Bangzheng Li, Yushi Hu, Noah A. Smith, Dan Roth, Wei-Chiu Ma, Ranjay Krishna, Yu Feng, Xudong Lin, Xingyu Fu

- 'BLINK' 벤치마크는 다양한 모달을 사용하는 언어 모델(LLM)의 기본 시각적 인식 능력을 평가하기 위해 소개되었습니다.
- 이 벤치마크는 사람들이 매우 빠르게 해결할 수 있는 여러 작업(예: 상대적 깊이 추정, 시각적 대응, 포렌식 탐지, 다중 뷰 추론)을 포함하고 있습니다.
- BLINK는 14가지 전통적인 컴퓨터 비전 작업을 3,807개의 다중 선택형 문제로 재구성하였으며, 이는 단일 또는 여러 이미지와 시각적 유도를 포함합니다.
- 인간은 평균적으로 95.70%의 정확도를 보였지만, 최고의 다중 모달 LLM인 GPT-4V와 Gemini조차 각각 51.26%와 45.72%의 정확도만을 보여, 문제 해결에 상당한 도전을 받고 있음을 나타냈습니다.
- 이 결과는 시각적 인지 능력이 최근의 다양한 모달 LLM에서 아직 "등장하지 않았음"을 시사하며, 전문 컴퓨터 비전 모델이 이러한 문제를 훨씬 더 잘 해결할 수 있음을 강조합니다.
- BLINK가 시각적 인식에서 인간 수준에 도달하기 위한 다양한 모달 LLM의 개선을 촉진할 것으로 기대됩니다.

### [AniClipart: Clipart Animation with Text-to-Video Priors](https://arxiv.org/abs/2404.12347)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2404.12347.png)

Vote: 3

Authors: Kede Ma, Wanchao Su, Ronghuan Wu, Jing Liao

- 클립아트는 이미지를 애니메이션으로 변환하기 위한 기존의 복잡한 작업 프로세스에 비해 효율적인 시각 콘텐츠를 제공합니다.
- 최근의 텍스트-비디오 생성 기술은 클립아트의 시각적 정체성을 유지하면서 만화 스타일의 움직임을 생성하는 데에 어려움을 겪고 있습니다.
- 본 논문에서는 클립아트 이미지를 고품질의 동작 시퀀스로 변환하는 'AniClipart' 시스템을 소개합니다.
- AniClipart는 베지어 곡선과 키포인트를 이용한 움직임 정규화와 텍스트 프롬프트에 맞춰 키포인트의 움직임 경로를 최적화합니다.
- 비디오 스코어 증류 샘플링(VSDS) 손실을 최적화하여 자연스러운 움직임의 지식을 인코딩하고, 비교적 강건한 형태 변형 알고리즘을 사용합니다.
- 실험 결과, AniClipart는 텍스트-비디오 정렬, 시각적 정체성 유지, 그리고 움직임 일관성 면에서 기존 이미지-비디오 생성 모델들을 일관되게 능가합니다.
- 또한 AniClipart는 위상 변화를 허용하는 계층적 애니메이션과 같은 다양한 애니메이션 형식을 생성할 수 있는 유연성을 보여줍니다.

### [MoA: Mixture-of-Attention for Subject-Context Disentanglement in Personalized Image Generation](https://arxiv.org/abs/2404.11565)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2404.11565.png)

Vote: 2

Authors: Wang, Yuwei Fang, Kuan-Chieh, Daniil Ostashev, Kfir Aberman, Sergey Tulyakov

- 텍스트-이미지 확산 모델의 개인화를 위한 새로운 구조인 Mixture-of-Attention(MoA)을 소개합니다.
- MoA는 큰 언어 모델에서 사용되는 전문가 혼합 메커니즘에서 영감을 받아 개인화된 경로와 비개인화된 사전 경로라는 두 개의 주의 경로 사이에서 생성 작업을 분배합니다.
- 이 모델은 사전 경로의 주의 층을 고정함으로써 원래 모델의 사전을 유지하면서, 개인화된 경로가 배치와 맥락을 생성하는 주제를 포함하도록 학습합니다.
- 새로운 라우팅 메커니즘은 이러한 경로들을 통한 개인화된 및 일반적인 컨텐츠 생성의 최적 혼합을 위해 각 층에서 픽셀의 배분을 관리합니다.
- 학습이 완료되면 MoA는 다중 주제가 포함된 고품질, 개인화된 이미지를 생성할 수 있게 해주며, 구성과 상호 작용이 기존 모델에 의해 생성된 것만큼 다양합니다.
- 중요한 점은, MoA가 모델의 기존 능력과 새롭게 추가된 개인화된 개입 사이의 구별을 향상시켜 이전에는 달성할 수 없었던 더 해체된 주제-맥락 제어를 제공한다는 것입니다.
- 프로젝트 페이지: https://snap-research.github.io/mixture-of-attention

### [TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding](https://arxiv.org/abs/2404.11912)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2404.11912.png)

Vote: 2

Authors: Yuandong Tian, Hanshi Sun, Zhuoming Chen, Xinyu Yang, Beidi Chen

- 대규모 언어 모델(LLMs)이 장문의 콘텐츠 생성에 널리 사용됨에 따라 효율적인 장문 시퀀스 추론 지원에 대한 수요가 증가하고 있다.
- 기존에는 키-값(KV) 캐시를 사용하여 재계산을 방지했으나, 시퀀스 길이와 함께 선형적으로 증가하는 캐시 크기가 병목 현상을 일으키는 주요한 문제로 부상했다.
- TriForce는 계층적 추론적 디코딩을 사용하여 장문 시퀀스 생성을 가능하게 하는 시스템으로, 동적인 스파스 KV 캐시를 통해 기존 모델의 가중치를 활용하고 초안 모델로서 작동한다.
- TriForce는 리트리시브(회복)과 정밀한 모델을 통해 초안 작성 지연 시간을 줄이며, Llama2-7B-128K에서 최대 2.31배의 속도 향상을 달성하였다.
- 더 긴 문맥을 처리할 때의 확장성을 보여주며, 두 개의 RTX 4090 GPU에서 0.108초/토큰의 성능을 나타내어 기존의 자기회귀 기반 벤치마크 대비 훨씬 뛰어난 성능을 제공한다.
- TriForce는 다양한 환경에서 일관되게 뛰어난 성능을 유지함을 보여주며, 해당 코드는 온라인에서 접근 가능하다.

### [Reuse Your Rewards: Reward Model Transfer for Zero-Shot Cross-Lingual Alignment](https://arxiv.org/abs/2404.12318)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2404.12318.png)

Vote: 1

Authors: Ahmad Beirami, Ananth Balashankar, Jacob Eisenstein, Yoon Kim, Zhaofeng Wu

- 이 연구는 한 소스 언어의 선호 데이터에서 훈련된 보상 모델을 다른 대상 언어에 직접 적용하는 제로샷 교차 언어 정렬을 평가합니다.
- 요약 및 개방형 대화 생성과 같은 작업에서, 이 방법은 포괄적인 평가 설정 하에서 일관되게 성공적임을 보여주고 있으며, 인간 평가도 포함됩니다.
- 교차 언어로 정렬된 모델들은 평가 인스턴스의 70% 이상에서 정렬되지 않은 모델보다 사람들에게 선호됩니다.
- 때로는 다른 언어의 보상 모델이 같은 언어의 보상 모델보다 더 잘 정렬된 모델을 생성할 수 있음을 발견했습니다.
- 언어 특정 데이터가 전혀 없는 경우에도 최적의 실천 방법을 확인합니다.

### [Introducing v0.5 of the AI Safety Benchmark from MLCommons](https://arxiv.org/abs/2404.12241)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2404.12241.png)

Vote: 1

Authors: Elie Alhajjar, Adarsh Agrawal, Leon Derczynski, Cody Coleman, Debojyoti Dutta, Ahmed M. Ahmed, James Ezick, Siméon Campos, Namir Al-Nuaimi, Zacharie Delpierre Coudert, +, Najla Alfaraj, Kal Chakra, Trupti Bavalatti, Borhane Blili-Hamelin, Bertie Vidgen, Kurt Bollacker, Ian Eisenberg, Lora Aroyo, Victor Akinwande, Canyu Chen, Marisa Ferrara Boston, Rishi Bomassani

- 이 논문은 MLCommons AI 안전 작업 그룹이 만든 인공지능 안전 벤치마크 v0.5를 소개합니다.
- 이 벤치마크는 채팅 튜닝된 언어 모델을 사용하는 AI 시스템의 안전 위험을 평가하도록 설계되었습니다.
- v0.5는 영어로 일반 목적의 보조와 대화하는 성인 사용 사례 한 가지와 제한된 세트의 페르소나(일반 사용자, 악의적 사용자 및 취약 사용자)만을 다룹니다.
- 13개의 위험 범주 분류를 새롭게 만들었으며, 그 중 7개 범주에 대한 테스트가 v0.5 벤치마크에 포함되어 있습니다.
- 총 43,090개의 테스트 항목이 포함된 테스트는 템플릿을 사용하여 생성되었습니다.
- 벤치마크에는 사용 사례, 시스템 유형, 언어 및 맥락, 페르소나, 테스트 및 테스트 항목을 명시하고 구성하는 원칙적 접근 방식이 포함됩니다.
- 이 벤치마크를 통해 AI 시스템의 안전성을 평가할 수 있는 공개 플랫폼 및 다운로드 가능한 도구인 ModelBench가 제공됩니다.
- 수십 개의 공개적으로 이용 가능한 채팅 조정 언어 모델의 성능을 벤치마킹하는 예시 평가 보고서도 제공됩니다.
- AI 안전 벤치마크 v1.0은 2024년 말까지 출시될 예정이며, AI 시스템의 안전성에 대한 의미 있는 통찰력을 제공할 것입니다.
- 현재 v0.5 벤치마크는 AI 시스템의 안전성 평가에 사용되어서는 안 되며 그 한계와 문제점이 철저히 문서화되어 있습니다.

### [OpenBezoar: Small, Cost-Effective and Open Models Trained on Mixes of Instruction Data](https://arxiv.org/abs/2404.12195)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2404.12195.png)

Vote: 1

Authors: Lahiru Lowe, Sachith Gunasekara, Yasiru Ratnayake, Chandeepa Dissanayake

- 기존 LLM들의 지시적 미세조정을 통해 다양한 하위 작업에서 뛰어난 성공을 보여주었으며, 이는 학계와 실무자 모두의 관심을 모으고 있습니다.
- 본 연구에서는 OpenLLaMA 3Bv2를 기반 모델로 사용하여 OpenBezoar 모델군을 미세조정하는 방법을 설명합니다.
- 먼저 Falcon-40B 모델의 지시적 미세조정 변형을 사용하여, 세 가지 구성(라미니 LM, 위자드 LM/Evol-Instruct, 오르카) 따라 합성 지시 미세조정 데이터를 생성하고 GPT-4를 사용하여 이러한 결과를 필터링합니다.
- 이후, QLoRA 기반의 비용 효율적인 감독 미세조정을 연속적으로 수행하고, HH-RLHF 데이터셋의 하위 집합으로 추가 미세조정을 거쳐 DPO 손실을 적용하여 최종 체크포인트를 얻습니다.
- 최종 체크포인트인 "OpenBezoar-HH-RLHF-DPO"는 3B 파라미터 크기의 여러 모델보다 우수한 성능을 보이며, Huggingface Open LLM 리더보드의 한 카테고리에서 최고 모델을 능가합니다.
- 연구 결과는 "OpenBezoar-SFT", "OpenBezoar-HH-RLHF-SFT", "OpenBezoar-HH-RLHF-DPO" 체크포인트와 생성된 데이터셋을 HuggingFace에서 공개하고, 코드베이스는 Bitbucket에서 제공합니다.



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
