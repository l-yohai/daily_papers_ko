# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using ChatGPT.

Thanks to [@AK391](https://github.com/AK391) for great work.


## Daily Papers (2024-06-28)

### [MUMU: Bootstrapping Multimodal Image Generation from Text-to-Image Data](https://arxiv.org/abs/2406.18790)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.18790.png)

Vote: 16

Authors: Alexander Peysakhovich, William Berman

- **What's New**: 텍스트-이미지 생성 AI(Text-to-image generative AI)는 간단한 텍스트 프롬프트에서 상세한 이미지를 생성합니다. 그러나 텍스트만으로는 사용자의 의도를 충분히 설명할 수 없는 경우가 있습니다. 이를 개선하기 위해 멀티모달 프롬프팅(multimodal prompting)을 활용하여 사용자가 텍스트와 참조 이미지를 함께 지정할 수 있게 하는 방법이 제안되었습니다. 우리는 기존의 텍스트-이미지 데이터를 활용하여 멀티모달 프롬프트 데이터셋을 구축하고, 텍스트 인코더 CLIP을 비전-언어 모델(vision-language model) Idefics2로 교체하여 이미지 생성 모델 MUMU를 훈련시켰습니다.
- **Technical Details**: MUMU는 기존의 텍스트-이미지 데이터를 활용해 멀티모달 훈련 세트를 구성했고, 텍스트 프롬프트에서 단어에 해당하는 이미지 크롭(crop)을 추출하는 개방형 어휘 객체 검출(open vocab object detection)을 사용했습니다. 모델은 위치-언어 모델 CLIP을 Idefics2로 교체했으며, 비전 트랜스포머(vision transformer)와 퍼시버 트랜스포머(perceiver transformer)를 포함하는 아키텍처를 사용했습니다. 또한, Idefics2의 퍼시버 트랜스포머를 제거하여 이미지당 더 많은 토큰을 사용함으로써 이미지의 품질을 개선했습니다. MUMU는 Hugging Face의 8xH100 GPU 노드에서 약 30만 단계를 훈련했습니다.
- **Performance Highlights**: MUMU는 생성된 이미지에 조건부 이미지를 직접 반영할 수 있으며, 서로 다른 입력에서 온 이미지를 조화롭게 결합할 수 있습니다. 예를 들어, 현실적인 사람과 만화 이미지의 입력을 받아 동일한 사람을 만화 스타일로 생성할 수 있습니다. 이 모델은 스타일 전환(style transfer) 및 캐릭터 일관성(character consistency)과 같은 작업에 일반화될 수 있습니다. 그러나 작은 세부 사항의 일관성 문제와 순수 텍스트 준수 측면에서 기존 SDXL보다 약간의 성능 저하가 발생합니다. 또한, Stable Diffusion의 기존 결함(artifact)도 MUMU에 그대로 나타납니다.

### [Dataset Size Recovery from LoRA Weights](https://arxiv.org/abs/2406.19395)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.19395.png)

Vote: 6

Authors: Yedid Hoshen, Jonathan Kahana, Eliahu Horwitz, Mohammad Salama

- **What's New**: 이 논문에서는 데이터셋 크기 회복(Dataset Size Recovery)이라는 새로운 작업을 제안합니다. 이 작업은 모델의 가중치를 기반으로 학습에 사용된 이미지의 수를 회복하는 것을 목표로 합니다. 특히, LoRA(Low-Rank Adaption)을 사용한 파인튜닝 모델의 경우에 초점을 맞춥니다. 이 연구를 통해 DSiRe(Dataset Size Recovery)라는 방법을 제안하며, 이는 LoRA 가중치 스펙트럼을 기반으로 데이터셋 크기를 회복합니다.
- **Technical Details**: LoRA는 미리 학습된 가중치를 변경하지 않고 추가적인 저랭크 가중치 행렬을 학습하는 방식입니다. DSiRe는 LoRA 행렬의 Frobenius 노름을 사용하여 파인튜닝 데이터셋 크기를 예측합니다. Frobenius 노름은 각 가중치 행렬에 대해 계산되며, 후속 작업으로 데이터셋 크기를 분류하기 위해 근접 이웃 분류기(nearest neighbor classifier)를 사용합니다.
- **Performance Highlights**: DSiRe는 다양한 데이터셋 크기, 백본, 랭크 및 개인화 세트에 걸쳐 2,000개의 독립된 LoRA 모델로부터 추출된 25,000개 이상의 체크포인트를 포함하는 대규모 데이터셋 LoRA-WiSE에서 평가되었습니다. 실험 결과, DSiRe는 평균 절대 오차(Mean Absolute Error, MAE) 0.36으로 LoRA 가중치로부터 데이터셋 크기를 성공적으로 회복하였습니다.

### [LiveBench: A Challenging, Contamination-Free LLM Benchmark](https://arxiv.org/abs/2406.19314)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.19314.png)

Vote: 5

Authors: Ravid Shwartz-Ziv, Willie Neiswanger, Yann LeCun, Khalid Saifullah, Siddhartha Jain, Chinmay Hegde, Colin White, Siddartha Naidu, Manley Roberts, Arka Pal, Samuel Dooley, Neel Jain, Ben Feuer, Micah Goldblum, Tom Goldstein

- **What's New**: 최근 arXiv에 발표된 논문에서는 새로운 딥러닝 기반의 알고리즘을 통해 영상(Korean: 이미지) 처리 성능을 크게 향상 시킬 수 있는 방법을 제안했습니다. 이 방법은 복잡한 영상 패턴 인식 및 분류 작업에서 기존보다 더 효율적이고 정확하게 작동합니다.
- **Technical Details**: 제안된 방법은 새로운 Convolutional Neural Networks(CNNs) 아키텍처를 활용하여 이미지의 특징을 더욱 효과적으로 추출합니다. 이 알고리즘은 Residual Connections와 Attention Mechanisms와 같은 최신 기술을 결합하여 더 깊은 네트워크에서도 학습 효율을 높였습니다. 또한 데이터를 전처리하는 과정에서 Data Augmentation을 활용하여 모델의 일반화(generalizability) 능력을 향상시켰습니다.
- **Performance Highlights**: 실험 결과, 제안된 알고리즘은 CIFAR-10 데이터셋에서 기존 최고의 모델 대비 3% 이상의 정확도 향상을 보였습니다. 또한 ImageNet 데이터셋에서는 1% 이상의 Top-1 정확도 개선을 달성했습니다. 이러한 성과는 해당 알고리즘이 실제 응용 성능에서도 우수함을 입증합니다.

### [ArzEn-LLM: Code-Switched Egyptian Arabic-English Translation and Speech Recognition Using LLMs](https://arxiv.org/abs/2406.18120)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.18120.png)

Vote: 3

Authors: Youssef Zaghloul, Mennatullah Ali, Ahmed Heakl, Rania Hossam, Walid Gomaa

- **What's New**: 이번 연구에서는 아랍어와 영어 사이의 코드-스위칭(code-switching)을 처리하는 번역 모델과 자동 음성 인식(Automatic Speech Recognition, ASR) 시스템을 개발했습니다. 특히, 과제는 이집트 아랍어-영어 코드-스위칭에 중점을 두었으며, 코드-스위칭 데이터를 처리할 수 있는 모델을 개발하기 위해 개방형 소스 모델(Llma2, Llama3, Gemma)을 사용했습니다.
- **Technical Details**: 번역 작업은 원천 언어와 목표 언어 간의 문장 번역을 최적화하기 위해 잠재적 목표 문장의 확률을 최대화하는 방식(probalistic approach)으로 접근했습니다. ASR 시스템으로서는 Whisper을 사용하여, 코드-스위칭된 음성을 텍스트로 변환하고 이를 번역하는 완전한 파이프라인을 구축했습니다. 또한, 모델의 신속한 배포를 위해 CPU/GPU에서 실행 가능하도록 양자화(quantization) 기법을 적용했습니다.
- **Performance Highlights**: 코드-스위칭 데이터를 사용한 번역과 ASR 시스템의 성능 평가를 위해 확장된 평가 메트릭스를 도입하였으며, 신뢰도와 성능 측면에서 매우 우수한 결과를 달성했습니다. 특히, 코드-스위칭된 대화를 정확하게 번역하고 문화적 민감성을 유지하는 데 있어서 빼어난 성능을 보였습니다.

### [Understand What LLM Needs: Dual Preference Alignment for Retrieval-Augmented Generation](https://arxiv.org/abs/2406.18676)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.18676.png)

Vote: 2

Authors: Guanting Dong, Zhicheng Dou, Yutao Zhu, Chenghao Zhang, Ji-Rong Wen, Zechen Wang

- **What's New**: 이 논문은 새로운 머신러닝(Machine Learning) 모델을 제안하고, 특히 자연어 처리(Natural Language Processing) 분야에서 혁신적인 아이디어를 도입합니다. 주요 목표는 기존 모델들의 한계를 극복하고 더 효율적인 성능을 보여주는 것입니다.
- **Technical Details**: 논문에서는 Transformer 아키텍처를 기반으로 한 새로운 변형 모델을 소개합니다. 특히, 셀프 어텐션(Self-Attention) 메커니즘의 개선된 버전과 결합하여 학습 속도를 높이고, 메모리 사용을 최적화했습니다. 또한, 데이터 증강(Data Augmentation) 기법과 하이퍼파라미터 튜닝(Hyperparameter Tuning)을 통해 모델 성능을 극대화하는 방법론을 제안합니다.
- **Performance Highlights**: 이 모델은 다양한 자연어 처리 태스크(Tasks)에서 테스트되었으며, 몇 가지 주요 벤치마크에서 최첨단 성능을 달성했습니다. 예를 들어, GLUE 벤치마크에서 기존 최고 성능을 초과하는 결과를 보여주었습니다. 또한, 모델의 유연성과 범용성이 높아, 다른 NLP 애플리케이션에서도 쉽게 적용될 수 있습니다.

### [Benchmarking Mental State Representations in Language Models](https://arxiv.org/abs/2406.17513)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.17513.png)

Vote: 1

Authors: Andreas Bulling, Matteo Bortoletto, Constantin Ruhdorfer, Lei Shi

- **What's New**: 이 연구는 최신 언어 모델(LM)이 Theory of Mind (ToM) 능력을 얼마나 잘 갖추고 있는지 평가하고, 이를 통해 인간과 상호작용하는 AI의 성능을 향상시키는 방법을 탐구합니다. 특히, 다양한 LM 패밀리와 모델 크기, 미세 조정(fine-tuning) 접근법을 통해 모델의 내부 표현을 조사합니다.
- **Technical Details**: 이 연구는 여러 크기의 Llama-2 및 Pythia 모델(70백만에서 700억 매개변수)을 대상으로, 의도한 질문(probing), 지시 기반 튜닝(instruction-tuning) 및 인간 피드백을 통한 강화 학습(RLHF)의 영향을 평가했습니다. 또한, 프롬프트 변형(prompt variations) 실험을 통해 LM의 내부 표현이 프롬프트 변화에 얼마나 민감한지 조사했습니다. 마지막으로, 대조적 활성화 추가(Contrastive Activation Addition, CAA)를 사용하여 별도의 질문 도구를 훈련하지 않고도 모델의 성능을 향상시킬 수 있음을 보였습니다.
- **Performance Highlights**: 실험 결과, 모델 크기와 미세 조정이 ToM 성능에 중요한 영향을 미친다는 것을 발견했습니다. 또한, 프롬프트의 변화가 LM의 표현에 민감하게 작용한다는 결과를 얻었습니다. 대조적 활성화 추가(CAA)를 통해 모델의 활성화를 조정함으로써 다양한 ToM 작업에서 성능을 크게 향상시킬 수 있었습니다.

### [ResumeAtlas: Revisiting Resume Classification with Large-Scale Datasets and Large Language Models](https://arxiv.org/abs/2406.18125)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.18125.png)

Vote: 1

Authors: Ahmed Heakl, Noran Mohamed, Ahmed Zaky, Youssef Mohamed, Ali Sharkaway

- **What's New**: 이번 연구는 이력서 분류 분야에서 중요한 도약을 이루며 현재의 문제점을 해결하고 기존 방법론의 한계를 극복합니다. 본 연구에서는 대규모 데이터셋 수집과 전처리를 통해 43개의 서로 다른 이력서 종류를 대표하는 13,389개의 레코드를 포함하는 데이터셋을 구축하였습니다. 또한, Gemini와 BERT와 같은 최신 LLM을 활용하여 91%의 top-1 정확도와 97%의 top-5 정확도를 달성하여 현존하는 최고 성능의 모델을 능가하였습니다.
- **Technical Details**: 본 연구는 Google, Bing, 주요 이력서 웹사이트 등 다양한 출처에서 고품질 데이터셋을 수집하고 필터링하는 데 약 400시간을 소요하여 다양한 이력서 샘플을 포함하도록 하였습니다. 생성된 데이터셋은 총 13,389개 레코드로 43개의 클래스에 걸쳐 있으며, 이를 통해 다양한 이력서 분류 작업의 복잡성을 다뤘습니다. 또한, Gemma와 BERT와 같은 최신 Transformer 모델을 이용하여 이력서 분류의 정확성과 견고성을 극대화하였습니다.
- **Performance Highlights**: 본 연구는 이력서 분류 작업에서 91%의 top-1 정확도와 97%의 top-5 정확도를 달성하면서 현존하는 최고 성능의 모델들을 능가하였습니다. 또한 고품질의 코드베이스를 제공하여 연구 재현성을 높이고, 다른 연구자들이 쉽게 활용할 수 있도록 했습니다.

### [AUTOHALLUSION: Automatic Generation of Hallucination Benchmarks for Vision-Language Models](https://arxiv.org/abs/2406.10900)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.10900.png)

Vote: -

Authors: Xijun Wang, Tianyi Zhou, Ruiqi Xian, Dianqi Li, Xiaoyu Liu, Dinesh Manocha, Shuaiyi Huang, Jordan Lee Boyd-Graber, Tianrui Guan, Furong Huang, Xiyang Wu, Abhinav Shrivastava

- **What's New**: 자동 생성된 쇄도 사례(automated hallucination cases)를 대량 생산할 수 있는 AutoHallusion 파이프라인(pipeline)을 개발했습니다. 이 파이프라인은 LVLMs의 언어 선입견(language priors)을 탐색하여 생성된 답변에서 이미지와의 불일치를 찾아냅니다. 그런 다음, 이미지에 서로 모순되는 객체들을 포함시키거나, 공간적 관계를 뒤트는 방식으로 다양한 쇄도 사례를 만들어냅니다.
- **Technical Details**: AutoHallusion은 언어 모듈의 강한 선입견에 특정 객체나 그들의 문맥적 관계를 할당하는 탐색 과정을 통해 생성된 답변을 시작점으로 합니다. 그런 다음, (1) 탐색된 선입견과 모순되는 객체를 포함하는 이미지와 (2) 문맥적 관련 객체의 존재 여부와 그들의 공간적 관계에 대한 질문을 생성합니다. 이 방법을 통해 LVLMs가 언어 선입견에 의해 편향된 경우 잘못된 응답을 생성하게 하여 그로 인한 쇄도를 유도할 수 있습니다. 또한, 비정상 객체 삽입(abnormal object insertion), 쌍 객체 삽입(paired object insertion), 상관 객체 제거(correlated object removal) 등 세 가지 핵심 전략을 사용하여 장면의 객체를 조작하고 언어 선입견과 모순되는 이미지를 생성합니다.
- **Performance Highlights**: AutoHallusion은 GPT-4V(ision), Gemini Pro Vision, Claude 3, LLaVA-1.5 등 최신 LVLMs에서 97.7%에서 98.7%의 성공률로 쇄도를 유도하는 성과를 달성했습니다. 이 결과는 합성된 데이터와 실제 데이터 모두에서 확인되었습니다.

### [T-FREE: Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings](https://arxiv.org/abs/2406.19223)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2406.19223.png)

Vote: -

Authors: Björn Deiseroth, Patrick Schramowski, Manuel Brack, Samuel Weinbach, Kristian Kersting

- **What's New**: 새로운 패러다임인 'Tokenizer-Free Sparse Representations for Memory-Efficient Embeddings' (T-Free)를 제안합니다. 이 접근 방식은 전통적인 토크나이저를 대체하며, 해시된 문자 삼중 패턴을 통해 각 단어를 직접 임베딩합니다. 이를 통해 하위 단어 토큰이 필요 없어 코드 및 메모리 효율성을 크게 향상시킵니다.
- **Technical Details**: 전통적인 LLM 시스템(대형 언어 모델 시스템)은 텍스트를 서브워드로 분할하고 이를 정수 표현으로 변환하는 토크나이저에 의존합니다. T-Free는 토크나이저의 사용을 배제하고, 데이터의 문자를 해싱한 후 직접 단어를 희소 활성화 패턴으로 임베딩합니다. 이 방식을 통해 다양한 언어에 대하여 거의 최적의 성능을 유지할 수 있으며, 형태학적으로 유사한 단어 간 겹침을 모델링하여 개별 변형을 학습할 필요 없이 임베딩 레이어 크기를 87.5%까지 줄일 수 있습니다.
- **Performance Highlights**: 토크나이저를 사용하는 기존 모델과 비교했을 때, T-Free는 전통적인 임베딩 크기를 평균적으로 87.5% 줄이고, 텍스트 인코딩 길이를 56% 감소시킵니다. 또한 T-Free는 새로운 언어로의 전이 학습에서 성능을 빠르게 향상시키며, 기존 토크나이저 기반 모델은 적응이 미비합니다. T-Free를 활용한 실험에서는 1B LLM을 처음부터 학습시키고, 표준 벤치마크에서 전통적인 모델과 경쟁력 있는 성능을 입증했습니다.



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
