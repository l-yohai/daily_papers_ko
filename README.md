# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using ChatGPT.

Thanks to [@AK391](https://github.com/AK391) for great work.


## Daily Papers (2024-03-13)

### [Stealing Part of a Production Language Model](https://arxiv.org/abs/2403.06634)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/dhiUnxDGTRatavk4pKf9h.png)

Vote: 57

Authors: A. Feder Cooper, Jonathan Hayase, Milad Nasr, Nicholas Carlini, Eric Wallace, David Rolnick, Krishnamurthy Dj Dvijotham, Katherine Lee, Thomas Steinke, Arthur Conmy, Daniel Paleka, Matthew Jagielski, Florian Tramèr

- 본 연구에서는 OpenAI의 ChatGPT나 Google의 PaLM-2와 같은 상용 언어 모델로부터 정밀하고 중요한 정보를 추출하는 최초의 모델 도용 공격을 소개합니다.
- 연구팀은 일반적인 API 접근을 통해 변환기 모델의 임베딩 투영 레이어(대칭성까지)를 복원하는 방법을 선보였습니다.
- 이 공격을 이용하여 20 USD 미만의 비용으로 OpenAI의 Ada와 Babbage 언어 모델의 전체 투영 행렬을 추출했으며, 각각의 모델이 숨겨진 차원을 1024와 2048을 가지고 있음을 확인했습니다.
- gpt-3.5-turbo 모델의 정확한 숨겨진 차원 크기를 복원하였고, 전체 투영 행렬을 복구하는 데 예상되는 비용이 2,000 USD 미만임을 추정했습니다.
- 논문은 가능한 방어책과 완화책을 제시하고, 이 공격이 미래에 어떻게 확장될 수 있는지에 대한 시사점을 논의합니다.

### [Adding NVMe SSDs to Enable and Accelerate 100B Model Fine-tuning on a Single GPU](https://arxiv.org/abs/2403.06504)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2403.06504.png)

Vote: 33

Authors: Fei Wu, Kaiqi Chen, Binhang Yuan, Zihan Yang, Changyue Liao, Zeke Wang, Mo Sun

- 대규모 언어 모델의 발전은 엄청난 파라미터 수를 활용하여 뛰어난 성능을 제공하지만, 현재 가장 큰 용량인 80GB GPU 메모리로는 충분하지 않습니다.
- 많은 GPU의 메모리를 집계하는 방법은 예산이 제한된 학술 연구자들에게 막대한 비용이 들어갑니다.
- 본 논문은 보편적인 서버와 저가형 GPU를 사용하여 단일 GPU에서 대형 모델의 미세조정을 가능하게 하는 방법에 집중합니다.
- 최신 연구인 ZeRO-Infinity는 보편 서버에서 저효율 스와핑 때문에 낮은 GPU 활용도와 CPU 메모리 용량 제한으로 모델 크기에 한계를 가지고 있습니다.
- 저희는 저비용 훈련 프레임워크인 Fuyou를 소개하여 SSD-CPU 통신을 최적화 차원에 추가하여 GPU 활용을 극대화하기 위해 계산과 데이터 스와핑을 시스템적으로 조화시킵니다.
- 실험 결과, Fuyou는 GPT-3 175B 모델을 소비자 GPU RTX 4090에서 고용도로 미세조정할 수 있었으며, ZeRO-Infinity는 실패했다는 것을 보여줍니다.
- 작은 GPT-3 13B 모델을 훈련 할 때, Fuyou는 RTX 4090 GPU에서 156 TFLOPS를 달성하는 반면, ZeRO-Infinity는 오직 45 TFLOPS을 달성합니다.

### [An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models](https://arxiv.org/abs/2403.06764)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2403.06764.png)

Vote: 16

Authors: Haozhe Zhao, Shuai Bai, Tianyu Liu, Chang Zhou, Liang Chen, Baobao Chang, Junyang Lin

- 이 연구에서는 대형 시각-언어 모델(LVLMs)에서 비효율적인 주의력 현상을 발견, 특히 LLaVA-1.5, QwenVL-Chat 및 Video-LLaVA와 같은 주목할 만한 모델에서 시각 토큰에 대한 주의 계산의 극단적인 비효율성을 지적하였다.
- 이러한 문제를 해결하기 위해, 연구진은 초기 레이어에서 적응형 주의 패턴을 학습하고 이후 레이어에서 시각 토큰을 가지치기하는 FastV라는 범용적인 플러그 앤 플레이 방법을 제안했다.
- FastV는 이미지 및 비디오 이해 작업의 넓은 범위에서 성능을 해치지 않으면서 계산 비용을 대폭 절감할 수 있는 능력을 입증하였다 (예를 들어, LLaVA-1.5-13B의 FLOPs을 45% 절감).
- FastV의 계산 효율성과 성능 트레이드오프는 매우 맞춤화 가능하며 효율적인 파레토 최적성을 가지고 있다.
- 이 방법론은 13B 파라미터 모델의 FLOPs를 7B 파라미터 모델의 예산보다 낮추면서도 여전히 우수한 성능을 유지할 수 있도록 압축할 수 있다.
- 연구진은 FastV가 에지 디바이스 및 상업 모델에서 LVLMs의 배치에 있어 실용적인 가치가 있다고 믿고 있으며, 관련 코드는 https://github.com/pkunlp-icler/FastV에서 공개되었다.

### [V3D: Video Diffusion Models are Effective 3D Generators](https://arxiv.org/abs/2403.06738)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2403.06738.png)

Vote: 16

Authors: Huaping Liu, Yikai Wang, Feng Wang, Zilong Chen, Zhengyi Wang

- 자동 3D 생성에 대한 관심이 최근 급증하면서, 기존 방법들은 생성 속도를 높였지만 한정된 모델 용량과 3D 데이터로 인해 상세도가 떨어지는 객체를 만들곤 했습니다.
- 우리는 비디오 확산 모델의 최근 발전에 동기를 얻어, 3D 생성을 용이하게 하는 V3D를 소개합니다.
- 3D 세계를 인식하는 비디오 확산의 잠재력을 최대한 활용하기 위해, 기하학적 일관성 선행 조건을 도입하고 비디오 확산 모델을 다중 시점 일관성이 있는 3D 생성기로 확장합니다.
- 이를 통해, 최신 비디오 확산 모델을 개선하여 단일 이미지로부터 물체를 둘러싸는 360도 궤도 프레임을 생성할 수 있게 됩니다.
- 맞춤형 복원 파이프라인을 통해, 3분 이내에 고품질 메쉬나 3D 가우시안을 생성할 수 있습니다.
- 또한, 우리의 방법은 희소한 입력 시점을 가진 장면 레벨의 새로운 시점 합성으로 확장 가능하며, 카메라 경로에 대한 정밀한 제어를 달성합니다.
- 광범위한 실험을 통해 제안된 접근법이 특히 생성 품질과 다중 시점 일관성 측면에서 우수한 성능을 보임을 입증했습니다.
- 이 연구의 코드는 https://github.com/heheyas/V3D에서 제공되고 있습니다.

### [VideoMamba: State Space Model for Efficient Video Understanding](https://arxiv.org/abs/2403.06977)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2403.06977.png)

Vote: 14

Authors: Yi Wang, Yinan He, Limin Wang, Xinhao Li, Yu Qiao, Kunchang Li, Yali Wang

- 이 연구에서는 비디오 이해의 중복성과 전역 의존성이라는 이중의 도전을 해결하기 위해 Mamba 모델을 비디오 분야에 혁신적으로 적용했습니다.
- 제안된 VideoMamba는 기존의 3D 합성곱 신경망과 비디오 트랜스포머의 제한을 극복하며, 효율적인 장기 모델링을 가능하게 하는 선형 복잡도 연산자를 사용합니다.
- 비디오Mamba는 광범위한 평가를 통해 네 가지 핵심 능력을 드러냈습니다: (1) 대규모 데이터셋 사전 학습 없이도 시각적 도메인의 확장 가능성, (2) 미세한 동작 차이를 갖는 단기 행동을 인식하는 민감도, (3) 전통적인 특징 기반 모델을 뛰어넘는 장기 비디오 이해에서의 우수성, (4) 다중 모달 맥락에서의 강건성을 보이며 다른 모달리티와의 호환성.
- VideoMamba는 비디오 이해를 위한 새로운 벤치마크를 설정하며, 비디오의 이해에 대해 확장 가능하고 효율적인 솔루션을 제공합니다.
- 모든 코드와 모델은 https://github.com/OpenGVLab/VideoMamba 에서 사용할 수 있습니다.

### [VidProM: A Million-scale Real Prompt-Gallery Dataset for Text-to-Video Diffusion Models](https://arxiv.org/abs/2403.06098)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2403.06098.png)

Vote: 10

Authors: Yi Yang, Wenhao Wang

- 본 논문에서는 텍스트-비디오 확산 모델을 위한 최초의 대규모 데이터셋인 VidProM을 소개하며, 이는 실제 사용자들로부터 얻은 167만 개의 고유 텍스트-비디오 프롬프트를 포함하고 있다.
- 데이터셋에는 4개의 최첨단 확산 모델들로 생성된 669만 개의 비디오와 관련 데이터도 함께 제공된다.
- 큰 규모의 데이터셋을 큐레이션하는 과정이 시간이 많이 소요되고 비용이 든다는 점을 보여주며, 이는 기존의 이미지 생성을 위한 프롬프트-갤러리 데이터셋 DiffusionDB와 VidProM의 차별점을 이해하는 데에 도움이 된다.
- 연구진들은 텍스트-비디오 프롬프트에 대한 신규 데이터셋 필요성을 확인하고 실제 사용자들이 비디오를 생성할 때 어떤 선호도를 보이는지에 대한 통찰을 얻는다.
- 제안된 VidProM 데이터셋은 텍스트-비디오 프롬프트 엔지니어링, 효율적인 비디오 생성, 확산 모델을 위한 비디오 복제 검출 등 새롭고 흥미로운 연구 영역을 많이 탐색할 수 있는 기회를 제시한다.
- 연구진은 CC-BY-NC 4.0 라이선스 하에 GitHub과 Hugging Face를 통하여 VidProM 데이터셋을 공개적으로 이용할 수 있게 제공한다.

### [Algorithmic progress in language models](https://arxiv.org/abs/2403.05812)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2403.05812.png)

Vote: 9

Authors: Tamay Besiroglu, Anson Ho, Ege Erdil, Jaime Sevilla, David Owen, Robi Rahman, David Atkinson, Zifan Carl Guo, Neil Thompson

- 본 연구에서는 딥러닝 도입 이후 언어 모델을 사전 학습하기 위한 알고리즘의 개선 속도를 조사했다.
- 2012년부터 2023년에 걸쳐 Wikitext와 Penn Treebank에서 이루어진 200개 이상의 언어 모델 평가 데이터셋을 사용하여, 설정된 성능 임계값을 달성하기 위한 계산 요구량이 약 8개월마다 절반으로 줄어들었다는 사실을 발견했다.
- 이러한 속도는 모어의 법칙에 따른 하드웨어 이득보다 상당히 빠르며, 95% 신뢰 구간은 5개월에서 14개월 사이였다.
- 알고리즘의 진보를 정량화하고 모델의 규모 확장과 학습 알고리즘의 혁신 기여도를 비교 분석할 수 있는 향상된 규모 법칙을 추정했다.
- 분석 결과 트랜스포머와 같은 새로운 아키텍처의 개발에도 불구하고, 시간이 지남에 따라 계산 능력 증가가 전체 성능 개선에 더 큰 기여를 했음을 밝혔다.
- 비록 벤치마크 데이터의 불확실성에 한계가 있지만, 해당 분석은 언어 모델링 분야의 빠른 발전을 정량화하고, 계산 능력과 알고리즘의 상대적 기여도를 조명하는 데 기여했다.

### [Multistep Consistency Models](https://arxiv.org/abs/2403.06807)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2403.06807.png)

Vote: 8

Authors: Tim Salimans, Jonathan Heek, Emiel Hoogeboom

- 확산 모델은 훈련이 비교적 쉽지만 샘플을 생성하는 데 많은 단계가 필요하며, 일관성 모델은 훈련이 훨씬 더 어렵지만 한 단계로 샘플을 생성할 수 있습니다.
- 본 논문에서는 일관성 모델과 확산 모델 간의 중간 형태인 '멀티스텝 일관성 모델'을 제시하며, 이는 샘플링 속도와 품질 사이의 절충을 가능하게 합니다.
- 특히 한 단계의 일관성 모델은 전통적인 일관성 모델을 나타내는 반면, 무한 단계의 일관성 모델은 확산 모델과 동일함을 보여줍니다.
- 멀티스텝 일관성 모델은 실제로 매우 효과적이며, 단일 단계에서 2-8 단계로 샘플 예산을 늘림으로써 샘플링 속도의 이점을 유지하면서도 더 쉽게 높은 품질의 샘플을 생성하는 모델을 훈련할 수 있습니다.
- 주목할 만한 결과로는 일관성 증류를 사용하여 8단계의 Imagenet 64에서 1.4 FID와 Imagenet128에서 8단계의 2.1 FID를 달성했습니다.
- 또한 이 방법이 텍스트-이미지 확산 모델로 확장될 수 있음을 보여주며, 원래 모델의 품질에 매우 가까운 샘플을 생성합니다.

### [FaceChain-SuDe: Building Derived Class to Inherit Category Attributes for One-shot Subject-Driven Generation](https://arxiv.org/abs/2403.06775)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2403.06775.png)

Vote: 3

Authors: Chang Liu, Lei Shang, Baigui Sun, Jie Chen, Pengchong Qiao, Xiangyang Ji

- 본 논문은 텍스트-이미지 생성을 개인화하는 데 관심을 가진 주제 중심 생성에 대한 새로운 접근 방식을 제안합니다.
- 일반적인 연구들은 새 주제의 개인적 속성을 배우는 데 중점을 두지만, 주제가 사전 훈련된 모델의 특정 카테고리의 전문화임이 중요하다는 사실을 간과합니다.
- 이로 인해 주제는 카테고리 내 속성을 종합적으로 물려받지 못하고, 속성 관련 생성이 불충분하게 됩니다.
- 본 논문은 객체 지향 프로그래밍에 영감을 받아 주제를 그 의미 카테고리의 기본 클래스라 할 수 있는 파생 클래스로 모델링합니다.
- 제안하는 방식인 Subject-Derived regularization (SuDe)은 사용자가 제공한 예시로부터 주제의 고유 속성을 학습하면서, 주제가 속한 카테고리의 공공 속성을 상속받도록 해줍니다.
- SuDe 방법은 주제 중심 생성 이미지가 주제의 카테고리에 의미적으로 속하도록 제약함으로써 기본-파생 클래스 모델링을 구축합니다.
- 다양한 주제에 걸쳐 세가지 기본 라인과 두 가지 백본을 사용한 광범위한 실험들은 SuDe가 창의적인 속성 관련 생성을 가능하게 하면서도 주제의 충실도를 유지한다는 것을 보여줍니다.
- 코드는 곧 FaceChain (https://github.com/modelscope/facechain)에서 오픈 소스로 공개될 예정입니다.



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
