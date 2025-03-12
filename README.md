# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using GPT-4o.

Thanks to [@AK391](https://github.com/AK391) for great work.

## Daily Papers (2025-03-12)

### [Crowdsource, Crawl, or Generate? Creating SEA-VL, a Multicultural Vision-Language Dataset for Southeast Asia](https://arxiv.org/abs/2503.07920)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.07920.png)

Vote: 73

Authors: Tack Hwa Wong, Evan, Bin Wang, Lynnette Hui Xian Ng, Amit Agarwal, Fajri Koto, Giang Nguyen, Yeshil Bangera, Mohamed Fazli Imam, John Amadeo Daniswara, Dan John Velasco, Kadek Hendrawan Palgunadi, Meisyarah Dwiastuti, Joel Ruben Antony Moniz, Jostin Jerico Rosal, Hanif Muhammad Zhafran, Jan Christian Blaise Cruz, Aye Hninn Khine, Adrian Xuan Wei Lim, Kenneth Ko Han Chen, Audra Aurora Izzani, Isaiah Flores, Kevin Pratama, Rochana Prih Hastuti, Jauza Akbar Krito, Ruochen Zhang, Alham Fikri Aji, Phakphum Artkaew, Takdanai Kreangphet, Salsabila Zahirah Pranida, Rifo Ahmad Genadi, Joseph Marvin Imperial, Vicky Feliren, M. Alif Al Hakim, Carlos Rafael Catalan, Saptarshi Saha, Holy Lovenia, Matthew Theodore Roque, Mithil Bangera, Karissa Vincentio, Onno P. Kampman, Piyalitt Ittichaiwong, Eryawan Presma Yulianrifat, Jiayun Luo, Patricia Nicole Monderin, Fenal Ashokbhai Ilasariya, Hitesh Laxmichand Patel, Robert Wijaya, Börje F. Karlsson, Anjela Gail Santos, Manuel Antonio Rufino, Priyaranjan Pattnayak, Samuel Cahyawijaya, Anab Maulana Barik, Tim Santos, Chengwei Wei, Filbert Aurelian Tjiaranata, Ikhlasul Akmal Hanif, Jun Kevin, Ayushman Singh, Ming Shan Hee, Yanzhi Yu, Mahardika Krisna Ihsani, Teddy Ferdinan, Yueqi Song, Supryadi, Michael Anugraha, Tirana Noor Fatyanosa, William Nixon, Frederikus Hudi, Muhammad Ravi Shulthan Habibi, Mohammad Rifqi Farhansyah, Muhammad Rizky Sya'ban, Kun Kerdthaisong, Adisai Na-Thalang, Bahrul Ilmi Nasution, Kaung Si Phyo, Haochen Li, Lester James V. Miranda, Kanyakorn Veerakanjana, David Anugraha, Muhammad Reza Qorib, Can Udomcharoenchaikit, Genta Indra Winata, Taki Hasan Rafi, Wan Shen Lim, Fadil Risdian Ansori, Peerat Limkonchotiwat, Christian Simon, Thant Thiri Maung, Rian Adam Rajagede, Richardy Lobo' Sapan

- ***What's New***: SEA-VL는 동남아시아(Southeast Asia; SEA) 언어를 위한 고품질 문화적으로 관련 있는 데이터셋을 개발하는 오픈 소스 이니셔티브입니다. 이 데이터셋은 SEA 국가의 기여자들이 참여하여 다양성을 높이고, 텍스트-이미지 정보의 통합 연구에 잘 반영될 수 있도록 설계되었습니다. 특히, 이미지 크롤링(image crawling)과 이미지 생성(image generation)을 통해 문화적으로 관련 있는 이미지를 자동으로 수집하는 방법을 탐구합니다.
- ***Technical Details***: SEA-VL 데이터셋은 총 1.28M개의 문화적으로 관련 있는 이미지를 수집하며 기존 데이터셋보다 50배 이상 큽니다. 이미지 크롤링을 통한 수집 과정에서 대략 85%의 문화적 관련성을 달성하였고, 이는 크라우드소싱(crowdsourcing)보다 비용 및 시간 효율적입니다. 그러나 생성된 이미지들은 여전히 SEA 지역의 전통과 문화적 문맥을 정확하게 반영하지 못하는 한계를 가집니다. 따라서 데이터 수집 과정은 세 가지 방법으로 진행됩니다: (1) 매뉴얼 데이터를 통한 인간 수집, (2) 크롤링된 이미지의 필터링 및 중복 제거 파이프라인, (3) 확산 모델(diffusion models)을 통한 이미지 생성입니다. 이외에도 메타데이터 수집의 자동화 역시 탐구하고 있습니다.
- ***Performance Highlights***: 현존 이미지 생성 모델들은 SEA 문화적 요소를 포착하는데 어려움을 겪고 있으며, 특히 문화적 정확성과 자연스러움에서 인간이 수집한 이미지에 비해 크게 미치지 못합니다. 가장 우수한 생성 모델인 Stable Diffusion 3.5도 모든 카테고리에서 평균 1.5 미만의 정확도를 기록하며, 자연스러움 역시저조한 상태입니다. 그러나, 영어 캡션 생성에서는 어느 정도 신뢰할 수 있는 결과를 보여주며, Pangea(7B)와 Qwen2-VL(7B) 모델이 가장 우수한 성능을 보였습니다.

### [LMM-R1: Empowering 3B LMMs with Strong Reasoning Abilities Through Two-Stage Rule-Based RL](https://arxiv.org/abs/2503.07536)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.07536.png)

Vote: 50

Authors: Jie Liu, Kai Yang, Zhiyuan You, Gongrui Zhang, Qipeng Zhu, Xingzhong Xu, Xin Geng, Miaosen Zhang, Yingzhe Peng, Xu Yang

- ***What's New***: LMM-R1은 3B 파라미터를 가진 대형 멀티모달 모델(Large Multimodal Models; LMMs)의 제한된 용량에서 강력한 추론 능력을 강화하기 위한 단계적 룰 기반 강화 학습(two-stage rule-based RL) 프레임워크를 제안합니다. 본 연구는 시각적 정보가 복잡한 논리적 추론과 상호작용할 때 발생하는 독특한 과제를 해결하기 위해 텍스트 전용 데이터로 시작해 기초 추론 능력을 먼저 강화하는 방법인 해당 프레임워크의 첫 번째 단계인 Foundational Reasoning Enhancement(FRE)을 제안합니다. 두 번째 단계인 멀티모달 일반화 훈련(Multimodal Generalization Training, MGT)에서는 이와 같은 추론 능력을 멀티모달 도메인으로 일반화합니다. 이를 통해 LMM-R1 프레임워크는 고비용의 고품질 멀티모달 훈련 데이터를 피할 수 있는 데이터 효율적인 패러다임을 제시합니다. 실험 결과, 모형이 여러 벤치마크에서 기존 모델 대비 4.83%(멀티모달 벤치마크), 4.5%(텍스트 전용) 및 3.63%(복잡한 Football Game 작업) 성능이 향상되었습니다. Text-based reasoning 강화가 효과적인 멀티모달 일반화를 가능하게 함을 입증합니다. 향후 연구에서는 더 많은 LMMs에 이 프레임워크를 확장하고 고품질 멀티모달 추론 데이터를 합성하는 방법을 개발할 계획입니다.riangles into four nonoverlapping convex quadrilaterals, as shown. If the sides of  are not parallel to the sides of [ 온라인 멀티모달 지각 요구사항이 복합적 도메인에서 지속 가능하도록 강화 추론 능력을 전이할 수 있는 모델의 역량을 평가합니다. 이 두 도메인은 일반적인 멀티모달 시나리오와 에이전트 관련 추론 도메인으로 구분됩니다. 이 단계를 통해 FRE이 단계에서 훈련된 기초 모델의 성능이 다양한 멀티모달 도메인에 효율적으로 전이되는지 평가합니다. ] 19/63 (2024/06), 6.63% 초과 증가했다. 관련된 벤치마크인 'GeoQA', 'Sokoban' 및 'Footbal Game'에서 각각 4.5%, 4.83% 및 3.63%의 성능 증가를 보였습니다. 주요 기여는 다음과 같습니다: • 우리는 장관적이며 비용 효율적인 규칙 기반 강화 학습(rule-based reinforcement learning; RL) 방법론이 혼합 모달 추론을 강화하여 고품격 멀티모달 훈련 데이터에 대한 의존성을 피할 수 있음을 입증하였습니다.

### [YuE: Scaling Open Foundation Models for Long-Form Music Generation](https://arxiv.org/abs/2503.08638)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.08638.png)

Vote: 46

Authors: Jianwei Yu, Junjie Wang, Zhaoxiang Zhang, Tianhao Shen, Chao Zhang, Hanfeng Lin, Yiming Liang, Ziya Zhou, Xiangzhou Wang, Xiaohuan Zhou, Zhen Ye, Ge Zhang, Chunhui Wang, Peng Li, Xingwei Qu, Ziyang Ma, Tianyu Zheng, Gus Xia, Lijun Yu, Xu Tan, Lingrui Mei, Liumeng Xue, Guojian Pang, Haohe Liu, Yinghao Ma, Jian Yang, Wei Xue, Shuyue Guo, Zeyue Tian, Jiahao Pan, Xiaowei Chi, Xingjian Du, Wenhao Huang, Xinrun Du, Yongyi Zang, Xie Chen, Minghao Liu, Xu Li, Xipeng Qiu, Xinyu Zhou, Shansong Liu, Xinyue Zhang, Wenye Ma, Zihao Wang, Yatian Wang, Wenhu Chen, Ruibin Yuan, Shangda Wu, Chenghua Lin, Yike Guo, Zhenzhu Yang, Emmanouil Benetos, Yong Chen, Jiaheng Liu, Roger Dannenberg, Yizhi Li, Jun Zhan

- ***What's New***: YuE는 LLaMA2 아키텍처를 기반으로한 재생 토큰(Trillions of Tokens)을 확장하여 최대 5분 길이의 고품질 음악을 생성할 수 있는 오픈소스 음악 생성 모델입니다. 이는 가사(literary alignment)에 맞는 음악적 구조와 매혹적인 보컬 멜로디를 유지하며, 여러 스타일 변환(Style Transfer, e.g., 일본 시티팝을 영어 랩으로 변환하며 원래의 반주 유지) 및 양방향 생성(Bidirectional Generation)을 가능하게 합니다.
- ***Technical Details***: YuE는 두 단계로 구성된 아토머그레시브 언어 모델(Autoregressive Language Model; AR LM)로, 음악 언어 모델링(Music Language Modeling)을 위한 Stage-1과 잔차 모델링(Residual Modeling)을 위한 Stage-2로 이뤄져 있습니다. Stage-1은 트랙 분리된 다음 토큰 예측(Track-Decoupled Next-Token Prediction), 구조적 점진적 조건부(Structural Progressive Conditioning), 음악 내 컨텍스트 학습(Music In-Context Learning)을 통해 길이와 내용 면에서 일관성을 유지하는 음악을 생성할 수 있도록 설계되었습니다. Stage-2에서는 잔여 모델링(Residual Modeling)을 통해 추가적인 코드북을 활용하여 오디오를 정교하게 만들어냅니다.
- ***Performance Highlights***: 유에는 노래의 음악성, 보컬의 민첩성, 생성된 오디오의 길이에서 강력한 성능을 보여주었으며, 몇몇 독점 시스템보다 뛰어난 성과를 기록했습니다. 특히, 노래의 전체 길이를 5분까지 생성할 수 있는 능력을 보여 줍니다. 주어진 텍스트와 오디오 출력 간의 3점이 높게 일치하는 것으로 나타나, 유에가 음악 이해 작업에서 최첨단의 방법들과 맞먹거나 그 이상의 결과를 달성하였음을 확인했습니다.

### [MagicInfinite: Generating Infinite Talking Videos with Your Words and Voice](https://arxiv.org/abs/2503.05978)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.05978.png)

Vote: 26

Authors: Tian Ye, Hanzhong Guo, Wei Li, Qingyu Yin, Michael Lingelbach, Daquan Zhou, Jiantong Zhao, Xuancheng Yang, Hongwei Yi, Shitong Shao, Lei Zhu, Terrance Wang, Zeke Xie

- ***What's New***: MagicInfinite는 새롭게 개발된 확산 기반 Transformer(Diffusion Transformer; DiT) 프레임워크로, 기존의 초상화 애니메이션의 한계를 극복하여 다양한 캐릭터 유형에 대해 고품질의 결과물을 제공합니다. 이 모델은 다양한 얼굴 포즈를 지원하고 여러 캐릭터를 애니메이트 할 수 있습니다.
- ***Technical Details***: MagicInfinite는 3D 풀어텐션 메커니즘과 슬라이딩 윈도우 노이즈 제거 전략을 통해 시간적 일관성과 시각적 품질을 유지하면서 다양한 스타일의 캐릭터들을 무한히 생성할 수 있습니다. 이 두 단계 커리큘럼 학습은 오디오, 텍스트 및 참조 이미지를 통합하여 멀티모달 제어를 가능케 하고, 지역 특화 마스크와 적응형 손실 함수를 통해 오디오 제어를 개선합니다.
- ***Performance Highlights***: MagicInfinite는 주어진 평가 벤치마크에서 오디오-입술 동기화, 정체성 보존, 운동 자연성에서 우수한 성능을 보여주었습니다. 효율성 측면에서 20배의 추론 속도 향상을 이루었습니다.

### [UniF^2ace: Fine-grained Face Understanding and Generation with Unified Multimodal Models](https://arxiv.org/abs/2503.08120)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.08120.png)

Vote: 25

Authors: Tingting Long, Liya Guo, Chun Fan, Delin Qu, Ming Li, Junzhe Li, Xuerui Qiu, Linrui Xu

- ***What's New***: UniF2ace는 얼굴 영역의 세부적인 이해 및 생성을 위해 설계된 첫 번째 통합 멀티모달 모델(Unified Multimodal Model; UMM)입니다. 이 모델은 새로운 확산 기법(Diffusion techniques)과 두 가지 수준의 전문가 혼합 구조(Two-level Mixture-of-Experts Architecture)를 활용하여 다양한 얼굴 속성을 아우르는 세부적인 이해 및 생성 능력을 높입니다.
- ***Technical Details***: UniF2ace는 자가 구축한 대규모 데이터셋 UniF2ace-130K에 대해 훈련되었습니다. 총 13만 개의 이미지-텍스트 쌍과 백만 개 이상의 시각적 질문-답변(Visual Question-Answering; VQA) 쌍을 포함하고 있으며, 46가지의 다양한 얼굴 속성과 관련된 질문을 포함하고 있습니다. 또한 두 가지 상호보완적인 디퓨전 기법과 이중 수준의 전문가 조합 아키텍처(Mixture-of-Experts; MoE)를 도입하여 텍스트와 이미지 간 원활한 교차 모달 정렬을 가능하게 하고, 정교한 얼굴 표현 학습을 통해 이해와 생성 작업 모두의 효율성을 높였습니다.
- ***Performance Highlights***: UniF2ace는 최첨단의 UMMs 및 생성 모델을 능가하여, 이해와 생성 작업 모두에 걸쳐 우수한 성능을 발휘했습니다. VQAscore와 Fréchet Inception Distance (FID), 그리고 VLM-score 등에서 SOTA 성능을 달성하였으며, 관련된 얼굴 속성을 보다 정교하게 캡처할 수 있음이 입증되었습니다.

### [SegAgent: Exploring Pixel Understanding Capabilities in MLLMs by Imitating Human Annotator Trajectories](https://arxiv.org/abs/2503.08625)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.08625.png)

Vote: 22

Authors: Chunhua Shen, Yuzhuo Tian, Hao Chen, Yang Liu, Chunluan Zhou, Ming Yang, Qingpei Guo, Muzhi Zhu

- ***What's New***: 이 논문에서는 Human-Like Mask Annotation Task (HLMAT)을 통해 멀티모달 대형 언어 모델(MLLM)의 세밀한 픽셀 수준 이해 능력을 평가하려고 합니다. HLMAT은 MLLM이 인간 주석자처럼 인터랙티브 세그멘테이션 도구를 사용하여 세그멘테이션 작업을 멀티 스텝 마르코프 결정 과정(Markov Decision Process)으로 모델링하며, 이 과정에서 모델 구조 변경 없이 고품질 마스크를 생성할 수 있도록 지원합니다.
- ***Technical Details***: HLMAT는 MLLM이 픽셀 수준 이해를 위해 인터랙티브 세그멘테이션 도구를 사용하여 사람처럼 동작하도록 설계되었습니다. 여기서 SegAgent라는 모델이 인간 주석 경로로 미세 조정되어, 최첨단 방법과 비교 가능한 성능을 달성합니다. StaR 및 PRM 기반의 트리 검색을 사용하여 복잡한 세그멘테이션 작업에서 모델의 견고성을 향상시키는 방법을 도입했습니다.
- ***Performance Highlights***: SegAgent는 REFCOCO 및 HRES 데이터셋에서 최첨단 기법과 비교 가능한 성능을 보였습니다. 특히 SegAgent-Qwen은 세밀한 픽셀 수준 이해 능력이 뛰어나게 나타났으며, PRM 및 트리 검색 기법을 활용하여 복잡한 상황에서 오류를 효과적으로 줄였습니다.

### [VACE: All-in-One Video Creation and Editing](https://arxiv.org/abs/2503.07598)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.07598.png)

Vote: 22

Authors: Zeyinzi Jiang, Chaojie Mao, Yu Liu, Yulin Pan, Zhen Han, Jingfeng Zhang

- ***What's New***: VACE는 영상 생성 및 편집을 위한 올인원(All-in-one) 프레임워크로, 다양한 영상 작업을 단일 모델에서 처리할 수 있도록 통합했습니다. 이를 통해 기존 영상 생성 모델들이 직면했던 시간적-공간적 일관성 문제를 해결하며, 사용자에게 다양한 창의적 영상 작업 시나리오를 제공합니다.
- ***Technical Details***: VACE는 디퓨전 트랜스포머(Diffusion Transformer; DiT) 구조를 기반으로 하며, 편집, 참고(reference), 마스킹 등의 영상 작업 입력을 비디오 조건 유닛(Video Condition Unit; VCU)이라는 인터페이스로 통합합니다. 이러한 입력 파라다임 하에서 다양한 멀티모달 입력을 처리하며, 개념 분리 전략(concept decoupling strategy)을 통해 어떠한 부분이 수정되어야 하고 보존되어야 하는지를 명확히 합니다.
- ***Performance Highlights***: VACE는 다양한 작업에서 특화된 모델과 동등한 성능을 보여주며, 특히 창의적인 영상 생성과 편집에서 뛰어난 결과를 제시합니다. 비디오 품질 및 일관성 평가에서 우수한 성능을 기록했으며, 사용자의 다양한 입력 요구를 지원하는 다기능 및 적응형 처리가 가능합니다.

### [Seedream 2.0: A Native Chinese-English Bilingual Image Generation Foundation Model](https://arxiv.org/abs/2503.07703)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.07703.png)

Vote: 22

Authors: Zhonghua Zhai, Fei Liu, Yichun Shi, Shijia Zhao, Xuefeng Xiao, Linjie Yang, Fanshi Li, Zhi Tian, Xinyu Zhang, Yu Tian, Xun Wang, Wei Liu, Xin Xia, Xiaochen Lian, Liyang Liu, Ye Wang, Jianchao Yang, Jie Wu, Yuwei Zhang, Guofeng Wu, Shiqi Sun, Qi Zhang, Lixue Gong, Xiaoxia Hou, Peng Wang, Wei Lu, Liang Li, Weilin Huang

- ***What's New***: Seedream 2.0은 차세대 중국어-영어 양방향 이미지 생성 기초 모델로, 기존 모델들의 편향성과 텍스트 렌더링 한계를 극복하여 고품질의 이미지를 생성합니다. 특히 중국 문화적 요소의 정확한 이해와 표현을 가능케 하는 자체 개발된 대형 언어 모델(LLM)을 텍스트 인코더로 통합하여 중국어와 영어 모두에서 뛰어난 성능을 발휘합니다.
- ***Technical Details***: Seedream 2.0은 중국어와 영어 두 가지 언어로 텍스트 프롬프트를 처리할 수 있는 강력한 데이터 시스템과 캡션 시스템을 갖추고 있으며, 상용 자체 개발된 대형 언어 모델(LLM)을 텍스트 인코더로 사용하여 방대한 데이터에서 직접적으로 원어 지식을 학습합니다. 또한 Glyph-Aligned ByT5가 유연한 문자 수준의 텍스트 렌더링을 지원하며, Scaled ROPE가 훈련되지 않은 해상도로의 일반화를 효과적으로 수행합니다. 다단계 후속 교육, SFT 및 RLHF 반복을 통해 모델 성능을 한층 향상시킵니다.
- ***Performance Highlights***: Seedream 2.0은 텍스트-이미지 정렬, 미학적 질, 구조적 정확성 부문에서 탁월한 ELO 점수를 기록하며, 인간 선호와 철저하게 정렬된 출력을 보여줍니다. 특히 중국어 텍스트 렌더링 및 문화적인 장면 생성 능력에서 뛰어난 성과를 입증하며, Doubao(豆包)와 Dreamina(即梦)와 같은 응용 프로그램에서 높은 호평을 받고 있습니다.

### [Video Action Differencing](https://arxiv.org/abs/2503.07860)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.07860.png)

Vote: 21

Authors: Alejandro Lozano, Lisa Dunlap, Trevor Darrell, Yuhui Zhang, Xiaohan Wang, Anita Rau, James Burgess, Serena Yeung-Levy

- ***What's New***: 이번 연구에서는 사용자가 동일한 동작을 수행하는 두 사람 간의 미세한 차이를 식별하는 새로운 비디오 테스크인 Video Action Differencing(VidDiff)을 소개하고, 이를 위해 549개의 비디오 쌍과 4,469개의 미세 동작 차이 어노테이션, 2,075개의 위치 시간 스탬프를 포함한 VidDiffBench라는 벤치마크 데이터셋을 처음으로 제공합니다.
- ***Technical Details***: VidDiff 메서드는 세 단계의 에이전트 워크플로우(Agentic Workflow)로 작업을 분해하여 수행됩니다: 'Action Difference Proposal(행동 차이 게시)', 'Keyframe Localization(키프레임 로컬라이제이션)', 'Frame Differencing(프레임 차이)'로 구성된 각 단계는 특화된 모델을 사용합니다. 비디오의 미세 동작을 비교하고, VLMs(Vision-Language Models) 및 LLMs(Large Language Models)를 사용해 차이를 제안하고, CLIP 모델을 사용해 프레임을 구체적으로 로컬라이징합니다.
- ***Performance Highlights***: VidDiffBench 벤치마크의 실험 결과, 기존의 최첨단 LMMs(Large Multimodal Models)인 GPT-4o와 Qwen2-VL은 제로샷에서 비디오 동작 차별화 작업에 어려움을 겪고 있으며, VidDiff 메서드를 통한 구조화된 접근 방식이 비디오 비교를 향상시킴을 보여주었습니다. 특히 문제의 난이도가 높아질수록, VidDiff 메서드가 더 나은 성능을 보이는 것으로 나타났습니다.

### [Gemini Embedding: Generalizable Embeddings from Gemini](https://arxiv.org/abs/2503.07891)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.07891.png)

Vote: 19

Authors: Kaifeng Chen, Iftekhar Naim, Ke Chen, Sahil Dua, Andreas Doumanoglou, Michael Boratko, Frank Palma Gomez, Simon Baumgartner, Yunhsuan Sung, Tom Duerig, Yichang Chen, Feng Han, Sandeep Mariserla, Min Choi, Blair Chen, Aashi Jain, Raphael Hoffmann, Karan Gill, Zhe Dong, Zhe Li, Sai Meher Karthik Duddu, Zach Gleicher, Madhuri Shanbhogue, Sonam Goenka, Shuo Huang, Jinhyuk Lee, Vikram Rao, Xiaoqi Ren, Henrique Schechter Vera, Cathy Yip, Nithi Gupta, Koert Chen, Jay Han, Shanfeng Zhang, Trevor Walker, Wenlei Zhou, Fedor Moiseev, Mojtaba Seyedhosseini, Shahrokh Shahi, Gustavo Hernández Ábrego, Paul Suganthan, Daniel Salz, Parashar Shah, Feiyang Chen, Rakesh Ghiya, Daniel Cer, Ye Xia

- ***What's New***: Gemini Embedding는 구글의 대형 언어 모델인 Gemini를 활용하여 고도로 일반화된 임베딩 모델을 소개하는 최신 기술입니다. 이 모델은 다국어 및 코드 이해 능력을 기반으로 다양한 언어와 텍스트 모달리티에 걸쳐 뛰어난 임베딩을 생성합니다. Gemini Embedding은 MMTEB( Massive Multilingual Text Embedding Benchmark)에서 평가되었으며, 기존 최고 수준의 모델들을 크게 능가하는 임베딩 품질의 개선을 보여주었습니다.
- ***Technical Details***: Gemini Embedding 모델은 Gemini의 넓은 지식을 기반으로 초기화됩니다. 모델은 쿼리, 긍정적인 목표, 그리고 (선택적으로) 하드 네거티브 목표를 포함하는 예시와 함께 노이즈 대조 추정(NCE) 손실을 사용하여 훈련됩니다. 초미세 조정(pre-finetuning) 및 미세 조정(finetuning)이라는 두 단계의 훈련 파이프라인을 사용하며, 훈련 단계마다 배치를 유연하게 조정하여 모델의 일반화 성능을 극대화하기 위한 'Model Soup' 기술을 활용합니다.
- ***Performance Highlights***: Gemini Embedding은 MTEB(Multilingual) 및 MTEB(Eng, V2), MTEB(Code), XOR-Retrieve, 그리고 XTREME-UP 등의 벤치마크에서 최고 성능을 기록하였으며, 특히 분류 및 클러스터링, 검색과 같은 태스크에서 뛰어난 성능을 보여주었습니다. Borda 순위 기반 공식 순위에서 1위를 차지하고, 과제 평균 점수는 68.32로, 두 번째로 뛰어난 모델에 비해 +5.09가 개선되었습니다. XTREME-UP에서는 낮은 리소스의 언어들에서도 뛰어난 다중 언어 간 검색 성능을 보였습니다.

### [Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning](https://arxiv.org/abs/2503.07572)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.07572.png)

Vote: 18

Authors: Matthew Y. R. Yang, Edward Emanuel Beeching, Amrith Setlur, Aviral Kumar, Ruslan Salakhutdinov, Yuxiao Qu, Lewis Tunstall

- ***What's New***: 이 논문은 테스트 타임 컴퓨트(Test-Time Compute)를 최적화하기 위한 메타 강화 학습(Meta Reinforcement Learning; Meta RL) 방법을 제안합니다. MRT(Meta Reinforcement Fine-Tuning)는 기존의 결과 보상 기반 강화 학습 방식의 한계점을 개선하고 테스트 타임에서의 효율성과 성능을 크게 향상시키는 새로운 미세 조정 방법입니다.
- ***Technical Details***: MRT는 모델이 테스트 타임에서 추가적인 시퀀셜 에피소드를 통해 지속적인 '진척(progress)'을 이루도록 촉진하는 밀집 보상(dense reward) 시스템을 통하여 누적 후회(cumulative regret)를 최소화합니다. 이 방법은 대칭적인 정책 메타 프로버(meta-prover) 정책을 사용하여 에피소드마다 개선된 성공 가능성을 측정하고 이를 기반으로 최적화를 수행합니다.
- ***Performance Highlights***: MRT를 사용한 모델들은 기존의 결과 보상 기반 강화 학습보다 2-3배의 성능 향상을 보였으며, 토큰 효율성도 약 1.5배 개선되었습니다. 또한, 에피소드 수가 증가해도 누적 후회가 줄어드는 양상을 보여주며, 기존 방식에 비해 빠르고 안정적인 성능 개선이 이루어졌습니다.

### [Implicit Reasoning in Transformers is Reasoning through Shortcuts](https://arxiv.org/abs/2503.07604)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.07604.png)

Vote: 17

Authors: Siyu Yuan, Deqing Yang, Jian Xie, Tianhe Lin

- ***What's New***: 이 연구는 GPT-2와 같은 언어 모델(Language Models)이 다단계 수학적 문제를 해결하는 방식인 암묵적 추론(Implicit Reasoning)을 탐구하며, 이 과정에서 발생하는 단축 학습(Shortcut Learning)의 한계를 분석합니다. 이 연구는 암묵적 추론이 고급 추론 능력을 발현하지 못하는 이유를 살펴보았습니다.
- ***Technical Details***: 연구팀은 다단계 수학 문제를 다루기 위해 GPT-2 모델을 처음부터 새롭게 학습시키고, 모델이 암묵적 추론을 통해 단계별로 어떻게 문제를 해결하는지 분석했습니다. 이를 위해 특정 패턴의 데이터로 훈련된 모델은 높은 정확도로 문제를 해결할 수 있었으나, 불규칙한 패턴의 데이터로 훈련된 모델은 특정 패턴에 과대 적합되어 일반화에 실패했습니다.
- ***Performance Highlights***: 고정 패턴 데이터로 훈련된 모델은 동일한 패턴의 문제에서 높은 정확도를 보였지만, 변수로만 이루어진 감수(Subtrahend) 문제가 많아질수록 정확도가 급격히 떨어졌습니다. 최신 모델(SoTA LLMs) 역시 이러한 단축 학습에 의존해 암묵적 추론 능력을 충분히 발휘하지 못했습니다.

### [Tuning-Free Multi-Event Long Video Generation via Synchronized Coupled Sampling](https://arxiv.org/abs/2503.08605)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.08605.png)

Vote: 16

Authors: Jui-Hsien Wang, Jinwoo Shin, Seoung Wug Oh, Joon-Young Lee, Subin Kim

- ***What's New***: 이 논문에서는 Tuning-Free 방식으로 멀티 이벤트 장기 비디오 생성을 위한 새로운 추론 프레임워크인 SynCoS(Synchronized Coupled Sampling)를 제안합니다. SynCoS는 디퓨전 모델(Diffusion Model)의 동조 된 결합 샘플링을 통해 지역적 매끄러움(Local Smoothness)과 전역적 일관성(Global Coherence)을 동시에 보장하여 장기 비디오 생성을 가능하게 합니다.
- ***Technical Details***: SynCoS는 DDIM(Denoising Diffusion Implicit Models)과 CSD(Collaborative Score Distillation)라는 두 가지 상호 보완적인 샘플링 방법을 결합하여, 각 샘플링 단계에서의 중간 출력을 CSD 기반 최적화의 개선 소스로 사용하여 지역적으로 매끄러운 업데이트가 전역적 일관성을 강화하는 근거가 되도록 합니다. 고정된 베이스라인 노이즈(Fixed Baseline Noise)와 구조화된 프롬프트(Structured Prompt)를 활용하여, 샘플링 과정의 모든 단계가 조화롭게 동작할 수 있도록 보장합니다.
- ***Performance Highlights***: 광범위한 실험 결과에서 SynCoS는 시간적 일관성, 비디오 품질 및 프롬프트 충실도에서 기존의 조정 없는 방법들을 크게 능가했으며, 장기 비디오의 다이나믹한 장면 전환을 우수한 품질로 구현할 수 있음을 확인했습니다.

### [LightGen: Efficient Image Generation through Knowledge Distillation and Direct Preference Optimization](https://arxiv.org/abs/2503.08619)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.08619.png)

Vote: 15

Authors: Zihao Wang, Xianfeng Wu, Yexin Liu, Harry Yang, Wen-Jie Shu, Harold Haodong Chen, Xianzu Wu, Ser-Nam Lim, Yajing Bai, Haoze Zheng, Xuran Ma

- ***What's New***: LightGen은 지식 증류(Knowledge Distillation; KD) 및 직접 선호 최적화(Direct Preference Optimization; DPO)를 활용하여 효율적인 이미지 생성 훈련 패러다임을 소개합니다. 이 새로운 접근법은 방대한 데이터세트와 대규모 파라미터 구조에 대한 의존을 줄이면서 효율적이고 높은 품질의 이미지 생성을 가능하게 합니다.
- ***Technical Details***: LightGen은 소량의 고품질 합성 데이터세트를 사용하며, 강도 높은 데이터 증류 기법을 통해 최첨단 모델(SOTA)의 지식을 0.7B 파라미터를 가진 컴팩트한 마스크드 오토회귀 모델(Masked Autoregressive; MAR)에 증류합니다. 훈련 효율을 높이기 위해 라이트웨이트 아키텍처를 도입하고, DPO (Direct Preference Optimization) 기술을 통합하여 이미지 세부 사항 및 공간적 정확성을 향상시킵니다. 이런 설계는 훈련 자원을 크게 절감하면서 SOTA 모델과 비교할 만한 이미지 생성 품질을 제공합니다.
- ***Performance Highlights***:  LightGen은 GenEval 벤치마크에서 256 × 256 해상도에서 전반적으로 0.53의 성능 점수를 달성하며, 특히 내용과 색상 관련 이미지 생성에서 기존 SOTA 모델보다 우수한 성능을 보였습니다. 512 × 512 해상도에서는 전반적으로 0.62의 성능 점수를 기록하여 대부분의 SOTA 모델들을 뛰어넘는 결과를 보였습니다. DPO 통합은 특히 위치 정확도 및 고주파 세부 사항에서 성능을 강화하는 것으로 나타났습니다.

### [OmniMamba: Efficient and Unified Multimodal Understanding and Generation via State Space Models](https://arxiv.org/abs/2503.08686)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.08686.png)

Vote: 13

Authors: Jialv Zou, Wenyu Liu, Bencheng Liao, Xinggang Wang, Qian Zhang

- ***What's New***: OmniMamba는 선형 아키텍처 기반의 최초의 멀티모달 생성 모델로, 단 2M 이미지-텍스트 쌍으로 훈련되어 Show-o를 능가하고 JanusFlow와 견줄만한 성능을 보여줍니다. 새로운 '데커플드 보카블러리'(Decoupled Vocabularies)와 '태스크-스페시픽 로라'(Task-Specific LoRA)를 도입하여 모달리티 별 생성 및 태스크 적응성을 향상시켰습니다.
- ***Technical Details***: OmniMamba는 Mamba-2를 기반으로 하여, 텍스트와 이미지를 모두 생성하는 통합된 다음 토큰 예측 패러다임을 사용합니다. 이 모델은 '데커플드 보카블러리'(Decoupled Vocabularies)를 통해 모달리티별 생성을 유도하고 '태스크-스페시픽 로라'(Task-Specific LoRA)를 통해 파라미터 효율성을 높입니다. 또한 데이터 불균형 문제를 해결하기 위해 데커플드된 2단계 훈련 전략을 제시합니다.
- ***Performance Highlights***: OmniMamba는 개별 변환기 기반 모델들에 비해 최대 119.2배의 속도 향상과 63%의 GPU 메모리 감소를 달성했습니다. 이는 긴 시퀀스 생성 작업에서 특히 두드러지며, Show-o와 JanusFlow와 비교하여 상당한 속도 향상과 메모리 절감을 보여줍니다.

### [Exploiting Instruction-Following Retrievers for Malicious Information Retrieval](https://arxiv.org/abs/2503.08644)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.08644.png)

Vote: 12

Authors: Parishad BehnamGhader, Nicholas Meade, Siva Reddy

- ***What's New***: 이 연구는 대형 멀티모달 모델(LLMs; Large Language Models)과 함께 사용되는 지시-따르기 검색기(instruction-following retrievers)의 안전성 위험을 조사했습니다. 특히, 악의적인 정보 검색에 대한 검색기의 능력과 그로 인한 위험성을 실증적으로 연구한 최초의 작업입니다.
- ***Technical Details***: 본 연구에서는 NV-Embed와 LLM2Vec를 포함한 여섯 가지 강력한 검색기들이 악의적인 쿼리에 대해 얼마나 잘 수행하는지 평가하였습니다. 검색기들이 직접 접근 방식(direct approach)과 RAG 기반 접근 방식(Retrieval-Augmented Generation; RAG-based approach)을 통해 악의적인 정보를 어떻게 검색하는지도 분석되었습니다. 검색기들이 악의적인 요청에 대해 높은 정확도로 관련 정보를 선택할 수 있다는 것이 밝혀졌습니다.
- ***Performance Highlights***: LLM2Vec는 악의적인 쿼리에 대해 61.35%의 정확도로 관련 패세지를 선택했으며, 다른 검색기들도 대부분 쿼리의 50% 이상을 동일하게 처리했습니다. 기존의 안전하게 설계된 LLMs조차도 검색된 악의적 정보를 기반으로 악의적인 요청을 충족시킬 수 있음이 확인되었습니다.

### [Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment](https://arxiv.org/abs/2503.07334)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.07334.png)

Vote: 12

Authors: Xing Xie, Huijie Fan, Yandong Tang, Ziyue Lin, Zhi Han, Jiawei Liu, Liangqiong Qu

- ***What's New***: 이 연구에서는 새로운 훈련 프레임워크인 Autoregressive Representation Alignment (ARRA)을 제안하여 대형 언어 모델(Large Language Models; LLMs)을 활용한 텍스트-이미지 생성의 전반적인 잠재력을 최고로 발휘할 수 있도록 하였습니다. ARRA는 기존의 복잡한 아키텍처 수정을 필요로 하지 않고 LLM의 숨겨진 상태(hidden states)와 외부 시각 기본 모델의 시각적 표현을 정렬함으로써 텍스트-이미지 생성에서 전례 없는 전역적인 연결성을 달성합니다.
- ***Technical Details***: ARRA 프레임워크는 기존 아키텍처를 변경하지 않고 훈련 목표에 전역 시각 정렬 손실(global visual alignment loss)을 도입합니다. 시스템은 새로운 하이브리드 토큰 <HYBNEXT>를 활용, LLM의 숨겨진 상태와 외부 모델의 시각적 표현을 정렬함으로써 공간적 및 문맥적 일관성을 학습합니다. <HYBNEXT> 토큰은 지역적(next-token prediction) 및 전역적(semantic distillation) 제약을 같이 받으며, 시각적 일관성을 갖춘 이미지를 생성하는 데 도움을 줍니다.
- ***Performance Highlights***: ARRA는 다양한 도메인에서 효과적인 성능 향상을 보여주었습니다. 의료 이미징(MIMIC-CXR)에서 18.6%의 FID 감소, 그리고 이미지넷과 같은 자연 영상 데이터 세트에서도 FID를 각각 7.5%까지 줄였습니다. 이는 기존 아키텍처 변경 없이도 괄목할 만한 결과를 창출함을 시사합니다. 특히, ARRA는 LLMs가 기존의 텍스트 생성 기능을 활용하여 이미지 생성 분야에 적용 가능하게 함으로써, 도메인 적응의 새로운 가능성을 열어줍니다.

### [CineBrain: A Large-Scale Multi-Modal Brain Dataset During Naturalistic Audiovisual Narrative Processing](https://arxiv.org/abs/2503.06940)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.06940.png)

Vote: 11

Authors: Baofeng Yang, Jianxiong Gao, Yanwei Fu, Yichang Liu, Jianfeng Feng

- ***What's New***: CineBrain은 자연스러운 오디오비주얼 스토리 전개 중에 참가자의 뇌에서 EEG와 fMRI 신호를 동시에 수집한 최초의 대규모 멀티모달 데이터셋입니다. 이 데이터셋을 활용해, 뇌 신호로부터 비디오와 오디오 자극을 효과적으로 재구성할 수 있는 멀티모달 디코딩 프레임워크인 CineSync를 제안하며, 이는 EEG와 fMRI 신호를 융합하여 복잡한 뇌 동작 및 멀티모달 신경 디코딩 연구에 기여합니다.
- ***Technical Details***: CineSync는 Multi-Modal Fusion Encoder와 diffusion 기반 Neural Latent Decoder로 구성되어 fMRI와 EEG 신호를 융합합니다. 영상 자극 재구성을 위해 dual transformer 아키텍처를 사용하며, 각 모달리티의 다중 프레임 신호를 별도로 인코딩 후, 통합된 뇌의 표현을 디코더에 투입하여 비디오를 재구성합니다. 시각적 및 텍스트 표현과의 정렬은 결합된 대조 손실을 사용해 신경 신호에서 의미 있는 특징을 추출합니다.
- ***Performance Highlights***: CineSync는 영상 재구성에서 최첨단의 성능을 보여주며, EEG와 fMRI의 융합을 통해 시간적으로 역동적인 자극의 품질을 개선합니다. 특히 CineSync-EEG 및 CineSync-fMRI와 같은 단일 모달리티 기반 접근법보다 향상된 성능을 기록했으며, CineSync는 비디오와 오디오 모달리티 모두에서 Cine-Benchmark를 통해 평가되어 최고의 성과를 달성했습니다.

### [Robusto-1 Dataset: Comparing Humans and VLMs on real out-of-distribution Autonomous Driving VQA from Peru](https://arxiv.org/abs/2503.07587)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.07587.png)

Vote: 9

Authors: Arturo Deza, David Ortega, Victor Flores-Benites, Dunant Cusipuma

- ***What's New***: Robusto-1 데이터셋은 페루의 실제 OOD(Out-Of-Distribution) 자율 주행 환경에서 인간과 비주얼 언어 모델(Visual-Language Models, VLMs)의 인지적 정렬을 탐구할 수 있는 새로운 데이터를 제공하는 최초의 데이터셋입니다. 이는 비전 기반의 자율 주행 차량 내 OOD 상황에서 인간과 VLM의 시각적 질문-응답(VQA; Visual Question Answering) 능력을 비교해보는 새로운 방식입니다.
- ***Technical Details***: Robusto-1은 페루의 '위험한'(공격적인) 운전자로 유명한 도시들로부터 285개의 대시캠 영상을 수집하여, 200개의 5초 영상 클립으로 샘플링하여 구성하였습니다. 각 영상은 Oracle LLM에 의해 다양한 메타 데이터를 포함하여 총 15개의 질문이 생성되고, 인간과 VLM에게 질문하여 그들의 답변을 수집합니다. 또한 시스템 신경과학의 대표적인 방법인 표현 유사성 분석(RSA; Representational Similarity Analysis)을 통해 인지적 정렬 정도를 조사하였습니다.
- ***Performance Highlights***: VLM은 질문 종류에 관계 없이 놀라운 일관성을 보였으나, 인간의 반응은 변수별 질문에서는 높은 일치를, 반면 반실과 가설적 질문에서는 거의 유사성이 없는 결과를 보였습니다. 여러 시스템의 응답을 비교한 결과, 인간과 VLM이 어떻게 서로 다른 질문에 대한 답변으로 정렬되는지에 대한 패턴이 달라진다는 것을 발견했습니다. 이는 비디오와 함께 제공된 15개의 질문 세트에 대한 실험 결과로 부터 여러 유의미한 결론을 도출하는데 중요한 시사점을 제공합니다.

### [^RFLAV: Rolling Flow matching for infinite Audio Video generation](https://arxiv.org/abs/2503.08307)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.08307.png)

Vote: 7

Authors: Claudio Ferrari, Filippo Botti, Andrea Prati, Giuseppe Gabriele Tarollo, Alex Ergasti, Tomaso Fontanini, Massimo Bertozzi

- ***What's New***: RFLAV는 제한 없는 무한 오디오-비디오 생성(Infinite Audio-Video Generation)에 최적화된 새로운 트랜스포머 기반 아키텍처입니다. 시청각 동기화와 시간상의 일관성을 유지하며 연속성을 보장하면서 비디오를 생성할 수 있다는 점에서 이전 모델들보다 우수합니다.
- ***Technical Details***: RFLAV는 '리롤링 흐름 매칭(Rolling Flow Matching)' 방식을 채택하여 임의의 길이의 오디오-비디오(AV) 시퀀스를 생성합니다. 이 모델은 별도의 비디오 또는 오디오 인코더에 의존하지 않아 비디오의 길이가 미리 정해진 길이의 배수가 될 필요가 없습니다. 오디오 및 비디오의 교차 모달 상호작용을 효과적인 차세대 아키텍처를 통해 처리하며, 자체 주의 메커니즘을 배제한 경량 크로스-모달리티(intermodality) 융합 모듈을 제안합니다.
- ***Performance Highlights***: RFLAV는 AIST++와 Landscape 데이터셋에서 기존 최신 모델보다 뛰어난 성능을 보였습니다. 20개 스텝에서 FVD 50.92, KVD 8.73, FAD 8.40의 수치를 기록했고, 200개 스텝에서는 FVD 38.36, KVD 6.15, FAD 8.28로 개선된 성능을 보였습니다. 특히 AV-DiT와의 비교에서도 경쟁력 있는 FVD 및 KVD 결과를 보였으며, 다소 떨어지는 FAD 결과는 vocoder 사용 때문으로 보입니다.

### [BiasEdit: Debiasing Stereotyped Language Models via Model Editing](https://arxiv.org/abs/2503.08588)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.08588.png)

Vote: 6

Authors: Ningyu Zhang, Wei Xu, Julian McAuley, Xin Xu

- ***What's New***: BiasEdit은 경량 네트워크(Editor Hyper-Networks)를 활용하여 언어 모델의 편향을 효율적으로 수정하는 새로운 탈편향(Debiasing) 기법입니다. 이 방법론은 일부 모델 파라미터를 지역적으로 수정하여 사회적 고정관념의 편향을 줄이면서 언어 모델의 기존 기능을 유지합니다.
- ***Technical Details***: BiasEdit는 에디터 하이퍼 네트워크(Editor Hyper-Networks)를 활용하여 일부 파라미터를 업데이트하고, 탈편향 손실(Debiasing Loss)을 사용하여 편향을 수정합니다. 이와 동시에 보류 손실(Retention Loss)을 통해 언어 모델링 능력을 보존합니다. StereoSet 및 Crows-Pairs 데이터셋에서의 실험을 통해 BIASEDIT의 효과 및 효율성이 검증되었습니다.
- ***Performance Highlights***: BIASEDIT는 StereoSet과 Crows-Pairs 데이터셋을 기준으로 이전의 탈편향 방법들에 비해 뛰어난 성능을 보여줍니다. Stereotypical Score(SS)를 기존 모델의 60%에서 46% 이하로 감소시킬 수 있었으며, 언어 모델 구성 요소 편집이 모델의 언어 모델링 능력에 미치는 부정적인 영향을 최소화하였습니다. 또한, BIASEDIT는 젠더 반응 및 의미론적 일반화에 대한 강력한 성능을 보여주었습니다.

### ["Principal Components" Enable A New Language of Images](https://arxiv.org/abs/2503.08685)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.08685.png)

Vote: 6

Authors: Bingchen Zhao, Xiaojuan Qi, Xin Wen, Jiankang Deng, Ismail Elezi

- ***What's New***: 이 논문은 새로운 시각적 토큰화(visual tokenization) 프레임워크를 제안합니다. 이는 잠재적 토큰 공간(latent token space)에 증명 가능한 PCA-like 구조를 삽입하여, 기존 시각적 토크나이저들이 주로 재구성 충실도에 최적화되던 것에서 벗어나 잠재 공간의 구조적 특성을 간과하지 않도록 설계되었습니다.
- ***Technical Details***: 이 논문에서는 1D 인과(causal) 토큰 시퀀스를 통해 이미지를 생성하며, 각 후속 토큰은 주어진 설명 변수의 감소된 비중에 따라 비중요하지만 보완적인 정보를 제공합니다. 이는 주로 디퓨전 디코더(diffusion decoder)를 활용하여 높은 수준의 의미 콘텐츠와 낮은 수준의 스펙트럼 세부 정보를 분리하는 방법을 제안합니다.
- ***Performance Highlights***: 제안된 접근 방식은 ImageNet 검증 세트에서 최첨단의 재구성 FID 점수를 달성하였으며, 기존 최고 점수보다 거의 10% 개선된 성능을 보였습니다. 또한 SEMANTICIST가 생성한 토큰을 훈련한 자가회귀 모델(auto-regressive model)은 32개의 토큰만으로도 SOTA 모델과 성능을 비교할 수 있음을 보여주었습니다.

### [Perplexity Trap: PLM-Based Retrievers Overrate Low Perplexity Documents](https://arxiv.org/abs/2503.08684)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.08684.png)

Vote: 5

Authors: Liang Pang, Gang Wang, Xiao Zhang, Zhenhua Dong, Ji-Rong Wen, Haoyu Wang, Jun Xu, Sunhao Dai, Haiyuan Zhao

- ***What's New***: 이번 연구에서는 사전학습언어모델(Pretrained Language Model; PLM) 기반의 검색 모델들이 낮은 Perplexity를 가진 문서에 지나치게 높은 가중치를 부여하는 문제를 제기했습니다. 새로운 설명적 프레임워크인 Causal Diagnosis and Correction (CDC)을 통해 이러한 영향을 진단하고 수정하는 방법을 제안했습니다.
- ***Technical Details***: 이 연구에서는 인과 그래프를 구축하여 Perplexity가 검색 모델의 가중치 추정에 미치는 인과적 영향을 실험 및 이론 분석을 통해 설명합니다. Perplexity는 LLM에서 생성된 문서가 인간이 작성한 것보다 낮다는 사실을 기반으로 함을 확인하였고, 이러한 원천 편향을 제거하기 위해, CDC라는 추론 시간에서 편향 제거 방법을 제안하였습니다. 이는 모델을 재훈련 하지 않고 편향된 효과를 분리하여 교정된 점수를 제공합니다.
- ***Performance Highlights***: 실험 결과, CDC는 여러 도메인에서 원천 편향을 효과적으로 제거하며, 검색 모델의 성능이 크게 저하되지 않음을 확인했습니다. 실험에서 CDC는 다양한 PLM 기반 검색 모델에 적용되었으며 새로운 LLM 및 데이터 도메인에서도 일반화 가능성을 입증했습니다.

### [RayFlow: Instance-Aware Diffusion Acceleration via Adaptive Flow Trajectories](https://arxiv.org/abs/2503.07699)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.07699.png)

Vote: 5

Authors: Huiyang Shao, Xin Xia, Yuxi Ren, Xing Wang, Xuefeng Xiao, Yuhong Yang

- ***What's New***: RayFlow는 인스턴스 특유의 경로를 이용해 샘플을 타겟 분포로 안내하여 샘플링 과정을 최소화하면서도 생성 다양성과 안정성을 유지하는 새로운 디퓨전 모델 프레임워크입니다. 또한 중요한 타임스텝에 중점을 둔 중요도 샘플링 기법인 Time Sampler를 도입하여 기존 가속화 기법이 가지고 있는 제약을 극복합니다.
- ***Technical Details***: RayFlow는 각 샘플이 고유한 경로를 따라 이동하도록 함으로써 기존 방법에서 발생하는 경로 중복과 샘플링 불안정 문제를 최소화합니다. 이 과정에서 미리 학습된 모델을 활용하여 일관성 있는 노이즈 기대값을 계산하고, 이를 바탕으로 단계 압축을 효율적으로 수행합니다. Time Sampler는 Stochastic Stein Discrepancies (SSD)를 기반으로 훈련 중 중요한 시간단계를 식별하여 효율성을 높입니다.
- ***Performance Highlights***: RayFlow는 기존 가속화 알고리즘에 비해 뛰어난 이미지 품질과 속도, 제어력을 제공하며, 여러 실험을 통해 그 효과가 입증되었습니다. 특히, RayFlow를 활용한 모델 SD15-Ray는 1-8 단계 샘플링에서 클립 점수와 Aes 점수, FID 점수에서 탁월한 성능 개선을 보였습니다. 여기에 Time Sampler 모듈을 적용한 경우, 더욱 향상된 결과가 나타났습니다. 다양한 데이터셋과 메트릭을 통해 평가한 결과, RayFlow의 고효율 텍스트-이미지 생성에 대한 우수한 성능이 검증되었습니다.

### [Benchmarking AI Models in Software Engineering: A Review, Search Tool, and Enhancement Protocol](https://arxiv.org/abs/2503.05860)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.05860.png)

Vote: 5

Authors: Philippe de Bekker, Roham Koohestani, Maliheh Izadi

- ***What's New***: 이 논문에서는 AI4SE(AI 팀의 소프트웨어 공학)에서 인공지능 모델을 벤치마킹하기 위한 'BenchScout'라는 새로운 검색 도구와 'BenchFrame'이라는 벤치마크 향상 프로토콜을 소개합니다. 특히, HumanEval이라는 벤치마크를 'BenchFrame'을 통해 강화하여 'HumanEvalNext'를 개발하였습니다. 이러한 접근법을 통해 높은 성능을 검증받은 최신 언어모델들 역시 새로운 벤치마크에서는 성능이 큰 폭으로 감소하였음을 보여주었습니다. 이는 강건하고 정밀한 벤치마크가 미래의 연구를 안내하는 이번 연구의 중요성을 강조합니다, 더해 벤치마크 선택이 수월해지는 BenchScout 툴 또한 개발되었습니다. 이 도구는 AI4SE(AI in Software Engineering) 분야의 모델을 평가하기 위한 적합한 벤치마크를 쉽고 효율적으로 찾는 데 도움을 줍니다. 이 확장형 의미 기반 검색 도구와 BenchFrame 개선 프로토콜을 개발하여, 폭넓은 사용자 층을 대상으로 툴의 유용성을 평가했습니다. 이후 이는 HumanEval에 적용되어 HumanEvalNext라는 개선된 벤치마크로 결과를 평가했습니다. 결과적으로 우수한 성능의 최신 LLM(대형 언어 모델)을 대상으로 HumanEvalNext 벤치마크를 활용해 모델의 전반적인 성능이 HumanEval과 비교하여 pass@1 스코어가 평균 31.2%(중간값 26.02%) 감소하였다는 결과를 확인했습니다. 이로써 더 엄격한 벤치마크의 필요성을 확인하였습니다. 이 연구는 AI 기반 소프트웨어 공학(AI4SE)의 신뢰할 수 있는 벤치마크가 더 견고한 모델 개발을 이끄는 데 있어 얼마나 중요한지를 강조합니다. BenchFrame을 통한 HumanEval 벤치마크 향상이 실질적인 모델 평가 제공에 얼마나 중요한지를 증명했습니다. BenchScout는 이러한 과정을 더욱 향상시켜, 연구원이 자신의 필요에 맞춘 관련 벤치마크를 발견하기 쉽게 해 줍니다. benCheScout을 사용하여 사용자들은 모델의 성능에 대한 깊은 통찰력을 제공받을 수 있습니다. 또한 다채로운 AI4SE 태스크에서 벤치마크를 정련하는 것이 향후 연구를 더 잘 이끌기 위해 필수적임을 강조합니다. 벤치마크가 모델 발전에 따라 점진적으로 도전 과제가 증가하도록 발전해야 함을 제언합니다. 
- **Technical Details**: 본 연구는 2014년 이후 AI4SE 벤치마크를 체계적으로 검토하여 173건의 연구에서 204개의 벤치마크를 식별하였으며 주요 메타데이터를 구조화하고, BenchScout(벰스카우트)을 개발하였습니다. 이 도구는 관련 연구에 대한 2D 맵을 제공하기 위해 문헌의 맥락을 클러스터링하고, 각 연구에 대한 주요 메타데이터를 수동으로 추출하여 AI4SE 벤치마크의 광경을 시각화하였습니다. 사용자 연구를 통해 4.5, 4.0, 4.1의 평균 점수를 획득하여 서칭 툴의 유용성, 효율성 및 직관성을 평가하였습니다. 또한, BenchFrame이라는 통합 메소드로 벤치마크의 품질을 향상시키기 위한 방안을 제안하였으며, 인기를 끈 HumanEval 벤치마크를 대상으로 하여 HumanEvalNext의 보다 열거된 버전을 제작하였습니다. 다양한 신기술 모델을 위 벤치마크에서 평가한 결과, 각기 다른 HumanEval 및 HumanEvalPlus에 비해 더 강한 난이도로 인증하여 벤치마킹의 정밀성과 현실감을 향상시켰습니다. 이번 연구의 발견 사항은 학계와 산업계가 AI4SE 벤치마크 신뢰성의 중요성을 강조하며, 개발자에게 보다 의미 있는 평가 환경을 제공할 가능성이 있습니다. 동시에, BenchScout 도구가 구체적인 평가 작업에 적합한 벤치마크를 식별하는데 효율성을 입증하였습니다. Benchmark를 개선하려면 BenchFrame 같은 통일된 기준의 적용이 요구됩니다. 발견된 데이터 누출 문제에도 불구하고, 지속적인 벤치마크 방법론 개선이 필요하다는 것을 보여줍니다. 추가적으로 BenchFrame은 여전히 주요 비교 벤치 마크의 품질 및 신뢰성을 향상하는데 사용됩니다. 향후 BenchFrame에 대한 연구는 다른 프로그래밍 언어에 더 확장하는 데 초점을 맞출 것이며 검증 패키지에 모든 데이터를 공개합니다.

### [QuoTA: Query-oriented Token Assignment via CoT Query Decouple for Long Video Comprehension](https://arxiv.org/abs/2503.08689)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.08689.png)

Vote: 4

Authors: Jinfa Huang, Wang Chen, Haojia Lin, Xiawu Zheng, Yongdong Luo, Weizhong Huang, Rongrong Ji, Jiayi Ji, Jiebo Luo, Chaoyou Fu, Shukang Yin

- ***What's New***: QuoTA는 기존의 대형 비디오-언어 모델(Large Video-Language Models; LVLMs)에 새로운 모듈로서 통합되어, 쿼리 지향적 프레임 중요도 평가를 기반으로 하는 시각적 토큰 할당(Query-oriented Token Assignment)을 제안합니다. 이는 교차 모달 상호 작용 전에 프레임 수준의 중요도를 전략적으로 평가하여 사전(pre-hoc)으로 시각적 토큰을 할당함으로써, 명령어(query)에 해당하는 의미 있는 콘텐츠를 보존하며 토큰 자원의 최적 사용을 가능하게 합니다.
- ***Technical Details***: QuoTA는 (i) 체인 오브 쏘츠(Chain-of-Thoughts) 추론을 통해 명령어(query)를 분리하여 더 정밀한 프레임 중요도 점수를 생성하고, (ii) 이 점수를 이용하여 시각적 토큰 할당을 수행합니다. 이는 시각적 처리 과정을 작업 특정 요구사항과 정렬하는 것을 중점으로 하는 쿼리 지향적 토큰 선택을 통해 작업 성능을 최적화하고, 의미 있는 콘텐츠를 보존하는 데 기여합니다. QuoTA는 또한 기존의 대형 비디오-언어 모델(Large Video-Language Models; LVLMs)에 플러그 앤 플레이 방식으로 확장될 수 있는 기능을 제공합니다.
- ***Performance Highlights***: 기존 LVLM에 QuoTA를 적용한 결과로, LLaVA-Video-7B를 사용하여 Video-MME, MLVU와 같은 여섯 개의 벤치마크에서 평균 3.2%의 성능 향상을 달성하였습니다. 특히 Video-MME에서 63.3%의 pass@full에서 65.9%로, 최선의 7B LVLM 중 다섯 개의 비디오 벤치마크에서 최고의 결과를 기록했습니다. QuoTA는 다른 최신 모델들인 AIM 및 FrameFusion 보다 뛰어난 성과를 보였습니다.

### [AnyMoLe: Any Character Motion In-betweening Leveraging Video Diffusion Models](https://arxiv.org/abs/2503.08417)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.08417.png)

Vote: 4

Authors: Seokhyeon Hong, Junyong Noh, Chaelin Kim, Kwan Yun

- ***What's New***: AnyMoLe는 비디오 확산 모델(Video Diffusion Models)을 활용하여 외부 데이터 없이 임의의 캐릭터를 위한 모션 인비트위닝(Motion In-betweening)을 생성하는 새로운 방법을 제시합니다. ICAdapt라는 비디오 확산 모델의 미세 조정 기술을 도입해 실세계와 렌더된 캐릭터 간의 도메인 간극을 극복하고, '모션-비디오 미미킹(Motion-Video Mimicking)' 최적화를 통해 임의의 관절 구조를 가진 캐릭터의 부드러운 모션 생성을 가능하게 합니다.
- ***Technical Details***: AnyMoLe는 두 단계의 프레임 생성 프로세스를 통해 컨텍스트 이해도를 향상시킵니다. 첫 번째 단계는 움직임 구조를 설정하는 희소 프레임을 생성하고, 두 번째 단계는 세부 사항을 채워넣는 밀집 프레임을 생성합니다. 도메인 간극을 줄이기 위해 ICAdapt를 사용하여 비디오 확산 모델의 공간 모듈만을 미세 조정하며, 이 과정에서 시간 모듈은 고정되어 원래의 모션 다이나믹스를 유지합니다. 또한, 장면별 관절 추정기를 새롭게 제안하여 맥락 프레임과 키프레임만을 사용해 훈련하여 임의의 캐릭터에 대한 모션 추정을 지원합니다.
- ***Performance Highlights***: AnyMoLe는 이전 방법들과 차별되게 외부 데이터 없이 임의의 캐릭터에 대해 3D 모션을 생성할 수 있는 첫 모션 인비트위닝 방법입니다. 실험 결과, 제안된 방법은 기존의 방법들보다 상당한 성능 향상을 보여 주었으며, 특히 다양한 캐릭터에 걸쳐 높은 정밀도와 자연스러운 전환을 실현했습니다.

### [Referring to Any Person](https://arxiv.org/abs/2503.08507)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.08507.png)

Vote: 4

Authors: Zhaoyang Zeng, Yuda Xiong, Lei Zhang, Qing Jiang, Lin Wu, Qin Liu, Tianhe Ren, Yihao Chen

- **What's New**: 이 논문에서는 'referring to any person'이라는 새로운 태스크를 도입하여 자연어 설명에 기반해 이미지 내 여러 개체를 탐지하는 모델 RexSeek을 제안했습니다. 이는 기존의 단일 개체 탐지 한계를 넘어서 다양하고 복잡한 실제 애플리케이션 환경을 반영하는 새로운 데이터세트 HumanRef를 소개합니다.
- **Technical Details**: RexSeek 모델은 멀티모달 대형 언어 모델 (Multimodal Large Language Model)과 물체 탐지 프레임워크를 결합하여 생성되었습니다. HumanRef 데이터세트는 시각적 이해 및 추론 능력을 평가하기 위해 고안되었으며, 103,028개의 지칭 표현과 평균 2.2개의 인스턴스를 포함합니다. 다단계 훈련 과정을 통해 탐지 및 언어 이해 능력을 향상시켰으며, 복잡한 언어 설명을 해석하는 능력을 강화했습니다.
- **Performance Highlights**: HumanRef 벤치마크에서 RexSeek 모델은 채용된 다단계 트레이닝 접근 방식을 통해 기존 모델들이 갖는 다중 인스턴스 탐지 및 환각 이슈들을 성공적으로 해결했습니다. RexSeek은 RefCOCO/+/g와 같은 기존 벤치마크에서 성능을 발휘하며, 특히 많은 인스턴스를 탐지하는 작업에서도 높은 성능을 유지합니다.

### [DiffCLIP: Differential Attention Meets CLIP](https://arxiv.org/abs/2503.06626)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.06626.png)

Vote: 4

Authors: Hasan Abed Al Kader Hammoud, Bernard Ghanem

- ***What's New***: DiffCLIP는 CLIP 기반의 비전-언어 모델(Vision-Language Models; VLMs)에서 주의력 노이즈를 감소시키기 위해 차등 주의력 메커니즘을 통합한 최초의 모델입니다. 이 접근법은 CLIP의 이중 인코더(이미지 및 텍스트)에 차등 주의력을 통합하여 미세한 신호 정렬을 가능하게 합니다.
- ***Technical Details***: DiffCLIP은 CLIP의 비전과 텍스트 인코더 모두에 주의력 계층을 차등 주의력(Differential Attention)으로 대체하여 성능을 향상시킵니다. 두 개의 주의력 지도를 학습하고 하나를 다른 하나에서 빼는 방식을 통해 잘못 정렬되거나 잡음이 많은 신호를 효과적으로 제거합니다. 이 개선점은 계산비용과 매개변수 증가가 미미한 수준입니다. DiffCLIP는 CC3M과 CC12M 데이터를 사용하여 여러 이미지-텍스트 이해 작업에 대해 실험적으로 검증되었습니다.
- ***Performance Highlights***: DiffCLIP는 Zero-shot ImageNet 및 다양한 분산 데이터셋에 대해 성능이 평균 2.1% 향상되었습니다. 또한 Fine-Grained Vision Tasks인 MMVP-VLM 벤치마크에서 CLIP 대비 5.7% 개선된 정확도를 보입니다. 이는 DiffCLIP가 미세한 시각적 특성까지 더 잘 파악할 수 있음을 나타냅니다.

### [Evaluating Intelligence via Trial and Error](https://arxiv.org/abs/2502.18858)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.18858.png)

Vote: 4

Authors: Yiqun Liu, Jingtao Zhan, Qingyao Ai, Min Zhang, Jiaxin Mao, Shaoping Ma, Hongning Wang, Jiayu Li, Bo Zhang, Jiahao Zhao

- ***What's New***: Survival Game은 시도와 오류를 통해 지능을 평가하는 새로운 프레임워크입니다. 성공적인 솔루션을 찾기 전까지의 실패 횟수에 기반해 지능을 측정하며, 실패가 적을수록 더 높은 지능을 나타냅니다.
- ***Technical Details***: Survival Game에서는 실패 횟수를 이산 확률 변수로 모델링하며, 통계적 기준을 사용해 지능을 평가합니다. 기대치와 분산이 유한할 때 지능을 'Autonomous Level'이라고 정의하며, 이는 안정적으로 새로운 과제를 해결할 수 있는 능력을 나타냅니다. 실패 횟수의 분포에 기반하여 지능을 세 가지 수준, Limited, Capable, Autonomous로 분류합니다.
- ***Performance Highlights***: 현재 AI 시스템은 간단한 작업에서는 Autonomous Level에 도달할 수 있지만, 복잡한 작업에서는 대부분 Limited Level에 머물러 있습니다. 복잡한 작업의 Autonomous Level을 달성하려면 1026 개의 파라미터가 필요하며, 이는 현재 하드웨어와 테크놀로지로는 거의 불가능한 숫자입니다.

### [VisualSimpleQA: A Benchmark for Decoupled Evaluation of Large Vision-Language Models in Fact-Seeking Question Answering](https://arxiv.org/abs/2503.06492)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.06492.png)

Vote: 4

Authors: Jing Zhang, Yihan Zhao, Yanling Wang, Haoyang Li, Shasha Guo, Qi Li, Ke Xu, Lixin Liu, Yong Xiao, Xiaodong Chen

- ***What's New***: VisualSimpleQA는 대형 비전-언어 모델(Large Vision-Language Models; LVLMs)의 사실 탐구형 질문 답변(Fact-Seeking Question Answering)에서 시각 및 언어적 모듈의 독립적인 평가를 가능하게 합니다. 이는 현재의 멀티모달 벤치마크가 주로 모델의 출력과 정답을 비교하는 데 중점을 두는 것과 달리, 개별 모듈의 성능에 대한 통찰을 제공합니다. 또한, VisualSimpleQA는 난이도 기준을 명확히 하여 사람의 주석을 안내하고, 더 어려운 하위집합인 VisualSimpleQA-hard를 추출할 수 있도록 합니다.
- ***Technical Details***: VisualSimpleQA는 500개의 잘 주석된 샘플로 구성되어 있으며, 이 중 300개는 기존의 이미지 데이터셋에서, 200개는 인터넷에서 새롭게 수집되었습니다. 각 샘플은 이미지 데이터, 멀티모달 질문, 해당 질문의 언어 전용 질문, 정확한 정답, 근거, 주석 및 태그를 포함합니다. 난이도 평가는 해상도(Resolution), 관심 영역 비율(Proportion of ROI), 근거의 상세도(Rationale Granularity), 이미지 내 텍스트 존재 여부(Presence or Absence of Text in Image), 지식 대중성(Knowledge Popularity) 등의 기준으로 정의됩니다.
- ***Performance Highlights***: VisualSimpleQA에서 GPT-4o와 같은 최첨단 모델은 멀티모달 질문에 대해 60% 이상의 정답률을 기록하였으며, VisualSimpleQA-hard에서는 30%+의 정답률만을 기록했습니다. 이는 현재 LVLMs가 개선의 여지가 상당하다는 것을 시사합니다. 대형 모델이 작은 모델에 비해 뚜렷한 성능 차이를 보이며 우수한 성과를 낼 수 있음을 확인하였습니다. 특히, 오픈소스 모델은 닫힌 소스 모델에 비해 더 큰 비율의 성능 저하를 보이며, 시각적 인식 작업에서 도전 과제가 여전히 존재함을 보여줍니다.

### [Beyond Decoder-only: Large Language Models Can be Good Encoders for Machine Translation](https://arxiv.org/abs/2503.06594)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.06594.png)

Vote: 3

Authors: Ziqiang Xu, Yongyu Mu, Bei Li, Yongqi Gao, Xiaoqian Liu, Yingfeng Luo, Qinghong Zhang, Tong Xiao, Jingbo Zhu, Tong Zheng, Peinan Feng

- ***What's New***: 이번 연구는 라마테(LaMaTE; Large Language Models as Machine Translation Encoders)라는 새로운 방법론을 제시하여, 대규모 언어 모델(LLM)을 기계 번역(NMT)에서 인코더로 사용하는 접근법을 제시합니다. 이는 LLM의 풍부한 표현력을 활용하여, 효율적이면서도 투명하게 최적화 가능한 번역 시스템을 목표로 합니다.
- ***Technical Details***: LaMaTE 모델은 고유의 인코더-디코더(Encoder-Decoder) 아키텍처를 따르며, LLM 디코더를 NMT 인코더로 대체합니다. 인코더는 소스 언어를 이해하는 강력한 능력을 제공하고, 디코더는 고품질 번역을 낮은 디코딩 비용으로 생성 가능합니다. LLM의 크기가 크기 때문에 고유한 어댑터(Adaptor)를 사용하여 LLM의 출력 표현을 NMT 디코더에 맞게 조정합니다. 또한, 새로운 번역 모델의 일반화 능력을 평가하기 위해 종합 기계 번역 벤치마크(ComMT)를 구축했습니다.
- ***Performance Highlights***: LaMaTE 모델은 기존의 여러 번역 시스템보다 2.4 ∼6.5배의 디코딩 속도 향상과 75%의 KV 캐시 메모리 사용량 감소를 달성하였습니다. ComMT 데이터셋에 대한 평가에서도 뛰어난 일반화 능력을 발휘, 여러 작업에서 기준 시스템보다 우수한 성능을 보였습니다.

### [Promote, Suppress, Iterate: How Language Models Answer One-to-Many Factual Queries](https://arxiv.org/abs/2502.20475)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.20475.png)

Vote: 2

Authors: Tianyi Lorena Yan, Robin Jia

- ***What's New***: 이 논문은 언어 모델(LM)이 하나의 주제에 대해 여러 답을 제공해야 하는 질문에 대해 어떻게 회신하는지에 대한 연구를 제공합니다. 저자들은 '프로모트-억제' 메커니즘, 즉 모든 가능한 답을 먼저 떠올린 후 이미 생성된 답을 억제하는 방법을 발견했습니다.
- ***Technical Details***: 논문에서 소개하는 방법론으로는 'Token Lens' 및 'attention knockout' 기술을 사용하여 LM의 주목(attention) 및 MLP가 서로 다른 입력 토큰과 어떻게 상호작용하는지를 분석했습니다. 'Token Lens'는 특정 토큰이 주목할 때의 결과를 집계하고 다시 분해하여 각 토큰이 출력 토큰을 어떻게 촉진하거나 억제하는지를 관찰할 수 있게 합니다.
- ***Performance Highlights***: 실험은 Llama-3-8B-Instruct와 Mistral-7B-Instruct 모델을 사용해 수행되었으며, 두 모델 모두 국가의 도시, 아티스트의 노래, 그리고 배우의 영화 같은 데이터셋에서 높은 정확도를 보였습니다. 특히, 주목과 MLP는 서로 다른 입력 토큰에서 정보를 이용하여 새로운 답을 찾고 중복된 답을 억제하는 комплекс한 작업을 수행했습니다.

### [AI-native Memory 2.0: Second Me](https://arxiv.org/abs/2503.08102)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.08102.png)

Vote: 2

Authors: Jiale Wei, Xiang Ying, Felix Tao, Tao Gao, Jingbo Shang

- ***What's New***: AI-native Memory 2.0: Second Me는 AI의 등장으로 개인화된 메모리 관리 시스템을 재정의하며, 사용자 특화 지식을 보유하고 컨텍스트에 맞는 답변 생성을 통해 상호작용의 효율성을 높이는 시스템입니다. 단순한 데이터 저장소를 넘어 대규모 언어 모델(LLM)의 메모리 파라미터화를 활용하여 구조화된 조직화, 컨텍스트 추론 및 적응형 지식 검색을 수행하며, 개인화된 AI 에이전트로서의 역할을 수행합니다. 이는 지속적이고 자기 최적화하는 메모리 시스템으로서의 AI-native 패러다임의 중요한 진전을 나타냅니다.
- ***Technical Details***: SECOND ME는 LLM 기반 파라미터화를 활용하여 구조화된 데이터 조직화, 컨텍스트 추론 및 적응형 지식 검색을 가능하게 합니다. 다양한 데이터 소스 및 훈련 스타일을 탐구하며, 감독 학습 튜닝(SFT) 및 Direct Preference Optimization(DPO)를 통합하여 LLM 성능을 향상시킵니다. 주요 과제로는 메모리 기반 다중 시점 Q&A, 사용자 요구에 따른 컨텍스트 완성, 사용자 선호도 및 외부 응답을 포함한 컨텍스트 비평이 포함됩니다. 이 시스템의 핵심은 사용자 데이터의 완전한 보안을 유지하면서도 학습 목표를 달성하기 위해 2단계 훈련 파이프라인과 자동화된 데이터 합성 전략을 실행하는 것입니다.
- ***Performance Highlights***: SECOND ME는 메모리와 관련된 질문에 답하고 전문가와의 소통을 지원하는 강력한 체인을 활용하여 모델 성능을 향상시킵니다. 실험 결과, 다양한 데이터 소스와 체인-오브-띵(Chain-of-Thought; COT) 스타일의 정규화를 결합하여 자동 평가에서 SECOND ME의 성능을 극대화할 수 있었습니다. DPO 기법의 사용은 성능을 더욱 개선시켰으며, 인간 사례 연구에서도 보이는 효과는 상당했습니다.

### [NullFace: Training-Free Localized Face Anonymization](https://arxiv.org/abs/2503.08478)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.08478.png)

Vote: 2

Authors: Nicu Sebe, Terence Sim, Han-Wei Kung, Tuomas Varanka

- ***What's New***: NullFace는 훈련 없이 텍스트-이미지(Diffusion Model)를 활용해 얼굴을 익명화하는 새로운 방법을 소개합니다. 이 방법은 랜드마크나 기타 정밀한 컨디셔닝 데이터에 의존하지 않고, 식별과 무관한 중요한 얼굴 속성을 유지하면서 익명화를 수행할 수 있습니다. 또한 특정 얼굴 영역만을 선택적으로 익명화하는 'Localized Anonymization' 기능을 제공하여 사용자가 얼굴의 특정 부분을 보호할 수 있습니다.
- ***Technical Details***: NullFace는 사전 훈련된 Text-to-Image Diffusion 모델을 이용하여 초기 노이즈를 역추산한 후, 식별된 임베딩을 수정하여 원래의 신원을 대체하는 Denoising Diffusion Probabilistic Model(DDPM)을 사용합니다. ID 임베딩(Embeddings)을 사용하여 로컬 라이즈드(Localization)된 익명화가 가능하며, 세그멘테이션 맵을 사용하여 마스킹 및 특정 얼굴 속성을 보존할 수 있습니다. IP-Adapter는 ID 임베딩 컨디셔닝을 가능하게 하여 신원 보존 대신 익명화하는 과정을 촉진합니다.
- ***Performance Highlights***: NullFace는 CelebA-HQ 및 FFHQ 데이터셋에서 최고 수준의 성능을 보여주며 익명화뿐만 아니라 이미지 품질 유지, 속성 보존에서도 좋은 성과를 보였습니다. 이전 방법들이 직면한 한계를 극복하고, 사람의 신원을 흐리면서도 머리 포즈, 표현, 시선 같은 비신원 속성을 잘 유지합니다.

### [Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts](https://arxiv.org/abs/2503.05066)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.05066.png)

Vote: 2

Authors: Weilin Cai, Jiayi Huang, Shwai He, Ang Li

- ***What's New***: 이 논문은 전문가 혼합(Mixture of Experts; MoE) 구조의 '느린 전문가(Straggler) 현상'을 완화하기 위해 '용량 인식 추론(Capacity-Aware Inference)'이라는 새로운 접근 방식을 제안합니다. 이 접근 방식은 과부하된 전문가의 최대 지연 시간을 조절하는 '용량 인식 토큰 드롭(Capacity-Aware Token Drop)'과 오버로드된 토큰을 덜 활용되는 전문가에게 재할당하는 '토큰 재라우팅(Tokens Reroute)' 두 가지 핵심 기술을 포함합니다.
- ***Technical Details***: 제안된 방법은 MoE 구조의 전문가 병렬성 환경에서 토큰-전문가 할당 불균형 문제를 해결하기 위한 것입니다. '용량 인식 토큰 드롭(Capacity-Aware Token Drop)'은 과도하게 할당된 비율이 높은 토큰을 삭제하여 모듈의 효율성을 향상시킵니다. '용량 인식 토큰 재라우팅'은 활용도가 낮은 전문가를 활용하여 오버플로우된 토큰들을 재할당함으로써 토큰 할당의 균형을 맞춥니다. 제안된 방법은 고부하와 저부하 전문가의 효율적인 활용으로 MoE 추론 파이프라인을 최적화합니다.
- ***Performance Highlights***: 이 논문에서 제안된 방법의 실험 결과, 혼합 전문가들 간의 효율성을 개선하는 데에 있어서 상당한 성과 향상을 기록했습니다. 예를 들어, Mixtral-8×7B-Instruct 모델에서 평균 성능이 0.2% 증가하고 추론 속도가 1.94배 증가했습니다. 이는 특히 용량 인식 추론이 고부하 전문가로 인한 대기시간을 효과적으로 줄이는 데 기여했음을 시사합니다.

### [Mixture of Experts Made Intrinsically Interpretable](https://arxiv.org/abs/2503.07639)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.07639.png)

Vote: 2

Authors: Puneet K. Dokania, Philip Torr, Adel Bibi, Ashkan Khakzar, Christian Schroeder de Witt, Xingyi Yang, Constantin Venhoff

- ***What's New***: 이 논문은 언어 모델의 해석 가능성을 내재적으로 향상시키기 위해 설계된 Mixture-of-Experts(MoE) 아키텍처, MoE-X를 소개합니다. MoE-X는 각 입력에 대해 전문가의 일부만 활성화하여 스파스하고 해석 가능한 구조를 형성합니다. 이는 기존의 포스트-호크 메서드에 의존하지 않고 모델의 내재적 해석 가능성을 목표로 한다는 점에서 차별화됩니다.
- ***Technical Details***: MoE-X는 ReLU 활성화 함수를 사용하여 각 전문가 내에서 스파스 버전의 활성화를 촉진함으로써 명백한 내장 스파스 다층 퍼셉트론(MLP)을 구현합니다. 또한, Sparsity-Aware Routing 메커니즘을 도입하여 가장 적은 양의 활성화를 생성하는 전문가를 우선으로 합니다. MoE-X 레이어는 폭넓은 심층의 MLP와 동일한 기능을 수행하며, 예측을 위해 전문가들의 출력을 동적으로 결합합니다.
- ***Performance Highlights***: 체스와 언어 데이터셋 실험 결과, MoE-X 모델은 밀집 트랜스포머(dense transformers)와 유사하거나 이를 초과하는 성능을 나타내었으며, 동시에 명확하고 해석 가능한 표현을 제공했습니다. MoE-X는 GPT-2보다 우수한 Perplexity를 달성했으며, Sparse Autoencoder(SAE) 기반 접근법보다 더 나은 해석 가능성을 보였습니다. 특히, 체스 이동 예측에서 F1 점수가 25% 증가하였음을 나타냈습니다.

### [TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models](https://arxiv.org/abs/2503.07389)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.07389.png)

Vote: 2

Authors: Lanjun Wang, Honglin Guo, Weizhi Nie, An-An Liu, Chenyu Zhang, Ruidong Chen

- ***What's New***: TRCE는 텍스트-이미지 확산 모델(Text-to-Image Diffusion Models)에서 악의적 개념을 안정적으로 제거하기 위한 이중 단계 제거 전략을 제안합니다. 첫 번째 단계에서는 [EoT](End of Text) 임베딩을 결정적인 매핑 목적으로 식별하여 부정적 의미를 효과적으로 제거합니다. 두 번째 단계에서는 초기 디노이즈(Denoising) 예측을 보다 안전한 방향으로 조정함으로써 모델의 기존 생성 능력을 보다 잘 보존합니다.
- ***Technical Details***: TRCE는 'Textual Semantic Erasure'와 'Denoising Trajectory Steering'으로 구성된 이단계(두 단계) 사상 제거 전략을 제안합니다. 'Textual Semantic Erasure' 단계에서는 [EoT] embedding에 초점을 맞추어 모델 파라미터를 조정하여 악의적 개념의 영향을 억제합니다. 'Denoising Trajectory Steering'(Denoising Trajectory Steering) 단계에서는 초기 디노이즈 예측을 안전한 방향으로 이끌어 비 악의적 콘텐츠 생성 및 모델의 고유한 지식을 잘 보존합니다. 종합 시험을 거쳐 TRCE는 적대적 프롬프트(Adversarial Prompts)에서도 안정적인 사상 제거 성능을 입증했습니다. 또한 모델의 본래 생성 능력을 잘 보존함을 확인할 수 있었습니다. 제공 코드는 http://github.com/ddgoodgood/TRCE 입니다. 이 논문은 불쾌한 콘텐츠를 담고 있을 수 있는 모델 생성 콘텐츠를 포함하고 있음을 주의하십시오. (This paper includes model-generated content that may contain offensive material.)에서 모델에 의해 생성된 불쾌한 콘텐츠의 컴포넌트 조사 연구로서 red-teaming 도구에 대한 소개가 포함되어 있습니다. 이와 같은 과제를 해결하기 위해, TRCE는 개념 삭제의 신뢰성을 더욱 향상시켜 잠재적 개념 삭제 모델이 특정 멀티모달 또는 적대적인 프롬프트에 충분한 견고성을 보장하려는 목표를 가지고 있습니다. 이를 통해 TRCE는 다양한 앞에서 토대를 형성하게 됩니다. 이 연구에서는, TRCE의 유망한 효과가 분석을 통해, 대조적인 미세 저항 기법을 사용하여  두뇌 매핑 및 표시 이중 단계 개념 삭제 전략 적용에 대한 추후 모듈을 통해, 추가 시험 결과를 제시합니다.
- ***Performance Highlights***: TRCE의 성능 평가 결과, '성적' 콘셉트를 삭제하는데 있어 공격 성공률(Attack Success Rate; ASR)은 1.29%이며, FID(Fréchet Inception Distance) 점수는 12.08, CLIP-Score는 30.71입니다. 이 실험 결과는 TRCE가 악의적 개념 삭제에서 효율성을 갖추고 모델의 기존 생성 능력을 잘 보존하는 것을 보여줍니다. TRCE는 또한 적대적 프롬프트에 대한 높은 견고성을 입증했으며, 이는 새로운 방어 능력과 구성 요소들이 효과적임을 나타냅니다.

### [A Data-Centric Revisit of Pre-Trained Vision Models for Robot Learning](https://arxiv.org/abs/2503.06960)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.06960.png)

Vote: 2

Authors: Bingchen Zhao, Xiaojuan Qi, Xin Wen, Yilun Chen, Jiangmiao Pang

- ***What's New***: 이 논문은 로봇 학습을 위한 사전 훈련된 비전 모델(Pre-Trained Vision Models; PVMs)의 효용성을 데이터 중심의 시각에서 재평가하고, 다양한 시각 데이터를 통해 학습할 수 있는 새로운 방법과 그 효율성을 보여줍니다. 특히 SlotMIM이라는 새로운 방법을 도입하여 단일 객체 중심 데이터뿐만 아니라 다양한 데이터 소스에서 객체 중심 표현(object-centric representations)을 학습하여 성능을 향상시키는 것을 목표로 합니다.
- ***Technical Details***: SlotMIM은 비 객체 중심 데이터(Non-Object-Centric; NOC)에서 객체 중심 학습을 촉진하기 위해 설계되었습니다. 이는 세 가지 주요 구성요소를 포함합니다: 1) 이미지 패치를 객체 레벨의 특징 추상화로 그룹화하는 것, 2) 교차 뷰 일관성 규제를 통한 의미론적 수준의 프로토타입 형성, 3) 객체 수준의 대조 학습을 통해 학습된 표현의 변별성을 촉진하는 것. 이 방법은 241K 이미지 규모의 데이터셋에서 사전 훈련 후, Franka Kitchen 및 Meta-World 등의 제어 작업과 PASCAL VOC 및 ADE20K의 세분화 작업에서 평가되었습니다.
- ***Performance Highlights***: SlotMIM은 동일한 조건에서 기존의 방법들보다 일관된 성능 향상을 보여주며, 특히 객체 탐지 및 세그멘테이션 작업에서 상당한 데이터 효율성과 확장성을 발휘합니다. 241K 샘플만을 사용한 사전 훈련으로도 1M 이상의 샘플을 사용한 이전 방법들보다 뛰어난 성과를 보였으며, COCO 벤치마크에서는 3배 더 많은 데이터를 사용한 기존 방법들을 능가하는 성과를 달성했습니다.

### [PhiloBERTA: A Transformer-Based Cross-Lingual Analysis of Greek and Latin Lexicons](https://arxiv.org/abs/2503.05265)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.05265.png)

Vote: 1

Authors: Rumi A. Allbert, Makai L. Allbert

- ***What's New***: PhiloBERTA는 고대 그리스어와 라틴어 어휘 간의 의미론적 관계를 분석하는 트랜스포머 기반의 새로운 교차 언어 모델(cross-lingual transformer model)입니다. 철학적 개념을 분석하고, 양각 유사성 메트릭(angular similarity metrics)을 사용하여 고대 그리스어 및 라틴어 어휘 간의 구체적 의미 연관성을 식별합니다.
- ***Technical Details***: PhiloBERTA 모델은 다국어 임베딩 기술을 기반으로 개발되었으며, 히스토리컬 코퍼스의 희소성 문제와 도메인별 용어를 처리하기 위한 발전된 방식을 통합합니다. 다국어 전이와 장르 데이터 다양성을 고려한 역방향 드롭아웃 레이어를 통해 직접적인 그리스어-라틴어 평행 코퍼스 없이도 의미 정렬을 가능하게 합니다. BERT 기반의 다국어 토크나이저를 사용하여 그리스어와 라틴어 텍스트를 처리하고, 시점 투영(Temporal Projection) 레이어가 의미적으로 시프트하는 문제를 다룹니다.
- ***Performance Highlights***: PhiloBERTA 모델은 교차 언어 유사성 점수를 0.92로 끌어올리며, 본 연구에서의 기존 대비 44% 개선된 페이프 성과를 나타냅니다. 특히, 어원적으로 관련된 철학적 개념 쌍에서는 높은 유사성을, 대조 쌍에서는 더 큰 다양성을 보여주어 의미 있는 구문적 보존(semantic preservation)을 입증합니다(σ=0.003 vs σ=0.023).

### [Ideas in Inference-time Scaling can Benefit Generative Pre-training Algorithms](https://arxiv.org/abs/2503.07154)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.07154.png)

Vote: 1

Authors: Jiaming Song, Linqi Zhou

- ***What's New***: 최근 몇 년간 생성적 사전 학습(Generative Pre-training)을 통한 기반 모델 연구가 크게 발전했지만, 이 분야의 알고리즘 혁신은 주로 이산 신호에 대한 자가 회귀 모델(Autoregressive Models)과 연속 신호에 대한 확산 모델(Diffusion Models) 위주로 정체되어 있습니다. 이 논문은 추론시간 스케일링(Inference-time Scaling)의 관점에서 이러한 생성적 사전 학습 알고리즘을 혁신할 수 있는 가능성을 제시하고 있습니다. 특히, 귀납적 모멘트 매칭(Inductive Moment Matching; IMM) 알고리즘을 통해 견고하고 효율적인 인퍼런스 프로세스를 시연하였습니다.
- ***Technical Details***: 추론 시간 스케일링(Reference-time scaling)의 두 가지 축인 시퀀스 길이(sequence length)와 세분화 단계(Refinement steps)를 자체 회귀 모델과 확산 모델에 적용하여, 새로운 생성적 사전 학습 알고리즘을 개발할 필요성을 제기하였습니다. 구체적으로, DDIM(Denoising Diffusion Implicit Models)을 예로 들어 현재 디노이징 네트워크 설계 하에서 목표 시그널에 대한 제한된 용량을 어떻게 해결할 수 있는지 설명하였습니다. DDIM 샘플러의 제한점을 개선하기 위해 vθ가 세 가지 인수(xt, t, s)를 취할 수 있도록 하는 해결책을 제안하였습니다.
- ***Performance Highlights***: Inductive Moment Matching (IMM) 기반의 고정밀 샘플 품질을 달성하며 인퍼런스 효율성을 기존 디퓨전 모델보다 10배 이상 개선했습니다. 임의로 오일러 선형 미분 방적식을 사용하는 다른 모델들에도 유사한 비판이 적용될 수 있습니다.

### [Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts](https://arxiv.org/abs/2503.02819)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.02819.png)

Vote: 1

Authors: Marta Skreta, Arnaud Doucet, Alán Aspuru-Guzik, Kirill Neklyudov, Roberto Bondesan, Alexander Tong, Viktor Ohanesian, Rob Brekelmans, Tara Akhound-Sadegh

- ***What's New***: 이 논문은 화학적 특성을 가진 다양한 모델을 결합하거나 복합 프롬프트의 다양한 측면을 포착하기 위해 Feynman-Kac 보정기(Feynman-Kac Correctors; FKCs)를 도입하여 샘플 분포를 정밀하게 제어할 수 있는 도구를 제안합니다. 또한, FKCs를 사용하여 먼저 높은 온도에서 샘플러를 학습하고 나서 낮은 온도로 점진적으로 이동하는 새로운 차원의 변활용 샘플러를 제공합니다.
- ***Technical Details***: Feynman-Kac 보정기는 가중치가 있는 확률 미분 방정식(weighted Stochastic Differential Equations; SDEs)을 구성하는 유연한 방법론을 제안합니다. Annealed, product, 또는 geometric average 분포를 적분하기 위한 일련의 Sequential Monte Carlo (SMC) 리샘플링 방식이 제안되고, 이를 통해 다양한 온도에서의 샘플링이 가능합니다. 또한, Pretrained diffusion models로부터 분자 생성의 조합적 생성 및 이미지 생성을 위한 무분류기 지침(classifier-free guidance)을 증대시킵니다.
- ***Performance Highlights***: 제안된 방법은 물질 생성에서 여러 가지 성향을 가지고 있는 분자의 조합적 생성을 향상시키고, 무분류기 가이던스를 사용한 이미지 생성에서 수치를 향상시킵니다. FKCs는 고온에서 학습을 효율적으로 가능하게 하며, 논문 실험에서 FKCs가 포함된 SMC 리샘플링을 통해서 질적 샘플의 향상이 관찰되었습니다.

### [Symbolic Mixture-of-Experts: Adaptive Skill-based Routing for Heterogeneous Reasoning](https://arxiv.org/abs/2503.05641)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.05641.png)

Vote: 1

Authors: Elias Stengel-Eskin, Mohit Bansal, Justin Chih-Yao Chen, Tianlong Chen, Sukwon Yun

- ***What's New***: SYMBOLIC-MOE는 기존의 대형 언어 모델(LLM)들을 활용해 다중 모드의 복잡한 문제들을 해결할 수 있는 경량의 'Symbolic Mixture-of-Experts' 프레임워크입니다. 이 프레임워크는 각 인스턴스의 필요성에 따라 적절한 전문가를 선택해 다양한 추론 작업을 수행하게끔 돕습니다.
- ***Technical Details***: SYMBOLIC-MOE는 사전에 학습된 LLM 전문가를 스킬 기반으로 동적으로 선택합니다. 선택된 전문가들은 각각 자신만의 추론 결과를 생성하고, 이는 최종적으로 집계자(aggregator)에 의해 하나의 고품질 응답으로 통합됩니다. 배치 추론 방식을 사용하여 GPU 사용을 최소화하면서, 16개의 모델을 단일 GPU에 통합할 수 있습니다. 이 프레임워크는 코드 생성에 있어 별도의 학습 없이도 사용자 정의가 가능하며, 다양한 벤치마크에서 탁월한 성능을 보여줍니다.
- ***Performance Highlights***: SYMBOLIC-MOE는 다양한 벤치마크(MMLU-Pro, GPQA, AIME, MedMCQA)에서 8.15%의 절대적인 평균 성능 향상을 보이며, GPT4o-mini와 같은 강력한 모델 및 다중 에이전트 접근 방식을 상회합니다. 특히, 작업별로 최적의 집계자를 선택함으로써 다중 라운드 토론이 필요 없는 더 효율적인 성능을 제공합니다. 이는 소수의 GPU를 사용할 때도 유리하며, 강력한 성능과 효율성을 동시에 실현합니다.

### [REF-VLM: Triplet-Based Referring Paradigm for Unified Visual Decoding](https://arxiv.org/abs/2503.07413)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.07413.png)

Vote: 1

Authors: Ynan Ding, Xiaohong Liu, Yiying Dong, Luhao Zhu, Guodong Guo, Yan Tai, Zhiqiang Chen

- ***What's New***: 이 연구는 다양한 시각 디코딩 작업을 통합적으로 학습할 수 있는 REF-VLM이라는 새로운 프레임워크를 소개합니다. 이를 통해 개념, 디코딩 유형, 대상의 세 가지 중요한 차원을 명확하게 구분하는 Triplet-Based Referring Paradigm(TRP)을 도입하여 다양한 비주얼 디코딩 시나리오를 처리할 수 있습니다.
- ***Technical Details***: REF-VLM은 여러 시각적 요구에 대응하기 위해 VT-Instruct라는 대규모 멀티태스크 데이터셋을 구성했으며, 1억 개 이상의 멀티모달 대화 샘플을 포함합니다. 여기에는 텍스트 입력과 출력뿐만 아니라 점(point), 상자(box), 스크리블(scribble), 마스크(mask) 같은 다양한 비주얼 입력이 포함되어 있으며, 출력 형태로는 상자, 키포인트(keypoint), 깊이(depth), 마스크 등이 포함됩니다.
- ***Performance Highlights***: REF-VLM은 다양한 표준 벤치마크를 통해 다른 MLLMs보다 뛰어난 성능을 보입니다. 예를 들어, 캡션 생성 작업에서 Flickr30k 데이터셋에서의 CIDEr 점수는 96.0, NoCaps 데이터셋에서는 122.4를 기록하며, VQA 작업에서도 OK-VQA 테스트 데이터셋에서 62.39%의 정확도를 달성했습니다.

### [Next Token Is Enough: Realistic Image Quality and Aesthetic Scoring with Multimodal Large Language Model](https://arxiv.org/abs/2503.06141)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.06141.png)

Vote: 1

Authors: Rui Wang, Lei Sun, Mingxing Li, Yancheng Bai, Xiangxiang Chu

- ***What's New***: 이 논문은 멀티모달 대형 언어 모델(Multimodal Large Language Models; MLLMs)을 활용하여 사용자 생성 콘텐츠 이미지(UGC images) 품질 및 미학 평가를 위한 새로운 데이터셋 RealQA(Realistic image Quality and Aesthetic)를 소개합니다. 이 데이터셋은 총 14,715개의 UGC 이미지로 구성되어 있으며, 각 이미지에는 10개의 세밀한 속성(fine-grained attributes)이 주어져 있습니다.
- ***Technical Details***: RealQA 데이터셋은 사용자 생성 콘텐츠 이미지에 대한 10개의 세부 속성으로 구성되어 있으며, 이는 저수준(예: 이미지 선명도), 중간 수준(예: 주제 보전성) 및 고수준(예: 구성)에 이르는 다양한 레벨을 포함합니다. 이 연구는 체계적인 조사와 함께, MLLMs이 다음 토큰을 예측하는 방식에서도 효과적인 수치 점수를 예측할 수 있는지를 탐구합니다. 이 과정에서는 10개의 세부 속성을 자가 레이블링하여 공공 데이터셋에 적용시키고 CoT(Chain of Thought) 방식을 활용하여 세밀한 속성과 수치 점수를 동시에 예측하게 합니다.
- ***Performance Highlights***: 제안된 방법은 5개의 공공 IQA(Image Quality Assessment) 및 IAA(Image Aesthetic Assessment) 데이터셋에서 최첨단(SOTA) 방법을 능가했으며, CoT를 활용한 제안된 방법은 Koniq-10k 데이터셋에서 특정 모델(Q-Align)의 성능을 PLCC에서 1.8% 개선했습니다. 또한 비디오 품질 평가(VQA) 데이터셋 KoNViD에서는 36.4%의 SRCC 개선을 보여 강력한 제로샷 일반화(generalization)도 입증했습니다.

### [What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization](https://arxiv.org/abs/2503.06698)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.06698.png)

Vote: 1

Authors: Xavier Thomas, Deepti Ghadiyaram

- ***What's New***: 이 논문은 사전 훈련된 모델의 특징 공간에서 잠재 도메인 구조를 발견하여 Domain Generalization(도메인 일반화)을 개선하는 새로운 방법을 제안합니다. 이는 기존에 훈련 데이터에서 도메인 레이블을 사용하지 않고도 학습할 수 있는 접근이기 때문에, 도메인 레이블이 불확실하거나 없는 상황에서 유용합니다.
- ***Technical Details***: 논문은 특히 사전 훈련된 특징 공간에서 Pseudo-Domain(의사 도메인)을 발견하기 위해 Kernel Mean Embeddings(KME)을 사용합니다. 그런 다음, 발견된 의사 도메인 표현을 기존 분류자에 보완적으로 사용하여 다양한 테스트 도메인에 맞게 일반화할 수 있도록 합니다. 이를 위해 GUIDE(Generalization using Inferred Domains from Latent Embeddings)라는 두 단계의 프레임워크를 개발하였으며, 라디얼 베이시스 함수(RBF) 커널 리지 회귀를 통해 특징 공간을 조정합니다.
- ***Performance Highlights***: 이 프레임워크는 TerraIncognita, VLCS, PACS 등의 데이터셋에서 테스트 정확도를 최대 4.3%까지 개선했습니다. GUIDE는 특히 Scalable Diffusion Models(SDM)의 특징 공간을 활용하여 두드러진 성능 향상을 보였으며, 이는 비유의미 매핑 없이 도메인 일반화에서 확실한 이점을 제공합니다.

### [Inductive Moment Matching](https://arxiv.org/abs/2503.07565)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.07565.png)

Vote: 1

Authors: Stefano Ermon, Jiaming Song, Linqi Zhou

- ***What's New***: Inductive Moment Matching(IMM)은 새로운 종류의 발생형 모델로, 단일 또는 소수의 단계 샘플링을 위한 단일 단계 훈련 절차를 통해 고품질 샘플을 생성할 수 있습니다. 기존의 Diffusion 모델이나 Consistency 모델과 달리 IMM은 사전 훈련 초기화가 필요 없으며 다양한 하이퍼파라미터 및 표준 모델 아키텍처 하에서도 안정성을 보장합니다.
- ***Technical Details*: IMM은 시간 종속 마진 분포의 확률적 보간(stochastic interpolants; SIP)을 기반으로 합니다. 이는 두 개의 확률 밀도 함수 간을 연결하는 연속 시간 스토캐스틱 프로세스 입니다. IMM은 수학적 귀납법을 사용하여 효율적으로 훈련될 수 있으며, 두 개의 간단 하고 적은 단계로 모델을 학습할 수 있습니다. IMM 모델은 시간 s < r < t에서 시료를 생성하여 r과 t시점에서의 1단계 IMM을 실행 함출분포를 형성합니다. 이 과정은 귀납적으로 데이터 분포의 수렴을 보장하며, 안정성 향상을 위해 일정한 stochastic interpolants에 따라 IMM을 모델링하고 모멘트 매칭 (Moment Matching; MMD)을 사용해 샘플 기반 발산 지표로 최적화됩니다. 특히, IMM은 Consistency 모델(Consistency Models; CMs)이 IMM의 단일 파티클, 첫순간 일치 특수 사례임을 입증하여 교육의 불안정성의 원인을 부분적으로 설명합니다.**: 3.2
- **고찰**: 기존의 분산 모델 및 Flow Matching(Flow Matching) 기법은 데이터와 사전 확률 간의 점 이법적 상호작용을 통해 고화질 샘플을 생성하는 데 사용됩니다. 그러나 다수의 초기화 및 최적화 단계를 필요로 하며, 처음부터 학습하는 경우에는 안정적인 훈련이 어려운 트릴레마를 야기합니다. IMM은 스크래치 모델을 안정적으로 학습하여, 중간의 조정 없이도 다단계 생성 및 추론에서 초점을 맞출 수 있으며, 최첨단 성능을 달성할 수 있습니다. 특히 CM의 훈련 불안정성을 부분적으로 설명하는 이유를 IMM의 특별한 단일 파티클, 첫순간 일치가 된다는 점입니다.
- ***Performance Highlights***: Inductive Moment Matching(IMM)은 ImageNet-256x256 데이터셋에서 기존의 분산 모델을 능가하며, 단 8단계의 추론으로 1.99 FID를 달성하였습니다. 뿐만 아니라 CIFAR-10 데이터셋에서는 2단계에서 훈련된 모델로 최첨단 성능인 1.98 FID를 기록하였습니다.

### [Collapse of Dense Retrievers: Short, Early, and Literal Biases Outranking Factual Evidence](https://arxiv.org/abs/2503.05037)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.05037.png)

Vote: 0

Authors: Nanyun Peng, Ali Modarressi, Hinrich Schuetze, Mohsen Fayyaz

- ***What's New***: 이 연구는 정보 검색 모델의 편향을 체계적으로 분석하여, 짧은 문서, 반복적인 내용, 서두에 위치한 정보를 과도하게 선호하는 문제점을 밝혀냈습니다. 특히 여러 편향이 결합될 때 모델의 성능 하락이 극심해지며, 이는 Retrieve-Augmented Generation(RAG) 시스템에 부정적인 영향을 미치고 있다는 것을 보여줍니다.
- ***Technical Details***: Re-DocRED 데이터셋을 활용하여 검색 모델들이 문서의 정확한 의미나 정답의 존재보다 표면적인 패턴에 의존하는 경향을 보이는 것을 확인했습니다. 이를 위해 위치 편향(Position Bias), 문자일치편향(Literal Bias), 반복편향(Repetition Bias), 간결성 편향(Brevity Bias) 등을 포함한 여러 편향을 개별적으로 분석하고, 모델의 취약점을 밝히는 실험을 설계했습니다.
- ***Performance Highlights***: 캠브리지 드래곤(Dragon)과 컨트리버(Contriever) 모델들이 적정 성능을 보였으나, 다중 편향 시나리오에서 실제 답이 포함된 문서를 선택할 확률이 3% 이하로 떨어졌습니다. RAG 시스템에서, 모델이 편향 홀리된 문서를 선택할 경우 성능이 아무런 문서도 제공되지 않았을 때보다 34% 떨어졌습니다.

### [ObjectMover: Generative Object Movement with Video Prior](https://arxiv.org/abs/2503.08037)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.08037.png)

Vote: 0

Authors: Xi Chen, Xiaojuan Qi, Xin Yu, Paul Guerrero, Soo Ye Kim, Qing Liu, Zhe Lin, Tianyu Wang

- ***What's New***: ObjectMover는 복잡하고 도전적인 장면에서 객체 이동을 수행할 수 있는 생성 모델로, 주어진 객체를 새로운 위치로 이동시키는데 필요한 조명 조정, 시야 변화, 그림자와 반사의 일관성 유지 등 다양한 이미지 편집 작업을 효과적으로 수행할 수 있습니다. 이 연구는 객체 이동 작업을 시퀀스-투-시퀀스(Sequence-to-Sequence) 문제로 모델링하고, 비디오 생성 모델을 미세 조정하여 적용하는 최초의 접근법을 제안합니다.
- ***Technical Details***: ObjectMover는 영상 생성 모델(Video Generation Model)을 사전 학습하고 이를 기반으로 한 시퀀스-투-시퀀스 예측 문제로 재구성한 후 미세 조정합니다. 대규모 객체 이동 데이터를 확보하기 어려운 문제를 해결하기 위해 현대 게임 엔진(Game Engine)을 사용하여 고품질의 합성 데이터 데이터를 생성하였습니다. 이 외에도 실세계 비디오 데이터로 다중작업학습(Multi-task Learning) 방법을 제안하여 모델의 일반화를 개선하였습니다.
- ***Performance Highlights***: ObjectMover는 최신의 다른 방법들에 비해 실력적으로 뛰어난 결과를 보여주었습니다. 삭제 작업에서는 PSNR 28.90을 기록하며 클립 점수(Clip-Score) 94.84와 DreamSim 0.143을 통해 최상의 결과를 냈습니다. 삽입 작업 및 이동 작업에 있어서도 다른 최첨단 방법들과 비교하여 더 높은 성능을 발휘했습니다. 사용자 연구에서는 응답자 중 상당한 수가 생성된 이미지와 실제 이미지를 혼동할 정도로 사실적인 이미지를 생성하였습니다.

### [OTTER: A Vision-Language-Action Model with Text-Aware Visual Feature Extraction](https://arxiv.org/abs/2503.03734)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2503.03734.png)

Vote: 0

Authors: Jitendra Malik, Letian Fu, Pieter Abbeel, Huang Huang, Tingfan Wu, Mustafa Mukadam, Ken Goldberg, Fangchen Liu

- ***What's New***: OTTER는 텍스트 인식이 가능한 시각적 기능 추출(Text-Aware Visual Feature Extraction)을 통해 기존 VLA 모델의 사전 훈련된 의미론적 정렬을 활용하는 새로운 비전-언어-액션(Vision-Language-Action; VLA) 아키텍처입니다. 이는 대규모 시뮬레이션 및 현실 실험에서 VLA 모델의 제로샷 일반화 능력을 향상시켜 이전보다 높은 성공률을 기록했습니다.
- ***Technical Details***: OTTER는 사전 훈련된 비전-언어 모델의 시각적 기능을 동결 상태로 유지하면서 정책 네트워크로 전달하기 전에 과제 관련 시각적 기능만 선택적으로 추출합니다. CLIP의 주목 출력 Xattn을 사용하여 언어 지침과 정렬된 시각적 토큰을 추출하여 로봇 정책 네트워크에 입력합니다.
- ***Performance Highlights***: OTTER는 기존 VLA 모델보다 훈련 및 보이지 않는 작업에서 더 높은 성공률을 기록했습니다. OTTER는 프리미티브 동작에서 평균 성공률 68%를 기록하며, OTTER-OXE는 대형 로봇 데이터세트에서의 사전 훈련을 통해 성공률이 72%로 증가했습니다. 이는 보다 큰 비전-언어 인코더와 정책 네트워크 용량 증가로 인해 성능이 확장될 수 있음을 보여줍니다.

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
