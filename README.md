# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using ChatGPT.

Thanks to [@AK391](https://github.com/AK391) for great work.


## Daily Papers (2024-01-08)

### [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](https://arxiv.org/abs/2401.02954)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/1odiLLHpdrgLiiUqP578d.png)

Vote: 25

Authors: DeepSeek-AI, Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, Huazuo Gao, Kaige Gao, Wenjun Gao, Ruiqi Ge, Kang Guan, Daya Guo, Daya Guo, Jianzhong Guo, Guangbo Hao, Zhewen Hao, Ying He, Wenjie Hu, +

- 오픈소스 대규모 언어 모델(LLM)의 발전이 빠르게 진행되고 있으나, 기존 문헌에서 설명하는 확장 법칙에 대한 결론은 다양하여 LLM의 확장에 대한 어두운 그림자를 드리웠다.
- 이 연구에서는 확장 법칙을 깊이 있게 연구하고, 7B와 67B라는 두 가지 일반적으로 사용되는 오픈소스 구성에서 대규모 모델의 확장을 용이하게 하는 독특한 발견을 제시한다.
- 장기적 관점을 가진 오픈소스 언어 모델을 발전시키기 위해 DeepSeek LLM 프로젝트가 소개되며, 이를 지원하기 위해 현재 2조 개의 토큰으로 구성되고 지속적으로 확장되고 있는 데이터셋을 개발했다.
- DeepSeek LLM Base 모델에는 감독된 미세 조정(Supervised Fine-Tuning, SFT)과 직접 선호 최적화(Direct Preference Optimization, DPO)를 수행하여, DeepSeek Chat 모델을 만들어냈다.
- 평가 결과는 DeepSeek LLM 67B가 코드, 수학, 추론 분야의 다양한 벤치마크에서 LLaMA-2 70B를 능가하는 성능을 보여주고, 개방형 평가에서는 DeepSeek LLM 67B Chat이 GPT-3.5보다 우수한 성능을 보임을 드러낸다.

### [Denoising Vision Transformers](https://arxiv.org/abs/2401.02957)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/olROnVzzUN7533BJnXI1d.png)

Vote: 15

Authors: Jiawei Yang, Jiawei Yang, Katie Z Luo, Jiefeng Li, Jiefeng Li, Kilian Q Weinberger, Kilian Q Weinberger, Yonglong Tian, Yonglong Tian, Yue Wang

- 비전 트랜스포머(ViTs)의 특징 맵에서 그리드 형태의 아티팩트가 발견되며, 이는 하류 작업에서 ViTs의 성능에 해가 됨을 분석하였습니다.
- 이 문제는 입력 단계에서 위치 임베딩에 기인한 것으로 추적된 바, 대응하기 위해 모든 ViTs에 적용 가능한 새로운 노이즈 모델을 제안합니다.
- 제안된 노이즈 모델은 비전 트랜스포머 출력을 세 가지 요소로 분해합니다: 아티팩트로부터 자유로운 의미론적 항목, 픽셀 위치에 따른 두 가지 아티팩트 관련 항목입니다.
- 신경 필드를 이용하여 교차 시각 특징의 일관성을 강화함으로써 개별 이미지 기반으로 이러한 분해가 가능해진 것으로, 원시 ViT 출력에서 아티팩트가 없는 특징을 추출할 수 있게 합니다.
- 오프라인 애플리케이션에 대한 깨끗한 특징을 제공하기 위해, 개별 이미지 최적화가 필요 없는 온라인 기능을 지원하게 하기 위해 학습 가능한 디노이저를 소개합니다. 이는 미처리 ViT 출력으로부터 직접 아티팩트가 없는 특징을 예측하는 데 뛰어난 일반화 능력을 보였습니다.
- 이러한 2단계 접근 방식을 'Denoising Vision Transformers(DVT)'라 명명하고, 기존에 사전 훈련된 ViTs를 다시 훈련할 필요 없이 모든 트랜스포머 기반 아키텍처에 즉시 적용할 수 있음을 강조합니다.
- 다양한 ViTs(DINO, MAE, DeiT-III, EVA02, CLIP, DINOv2, DINOv2-reg)에 대해 평가를 실시하였으며, 여러 데이터셋에서 의미론적 및 기하학적 작업 모두에 있어 일관되고 상당한 개선을 보임을 입증하였습니다(예: mIoU +3.84).
- 이 연구는 ViT 설계의 재평가를 촉구하는 것으로, 특히 위치 임베딩의 단순한 사용에 관해서 이루어집니다.

### [DocGraphLM: Documental Graph Language Model for Information Extraction](https://arxiv.org/abs/2401.02823)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/JIFMRpzyIuX_rOzUT_yN_.png)

Vote: 14

Authors: Dongsheng Wang, Dongsheng Wang, Zhiqiang Ma, Zhiqiang Ma, Armineh Nourbakhsh, Armineh Nourbakhsh, Kang Gu, Sameena Shah

- 본 논문에서는 Visually Rich Document Understanding (VrDU)의 발전을 바탕으로 복잡한 레이아웃을 가진 문서에서 정보 추출 및 질의응답을 가능하게 하는 새로운 프레임워크인 DocGraphLM을 소개한다.
- 이 모델은 사전 훈련된 언어 모델과 그래프 의미론을 결합하여 1) 문서를 표현하기 위한 공동 인코더 아키텍처를 제안하고, 2) 문서 그래프를 재구성하기 위한 새로운 링크 예측 방법을 제안한다.
- DocGraphLM은 노드 사이의 방향성과 거리를 예측하며, 이웃 복원을 우선시하고 먼 노드 탐지는 가중치를 낮출 수 있는 수렴하는 공동 손실 함수를 사용한다.
- 세 가지 최신 데이터셋에서 정보 추출(IE)과 질의응답(QA) 작업에 대한 실험을 통해 그래프 특징을 채택함으로써 일관된 성능 향상을 보여준다.
- 또한, 링크 예측을 통해서만 구성되었음에도 학습 과정에서 학습 수렴 속도를 가속화한다는 점을 보고한다.

### [Progressive Knowledge Distillation Of Stable Diffusion XL Using Layer Level Loss](https://arxiv.org/abs/2401.02677)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/rm1eHI_h17TwoUqA6LrEg.png)

Vote: 12

Authors: Yatharth Gupta, Yatharth Gupta, Vishnu V. Jaddipal, Vishnu V. Jaddipal, Harish Prabhala, Harish Prabhala, Sayak Paul, Sayak Paul, Patrick Von Platen, Patrick Von Platen

- Stable Diffusion XL (SDXL) 모델은 다양성과 최상의 이미지 품질로 텍스트-이미지 변환(T2I) 분야에서 최고의 오픈 소스 모델로 자리 잡았습니다.
- 본 논문에서는 SDXL 모델의 계산 요구 사항을 효율적으로 해결하기 위해, 크기를 줄이면서도 생성 품질을 유지하는 방법으로 1.3B 파라미터 UNet인 Segmind Stable Diffusion (SSD-1B)와 0.74B 파라미터인 Segmind-Vega라는 축소된 두 가지 변형을 소개합니다.
- 이 모델들은 SDXL의 U-Net 구조에서 잔여 네트워크와 트랜스포머 블록을 단계적으로 제거함으로써 파라미터 수와 시간 지연을 현저하게 줄였으며, https://hf.co/Segmind 에서 모델 가중치를 공개하였습니다.
- 우리의 소형 모델들은 지식 전달을 통해 원본 SDXL을 효과적으로 모방하며, 다양한 더 큰 규모의 SDXL과 비교할 때 경쟁력 있는 결과를 달성합니다.
- 이 연구는 SDXL의 고품질 생성 능력을 보존하는 동시에 모델 크기를 줄이기 위해 레이어 레벨 손실과 지식 증류를 결합한 효과성을 강조하며, 자원이 제한된 환경에서의 접근성을 높일 수 있는 배포를 용이하게 합니다.

### [Pheme: Efficient and Conversational Speech Generation](https://arxiv.org/abs/2401.02839)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/u11v048sLbmsWLQq4hoWT.png)

Vote: 8

Authors: Paweł Budzianowski, Paweł Budzianowski, Taras Sereda, Taras Sereda, Tomasz Cichy, Tomasz Cichy, Ivan Vulić, Ivan Vulić

- 최근 몇 년 간 음성 생성 기술이 눈에 띄게 발전하여, 실제 인간 목소리와 구별할 수 없는 원샷 생성 능력을 달성했습니다.
- 음성 생성 기술의 혁신을 큰 언어 모델과 통합하면 다양한 응용 프로그램이 혁명적으로 변화할 수 있습니다.
- 그러나 보조 대화 시스템과 같은 특정 응용 프로그램은 실시간으로 효율적으로 작동하는 자연스럽고 대화형의 음성 생성 도구를 요구합니다.
- 현대 최고의 기술인 VALL-E와 SoundStorm은 큰 신경 코덱스의 계층과 방대한 훈련 데이터를 필요로 합니다.
- 이에 반해, MQTTS는 소규모의 현실 대화 음성 데이터를 활용하는 더 작은 대화형 TTS 모델을 개발하려 하지만, 자동회귀적 성질 때문에 실시간 사용에 한계가 있습니다.
- 본 논문에서 우리는 Pheme 모델 시리즈를 소개하여 기존 TTS 모델의 장점을 활용하면서 한계를 극복하고자 하는데, 1) 작지만 고성능 모델을 제공하고, 2) 병렬 음성 생성을 가능하게 하며, 3) 자연스러운 대화형 음성을 생성하고, 4) 소규모 대화 데이터에서도 효율적인 훈련이 가능하여, 10배 이상의 데이터 요구량을 줄이면서도 자동회귀 TTS 모델의 품질에 맞출 수 있습니다.
- 미리 훈련된 Pheme 체크포인트를 바탕으로 더 큰 교사 모델들에 의해 생성된 합성 음성만을 사용하여 간단한 교사-학생 증류를 통해 단일 화자 설정에 대한 음성 품질을 크게 향상시킬 수 있음을 보여줍니다.
- 오디오 샘플과 미리 훈련된 모델은 온라인으로 제공됩니다.

### [Open-Vocabulary SAM: Segment and Recognize Twenty-thousand Classes Interactively](https://arxiv.org/abs/2401.02955)

[Watch Video](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/_kjrFWrT3hgtg53OgrIXB.mp4)
<div><video controls src="https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/_kjrFWrT3hgtg53OgrIXB.mp4" muted="false"></video></div>

Vote: 6

Authors: Haobo Yuan, Haobo Yuan, Xiangtai Li, Xiangtai Li, Chong Zhou, Chong Zhou, Yining Li, Yining Li, Kai Chen, Chen Change Loy

- 이 논문은 두 가지 주목할 만한 시각 기초 모델인 CLIP과 SAM(Segment Anything Model)을 통합한 새로운 통합 프레임워크에 대한 심도 있는 탐구를 제시한다.
- 여기서 소개하는 Open-Vocabulary SAM은 동시에 상호 작용적인 분할과 인식을 수행하는 SAM 기반 모델로, SAM2CLIP과 CLIP2SAM이라는 두 가지 독특한 지식 전달 모듈을 활용한다.
- SAM2CLIP은 SAM의 지식을 증류 및 학습 가능한 트랜스포머 어댑터를 통해 CLIP에 적응시키고, CLIP2SAM은 SAM의 인식 능력을 향상시키기 위해 CLIP의 지식을 SAM에 전달한다.
- 다양한 데이터셋과 탐지기(detectors)에서 진행한 광범위한 실험은 Open-Vocabulary SAM이 분할 및 인식 작업에서 상당한 성능 향상을 이뤄냄을 보여준다.
- 또한 이미지 분류 데이터 학습을 활용함으로써, 우리의 방법은 약 22,000개의 클래스를 분할하고 인식할 수 있다.

### [Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache](https://arxiv.org/abs/2401.02669)

![](https://cdn-uploads.huggingface.co/production/uploads/60f1abe7544c2adfd699860c/v7TAWEoRIn2nf3mLoq7oi.png)

Vote: 4

Authors: Bin Lin, Tao Peng, Chen Zhang, Minmin Sun, Minmin Sun, Lanbo Li, Hanyu Zhao, Hanyu Zhao, Wencong Xiao, Qi Xu, Xiafei Qiu, Xiafei Qiu, Shen Li, Zhigang Ji, Yong Li, Wei Lin

- 대규모 언어 모델(LLMs)의 급속한 확산은 클라우드 기반 LLM 서비스의 성장을 촉진하고 AI 애플리케이션의 발전에 중요한 역할을 하고 있다.
- LLM 서비스의 동적 자동 회귀 특성과 매우 긴 맥락을 지원할 필요성은 유연한 자원 할당 및 해제에 대한 요구를 만든다.
- 이러한 요구에 대처하기 위해, 본 연구에서는 KV 캐시를 작고 관리하기 쉬운 단위로 분할하여 분산 처리와 저장을 가능하게 하는 새로운 분산 주의(attention) 알고리즘인 DistAttention을 소개한다.
- 이를 바탕으로, 본 논문은 DistKV-LLM이라는 분산 LLM 서빙 시스템을 제안하여, 데이터 센터에 걸친 모든 접근 가능한 GPU 및 CPU 메모리를 활용하여 KV 캐시를 동적으로 관리하고 효과적으로 조정한다.
- 이 시스템은 다양한 맥락 길이에 적응할 수 있도록 클라우드 상에서 고성능 LLM 서비스를 보장한다.
- 32개의 NVIDIA A100 GPU를 사용한 클라우드 환경에서 검증된 시스템은 2개에서 32개 인스턴스 구성에서 현재 최신 LLM 서비스 시스템보다 1.03-2.4배의 종단 간 처리량 향상을 보였고, 최대 1,900K까지의 맥락 길이를 지원하는 것으로 나타났다.



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
