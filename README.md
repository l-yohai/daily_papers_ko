# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using GPT-4o.

Thanks to [@AK391](https://github.com/AK391) for great work.

## Daily Papers (2024-10-18)

### [Retrospective Learning from Interactions](https://arxiv.org/abs/2410.13852)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13852.png)

Vote: 1

Authors: ['Yiwei Chen', 'Zizhao Chen', 'Anne Wu', 'Mustafa Omer Gul', 'Yoav Artzi', 'Gloria Geng']

- ***What's New***: RESPECT 방법은 대화의 과거 상호작용에서 얻은 신호를 활용하여 대규모 언어 모델(LLM)이 계속해서 학습할 수 있게 합니다. 이 접근 방식은 외부 주석 없이 다중 턴 상호작용에서 신호를 추출하여 적응적 학습을 가능하게 합니다.
- ***Technical Details***: RESPECT는 사용자의 지시를 기반으로 하여 LLM이 다중 턴 추론 작업을 수행하도록 설계되었습니다. 이 방법은 이전 상호작용의 피드백 신호를 복기(retrospection)하고 이에 기반하여 학습을 진행합니다. 구체적으로, 각 상호작용 후 LLM은 자신의 행동을 재검토하여 피드백을 디코딩합니다. MULTIREF라는 새로운 다중 턴 상호작용 시나리오에서 이 방법이 적용되며, 사용자가 3D 도형들로 구성된 난해한 문제를 해결하도록 LLM에게 지시합니다.
- ***Performance Highlights***: 실험 결과, RESPECT 방법을 적용한 후 LLM의 작업 완료 비율이 31%에서 82%로 증가했습니다. 이 과정에서 피드백을 디코드하는 정확성은 시간에 따라 안정적으로 유지되었으며, LLM은 다중 턴 상호작용에서의 성능을 개선하는 데 성공했습니다.

### [AERO: Softmax-Only LLMs for Efficient Private Inference](https://arxiv.org/abs/2410.13060)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2410.13060.png)

Vote: 0

Authors: ['Brandon Reagen', 'Nandan Kumar Jha']

- ***What's New***: AERO는 비선형성(Nonlinearity)을 제거하고 더 효율적인 개인 정보 추론(Private Inference; PI)을 위한 Softmax-only LLM 아키텍처를 제안합니다. 이 연구는 기존 대형 언어 모델(Large Language Models; LLMs)의 아키텍처를 최적화하고, FLOPs(count)을 줄이는 데 중점을 두었습니다.
- ***Technical Details***: AERO는 네 단계로 구성된 아키텍처 최적화 프레임워크로, Transformer 기반 LLM의 기존 아키텍처를 비선형성을 제거하여 재구성합니다. 구체적으로, LayerNorm, GELU와 같은 비선형 컴포넌트를 제거하고 FLOPs 수를 줄이는 설계를 구현했습니다. Softmax-only 아키텍처는 비선형적인 활성화 함수가 없는 경우에 최적의 성능을 발휘할 수 있도록 설계되었습니다. 이 과정에서 새로운 엔트로피 정규화(Entropy Regularization) 기법을 도입하여 Softmax-only 모델의 성능을 향상시킵니다.
- ***Performance Highlights***: AERO는 통신 비용을 최대 4.23배 줄이고, 지연(latency)을 1.94배 단축시킬 수 있습니다. 표준 벤치마크에 따르면, Softmax-only 모델은 perplexity에서 기존의 SOTA(상태 최선; State-of-the-Art) 모델에 비해 6%에서 8% 향상된 성능을 기록했습니다.

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
