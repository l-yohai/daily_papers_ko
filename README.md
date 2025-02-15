# Daily Papers

## Project Description

This project aims to automatically translate and summarize [Huggingface's daily papers](https://huggingface.co/papers) into Korean using GPT-4o.

Thanks to [@AK391](https://github.com/AK391) for great work.

## Daily Papers (2025-02-15)

### [Latent Radiance Fields with 3D-aware 2D Representations](https://arxiv.org/abs/2502.09613)

![](https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2502.09613.png)

Vote: 4

Authors: Siyu Huang, Xi Liu, Chaoyi Zhou, Feng Luo

- ***What's New***: 이 논문은 2D 잠재 공간(latent space)에서 직접 3D 복원 성능을 얻을 수 있는 잠재 광선장(Latent Radiance Field; LRF)을 제안하였습니다. 이는 다양한 환경, 특히 실내 및 무경계의 야외 장면에서 사실적인 3D 재구축을 수행할 수 있는 최초의 작업입니다.
- ***Technical Details***: 제안된 프레임워크는 VAE의 2D 잠재 표현에 3D 인식을 통합하는 세 가지 주요 단계로 구성됩니다: (1) 3D 일관성을 높이는 correspondence-aware 자동 인코딩 방법, (2) 3D-aware 2D 표현을 3D 공간으로 들어 올리는 LRF 생성, (3) VAE-Radiance Field (VAE-RF) 정렬을 통해 2D 표현의 이미지 디코딩 성능 향상.
- ***Performance Highlights***: 본 방법은 다수의 NVS, 3D 생성, 그리고 few-shot NVS 실험에서 기존 방법들을 능가하는 합성 품질과 cross-dataset 일반화 가능성을 보여주었습니다. LRF는 PSNR, SSIM, LPIPS 측면에서 DL3DV-10K와 같은 다양한 데이터셋에서 뛰어난 성과를 기록하였습니다.

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
