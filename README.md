# Monocular Open Vocabulary Occupancy Prediction for Indoor Scenes (LegoOcc)

[![arXiv](https://img.shields.io/badge/arXiv-2602.22667-b31b1b.svg)](https://arxiv.org/abs/2602.22667)

> **LegoOcc** tackles **monocular open-vocabulary 3D semantic occupancy** in **large-scale indoor scenes** under **geometry-only supervision** (binary occupancy labels only).
> We represent scenes as **Language-Embedded Gaussians**, propose an **opacity-aware Poisson Gaussian-to-Occupancy** operator for stable volumetric aggregation,
> and introduce **Progressive Temperature Decay** to reduce feature mixing during splatting and strengthen Gaussian–language alignment.

<p align="center">
  <img src="assets/framework.svg" width="95%">
</p>

---

## News
- Accepted to CVPR2026, code will be released before the conference

---

## Citation
If you find this work useful, please consider citing:

```bibtex
@misc{zhou2026monocularopenvocabularyoccupancy,
      title={Monocular Open Vocabulary Occupancy Prediction for Indoor Scenes}, 
      author={Changqing Zhou and Yueru Luo and Han Zhang and Zeyu Jiang and Changhao Chen},
      year={2026},
      eprint={2602.22667},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.22667},
}
```