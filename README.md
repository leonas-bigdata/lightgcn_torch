# LightGCN-PyTorch

This repository is our team’s fork of the original **LightGCN-PyTorch** implementation, based on the SIGIR 2020 paper by Xiangnan He et al.

> **SIGIR 2020** — Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang.  
> **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation**  
> arXiv: https://arxiv.org/abs/2002.02126

- Original PyTorch repo: https://github.com/gusye1234/LightGCN-PyTorch  
- TensorFlow implementation: https://github.com/kuandeng/LightGCN  
- Author homepage: http://staff.ustc.edu.cn/~hexn/

---

## 📌 Overview

**LightGCN** simplifies traditional Graph Convolutional Networks for collaborative filtering by **removing unnecessary feature transformations and nonlinearities**, and keeping only the most essential component: **neighborhood aggregation**.

This design makes the model:

- Simpler and more elegant
- More efficient to train
- Highly effective for recommendation tasks

Our fork focuses on:

- Cleaner experiment tracking
- Support for multiple benchmark datasets
- Reproducible runs on Google Colab
- Integration with additional custom datasets

---

## 📂 Supported Datasets

This implementation supports four widely used benchmark datasets and one custom dataset prepared by our team:

| Dataset       | Description                                  |
|---------------|----------------------------------------------|
| **Gowalla**   | Location-based social network check-ins      |
| **Yelp2018**  | Business reviews and user ratings            |
| **Amazon-Book** | Book purchase and rating records          |
| **MovieLens1M** | Movie rating dataset                      |
| **GitStar**   | Self-crawled dataset based on GitHub stars  |

---

## 🧪 Experiments

We conducted careful experiments using the PyTorch version of LightGCN.  
All experimental runs, logs, and metric tracking are stored in the `exp/` directory as Jupyter notebooks.

To ensure transparency and academic integrity, we provide public Google Drive links containing:

- The source code from this repository that we uploaded on Google Drive to mount for experiment running
- Execution logs
- Output results
- Saved checkpoints and metrics

These materials serve as **self-proof** that all experiments were conducted by our team on **Google Colab**, without reusing results from external sources. Accessing each link will lead to a folder with three folder LightGCN, UltraGCN and LayerGCN. The source code for this repository is uploaded inside the LightGCN folder for all three links

### Environment

- Experiments were executed on **Google Colab**
- GPU-enabled runtime
- Repeated runs to ensure reproducibility and stability

### Experiment Notebooks

| Notebook             | Description                        | Link |
|----------------------|------------------------------------|------|
| `lightgcn_exp`       | Initial experimental run           | https://drive.google.com/drive/folders/1Iq_NvTZkTy8MYP-0DPvymDHFO2nDsFK_?usp=drive_link |
| `lightgcn_rerun_1`   | Second run for verification        | https://drive.google.com/drive/folders/1l93GDNQzRcL9pg4eR6vriTOxrzcUY_G5?usp=drive_link |
| `lightgcn_rerun_2`   | Third run for consistency checking | https://drive.google.com/drive/folders/12VtE9kucBlikg8x8cVtrnuwBRTTBC68d?usp=drive_link |

These notebooks contain:

- Training configurations
- Evaluation metrics (Recall, NDCG, etc.)
- Result comparisons across runs

---

## 🎯 Purpose of This Fork

This fork was created for:

- Academic study and experimentation across 5 datasets, each dataset is executed three times to get the average score 
- Understanding LightGCN behavior across datasets
- Ensuring reproducible results
- Extending the original implementation with additional datasets and structured experiment tracking

---

## 🙏 Acknowledgements

We sincerely thank the authors of the original LightGCN paper and repository for making their work publicly available.

Please consider citing the original paper when using this codebase.

