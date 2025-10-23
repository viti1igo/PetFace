# CAT FACE RECOGNITION 
## Overview
This is a base Siamese Network from the paper by Koch et al. (2015) to do the Face Recognition for CATS using the PetFace dataset.

## Result
  • Accuracy: 0.9509 

  • Precision: 0.9673

  • Recall: 0.9334

  • F1-Score: 0.9500

  • ROC-AUC: 0.9807

  • Optimal Threshold: 0.7514
  
  ![Model Results](Result_2/roc_curve.png)
  ![Model Results](Result_2/distance_distribution.png)

## Distance Analysis
  • Same Cat Mean: 0.2750 ± 0.2836
  • Different Cat Mean: 1.3934 ± 0.2297
  • Separation: 1.1184

## References

[1] Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese neural networks for one-shot image recognition. In *ICML deep learning workshop* (Vol. 2, No. 1). [[PDF]](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

[2] Shinoda, R., & Shiohara, K. (2024). PetFace: A Large-Scale Dataset and Benchmark for Animal Identification. In *European Conference on Computer Vision (ECCV) 2024*. [[arXiv]](https://arxiv.org/abs/2407.13555)
