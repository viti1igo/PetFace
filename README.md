# CAT FACE RECOGNITION WITH FEW-SHOT LEARNING
## Overview
This is a base Siamese Network from the paper by Koch et al. (2015) to do the Face Recognition for CATS using the PetFace dataset.
- **Goal:** To achieve robust identification of individual cats from visual inputs, utilising a few annotated images per animal.
- **Data:** Uses the PetFaceDataset, comprising expertly aligned and labeled images of cats, with high metadata richness (ID, breed, color, coat).
- **Model:** The Siamese network extracts and compares dense feature embeddings using a triplet-loss approach.
- **Output:** Produces a low-dimensional embedding for each input and uses distance metrics (e.g., Euclidean, cosine) for similarity-based identification.

## Model Architecture

- Siamese Network built on EfficientNet-B0 with shared weights.
- Embedding head includes:
  - Two fully connected layers (128, then 224 units)
  - Batch normalization and L2-normalisation
- Loss function: Triplet margin loss to maximize distance between different individuals and minimize it for the same individual.
- Input: Cropped face images of size 224x224 pixels.
- Identification: Embedding of query image is compared against stored references, with a configurable threshold (default ≈ 0.75) determining matches.

## Result
  - Accuracy: 0.9509 
  - Precision: 0.9673
  - Recall: 0.9334
  - F1-Score: 0.9500
  - ROC-AUC: 0.9807
  - Optimal Threshold: 0.7514

## Distance Analysis
  - Same Cat Mean: 0.2750 ± 0.2836
  - Different Cat Mean: 1.3934 ± 0.2297
  -  Separation: 1.1184

## References

[1] Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese neural networks for one-shot image recognition. In *ICML deep learning workshop* (Vol. 2, No. 1). [[PDF]](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

[2] Shinoda, R., & Shiohara, K. (2024). PetFace: A Large-Scale Dataset and Benchmark for Animal Identification. In *European Conference on Computer Vision (ECCV) 2024*. [[arXiv]](https://arxiv.org/abs/2407.13555)
