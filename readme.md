This project is inspired by [cs230 project](cs230.stanford.edu/projects_spring_2019/reports/18681651.pdf). 
We analyze 2 chest datasets from `kaggle`: 
- [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) - 
contains around 5K examples (pneumonia set); and
- [NIH Chest X-rays](https://www.kaggle.com/nih-chest-xrays/data) - contains around 100K examples 
(NIH dataset);

That's a typical image classification problem with 2 or 3 classes for the pneumonia set and 15 classes
for the NIH dataset. We are going (preliminary):
- to use transfer learning (probably `resnet50`) on the pneumonia set with some augmentation;
- retrain `resnet50` (some convolution layers) on the NIH dataset and use it for transfer learning on the pneumonia set (challenge);

We're going to use `tf.keras` from `tensorflow 2.0`. We train in Google Cloud:
- Machine type: `n1-standard-8 (8 vCPUs, 30 GB memory)`;
- GPUs: `1 x NVIDIA Tesla T4`;