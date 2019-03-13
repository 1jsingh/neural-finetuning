## Finetuning Convolutional Neural Networks using Neural Activation Data

This work looks at ways of improving the existing Convolutional Neural Networks using neural activation data obtained from the Visual Cortex of [Macaques](https://en.wikipedia.org/wiki/Macaque) while being presented with correspoding image stimuli.

## Similarity to IT Dissimilarity Matrix (SIT)

|  **Model** | **Top-1 error** | **Top-5 error** | **SIT_mean** | **SIT_std** |
| --- | --- | --- | --- | --- |
|  **AlexNet** | 43.45 | 20.91 | 0.5312 | 0.0369 |
|  **VGG-11** | 30.98 | 11.37 | 0.5604 | 0.0313 |
|  **VGG-13** | 30.07 | 10.75 | 0.54387 | 0.0326 |
|  **VGG-16** | 28.41 | 9.62 | 0.5357 | 0.031 |
|  **VGG-19** | 27.62 | 9.12 | 0.5242 | 0.0307 |
|  **VGG-11 with batch normalization** | 29.62 | 10.19 | 0.523 | 0.0282 |
|  **VGG-13 with batch normalization** | 28.45 | 9.63 | 0.5262 | 0.0309 |
|  **VGG-16 with batch normalization** | 26.63 | 8.5 | 0.4955 | 0.0307 |
|  **VGG-19 with batch normalization** | 25.76 | 8.15 | 0.4986 | 0.029 |
|  **ResNet-18** | 30.24 | 10.92 | 0.51599 | 0.028 |
|  **ResNet-34** | 26.7 | 8.58 | 0.4894 | 0.0297 |
|  **ResNet-50** | 23.85 | 7.13 | 0.4464 | 0.0302 |
|  **ResNet-101** | 22.63 | 6.44 | 0.457 | 0.0311 |
|  **ResNet-152** | 21.69 | 5.94 | 0.4794 | 0.0321 |
|  **SqueezeNet 1.0** | 41.9 | 19.58 | 0.503 | 0.0375 |
|  **SqueezeNet 1.1** | 41.81 | 19.38 | 0.545 | 0.03686 |
|  **Densenet-121** | 25.35 | 7.83 | 0.4777 | 0.0325 |
|  **Densenet-169** | 24 | 7 | 0.4838 | 0.0286 |
|  **Densenet-201** | 22.8 | 6.43 | 0.4572 | 0.0309 |
|  **Densenet-161** | 22.35 | 6.2 | 0.4841 | 0.0299 |
|  **Inception v3** | 22.55 | 6.44 | 0.521 | 0.0281 |



## Linear SVM Analysis

|  **Features** | **lsvm_acc_mean** | **lsvm_acc_std** |
| :--- | :---: | :---: |
|  **IT-multi** | 0.6543 | 0.0166 |
|  **V4-multi**(128)** | 0.31872 | 0.017 |
|  **AlexNet** | 0.5451 | 0.0221 |
|  **VGG-11** | 0.7192 | 0.0234 |
|  **VGG-13** | 0.7356 | 0.0218 |
|  **VGG-16** | 0.7443 | 0.0233 |
|  **VGG-19** | 0.7484 | 0.019 |
|  **VGG-11 with batch normalization** | 0.746 | 0.0202 |
|  **VGG-13 with batch normalization** | 0.7588 | 0.019 |
|  **VGG-16 with batch normalization** | 0.7805 | 0.0192 |
|  **VGG-19 with batch normalization** | 0.7739 | 0.0241 |
|  **ResNet-18** | 0.7204 | 0.023 |
|  **ResNet-34** | 0.7185 | 0.0234 |
|  **ResNet-50** | 0.7401 | 0.0207 |
|  **ResNet-101** | 0.7305 | 0.0197 |
|  **ResNet-152** | 0.7794 | 0.0199 |
|  **SqueezeNet 1.0** | 0.6276 | 0.0249 |
|  **SqueezeNet 1.1** | 0.6397 | 0.025 |
|  **Densenet-121** | 0.648 | 0.028 |
|  **Densenet-169** | 0.6365 | 0.0313 |
|  **Densenet-201** | 0.6404 | 0.036 |
|  **Densenet-161** | 0.6498 | 0.0321 |
|  **Inception v3** |  |  |