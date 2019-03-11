## Finetuning Convolutional Neural Networks using Neural Activation Data

This work looks at ways of improving the existing Convolutional Neural Networks using neural activation data obtained from the Visual Cortex of [Macaques](https://en.wikipedia.org/wiki/Macaque) while being presented with correspoding image stimuli.

## Similarity to IT Dissimilarity Matrix (SIT)

|  **Model** | **Top-1 error** | **Top-5 error** | **SIT_mean** | **SIT_std** |
| --- | --- | --- | --- | --- |
|  **AlexNet** | 43.45 | 20.91 | 0.5183 | 0.0379 |
|  **VGG-11** | 30.98 | 11.37 | 0.5425 | 0.0317 |
|  **VGG-13** | 30.07 | 10.75 | 0.5257 | 0.0307 |
|  **VGG-16** | 28.41 | 9.62 | 0.5335 | 0.0309 |
|  **VGG-19** | 27.62 | 9.12 | 0.511 | 0.0298 |
|  **VGG-11 with batch normalization** | 29.62 | 10.19 | 0.5156 | 0.0293 |
|  **VGG-13 with batch normalization** | 28.45 | 9.63 | 0.5208 | 0.0299 |
|  **VGG-16 with batch normalization** | 26.63 | 8.5 | 0.4972 | 0.0302 |
|  **VGG-19 with batch normalization** | 25.76 | 8.15 | 0.4964 | 0.0294 |
|  **ResNet-18** | 30.24 | 10.92 | 0.5204 | 0.026 |
|  **ResNet-34** | 26.7 | 8.58 | 0.4894 | 0.0297 |
|  **ResNet-50** | 23.85 | 7.13 | 0.4475 | 0.027 |
|  **ResNet-101** | 22.63 | 6.44 | 0.457 | 0.0311 |
|  **ResNet-152** | 21.69 | 5.94 | 0.4794 | 0.0321 |
|  **SqueezeNet 1.0** | 41.9 | 19.58 | 0.4968 | 0.0365 |
|  **SqueezeNet 1.1** | 41.81 | 19.38 | 0.545 | 0.0374 |
|  **Densenet-121** | 25.35 | 7.83 | 0.469 | 0.0325 |
|  **Densenet-169** | 24 | 7 | 0.4778 | 0.0269 |
|  **Densenet-201** | 22.8 | 6.43 | 0.4487 | 0.0324 |
|  **Densenet-161** | 22.35 | 6.2 | 0.4688 | 0.033 |
|  **Inception v3** | 22.55 | 6.44 | 0.521 | 0.0281 |