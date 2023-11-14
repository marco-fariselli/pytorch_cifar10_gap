# Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Training
```
# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01
```

## Accuracy after 20 epochs
| Model                                                 | Acc.    | Acc. Small | Ops [M]   | Acc. Fp16 | Cycles fp16 [M]  | Latency fp16 (@370MHz) | MAC/Cyc fp16 | Acc. int8 | Cycles int8 [M]  | Latency int8 (@370MHz) | MAC/Cyc int8 | Acc. ne16 | Cycles ne16 [M]  | Latency ne16 (@370MHz) | MAC/Cyc ne16 |
| ----------------------------------------------------- | ------- | ---------- | --------- | --------- | ---------------- | ---------------------- | ------------ | --------- | ---------------- | ---------------------- | ------------ | --------- | ---------------- | ---------------------- | ------------ |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 79.34%  |    80.90%  |  152.9    |           |                  |                        |              |    80.30% | 9.74             |                        | 15.69        |           |                  |                        |              |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 82.66%  |    83.30%  |           |           |                  |                        |              |           |                  |                        |              |           |                  |                        |              |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 75.01%  |    75.20%  |           |           |                  |                        |              |           |                  |                        |              |           |                  |                        |              |
| [DenseNet](https://arxiv.org/abs/1608.06993)          | 83.75%  |    84.00%  |           |           |                  |                        |              |           |                  |                        |              |           |                  |                        |              |
| MobileNet                                             | 82.79%  |    82.20%  |           |           |                  |                        |              |           |                  |                        |              |           |                  |                        |              |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 78.00%  |    77.20%  |           |           |                  |                        |              |           |                  |                        |              |           |                  |                        |              |
| SeNet18                                               | 84.91%  |    85.00%  |           |           |                  |                        |              |           |                  |                        |              |           |                  |                        |              |
| ShuffleNetV2                                          | 79.16%  |    79.00%  |           |           |                  |                        |              |           |                  |                        |              |           |                  |                        |              |
| EfficientNet                                          | 77.33%  |    78.60%  |           |           |                  |                        |              |           |                  |                        |              |           |                  |                        |              |
| RegNet                                                | 69.09%  |    68.80%  |           |           |                  |                        |              |           |                  |                        |              |           |                  |                        |              |

<!-- 
| ShuffleNet                                            | 77.64%  |    78.20%  |           |           |                  |                        |              |           |                  |                        |              |           |                  |                        |              |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 77.22%  |    78.20%  |           |           |                  |                        |              |           |                  |                        |              |           |                  |                        |              |
| GoogleNet                                             | 82.80%  |    84.50%  |           |           |                  |                        |              |           |                  |                        |              |           |                  |                        |              |
| DLA                                                   | 78.04%  |    78.00%  |           |           |                  |                        |              |           |                  |                        |              |           |                  |                        |              | -->
