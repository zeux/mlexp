This is a notebook that explores various approaches to solving MNIST. The training takes
a few minutes for each network on 4070 Ti, using augmentation to avoid overfitting.

1. 3 fully connected layers; reaches 0.029 loss in 50 epochs (2 minutes), 98.2% accuracy
Train [0.670M params]: 50 epochs took 114.16 sec, train loss 0.031799, val loss 0.029352

2. LeNet'98; reaches 0.026 loss in 50 epochs (2 minutes), 98.3% accuracy
Train [0.062M params]: 50 epochs took 137.43 sec, train loss 0.026267, val loss 0.026350

3. Pretrained VGG11, with FC layer replaced: reaches 0.033 loss in 10 epochs (10 minutes), 97.7% accuracy
Train [12.433M params]: 10 epochs took 566.17 sec, train loss 0.066222, val loss 0.033030

4. VGG-inspired 3-level reduction + 2 fully connected layers; reaches 0.017 loss in 50 epochs (7.5 minutes), 98.6% accuracy
Train [0.164M params]: 50 epochs took 462.09 sec, train loss 0.008975, val loss 0.017446