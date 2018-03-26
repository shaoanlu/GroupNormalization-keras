# GroupNormalization_keras
keras implementation of group normalization. https://arxiv.org/abs/1803.08494

### [Group Normalization](https://arxiv.org/abs/1803.08494)
Yuxin Wu and Kaiming He

## [WIP Alert]
This repository is still work in progress.

The functionality of Group Normalization has not been fully checked. The implementation could be wrong.

## Usage
```python
from GroupNormalization import GroupNormalization

# GroupNormalization(axis=-1, epsilon=1e-6, group=32, **kwargs)

G = 8

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), kernel_initializer='he_normal', input_shape=input_shape))
model.add(GroupNormalization(group=G))
model.add(Activation('relu'))
...
```

## Experiments with group normalization

- [Experiment notebook](https://github.com/shaoanlu/GroupNormalization-keras/blob/master/group_norm_experiments.ipynb)

### Comparison between BatchNorm, groupNorm and InstanceNorm

#### Setup
- Dataset: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
- Architecture
  - ![arch](https://github.com/shaoanlu/GroupNormalization-keras/raw/master/figures/GN_exp_arch.jpg)
- Batch size: 1
- Optimizer: Adam
- Learning rate: from 1e-3 with callback `ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)`
- Epochs: 13
1. Training loss

![](https://github.com/shaoanlu/GroupNormalization-keras/raw/master/figures/trn_loss0.png)

2. Validation loss

![](https://github.com/shaoanlu/GroupNormalization-keras/raw/master/figures/val_loss0.png)

3. Training accuracy

![](https://github.com/shaoanlu/GroupNormalization-keras/raw/master/figures/trn_acc0.png)

4. Validation accuracy

![](https://github.com/shaoanlu/GroupNormalization-keras/raw/master/figures/val_acc0.png)

### More comparisons

#### a. GroupNorm w/ optimizer [AMSGrad](https://openreview.net/forum?id=ryQu7f-RZ), batch size = 1
  - Training time: 3 hrs on Google Colab
#### b. GroupNorm w/ optimizer [AMSGrad](https://openreview.net/forum?id=ryQu7f-RZ), batch size = 128, epochs = 39
  - Training time: 17 mins on Google Colab

1. Training loss

![](https://github.com/shaoanlu/GroupNormalization-keras/raw/master/figures/trn_loss.png)

2. Validation loss

![](https://github.com/shaoanlu/GroupNormalization-keras/raw/master/figures/val_loss.png)

3. Training accuracy

![](https://github.com/shaoanlu/GroupNormalization-keras/raw/master/figures/trn_acc.png)

4. Validation accuracy

![](https://github.com/shaoanlu/GroupNormalization-keras/raw/master/figures/val_acc.png)

## Acknowledgments
Code borrows from [DingKe](https://github.com/DingKe/nn_playground/blob/master/layernorm/layer_norm_layers.py).
