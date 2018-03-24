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

# GroupNormalization(axis=-1, 
#                    gamma_init='one', beta_init='zero', 
#                    gamma_regularizer=None, beta_regularizer=None, 
#                    epsilon=1e-6, 
#                    group=32, **kwargs)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), kernel_initializer='he_normal', input_shape=input_shape))
model.add(GroupNormalization(group=8))
model.add(Activation('relu'))
```

## Acknowledgments
Code borrows from [DingKe](https://github.com/DingKe/nn_playground/blob/master/layernorm/layer_norm_layers.py).
