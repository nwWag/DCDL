# DCDL
Deep Convolutional Rule Learner

The DCDL is split into three parts. First, training a DCDL-Network. Second, generating training sets. 
Third, running SLS rule extraction.

## DCDL-Network
This repo contains a DCDL-18 network for MNIST/Fashion-MNIST and CIFAR10 respectively. In general a network is build by adding

```python
archs.append(network("baseline-bn_before-pool_before",avg_pool=False, real_in=False,
                    lr=1E-4, batch_size=2**8, activation=Clipped_STE,
                     pool_by_stride=False, pool_before=True, pool_after=False,
                     skip=True, pool_skip=True,
                     bn_before=True, bn_after=False, ind_scaling=False
                     ))
archs[-1].training(train_nn, label_train_nn, val, label_val)
evaluate(archs[-1])
```

into a pipeline. For the shorter DCDL-10 layers have to be disabled, no particular file is provided. In the following we describe the function call of network() that corresponds to each validated block.


### Block 1
```python
network("Block 1",avg_pool=False, real_in=False,
        lr=1E-4, batch_size=2**8, activation=binarize_ClippedSTE,
         pool_by_stride=False, pool_before=False, pool_after=False,
         skip=False, pool_skip=False,
         bn_before=False, bn_after=False, ind_scaling=False
         )
```

### Block 2
```python
network("Block 2",avg_pool=False, real_in=False,
        lr=1E-4, batch_size=2**8, activation=binarize_ClippedSTE,
         pool_by_stride=False, pool_before=False, pool_after=False,
         skip=False, pool_skip=False,
         bn_before=False, bn_after=True, ind_scaling=False
         )
```

### Block 3
```python
network("Block 3",avg_pool=False, real_in=False,
        lr=1E-4, batch_size=2**8, activation=binarize_STE,
         pool_by_stride=False, pool_before=False, pool_after=False,
         skip=False, pool_skip=False,
         bn_before=False, bn_after=True, ind_scaling=False
         )
```

### Block 4
```python
network("Block 4",avg_pool=False, real_in=False,
        lr=1E-4, batch_size=2**8, activation=binarize_BetterSTE,
         pool_by_stride=False, pool_before=False, pool_after=False,
         skip=False, pool_skip=False,
         bn_before=False, bn_after=True, ind_scaling=False
         )
```

### Block 5
```python
network("Block 5",avg_pool=False, real_in=False,
        lr=1E-4, batch_size=2**8, activation=binarize_STE,
         pool_by_stride=False, pool_before=True, pool_after=False,
         skip=False, pool_skip=False,
         bn_before=False, bn_after=True, ind_scaling=False
         )
```

### Block 6
```python
network("Block 6",avg_pool=False, real_in=False,
        lr=1E-4, batch_size=2**8, activation=binarize_STE,
         pool_by_stride=False, pool_before=False, pool_after=True,
         skip=False, pool_skip=False,
         bn_before=False, bn_after=True, ind_scaling=False
         )
```

### Block 7
```python
network("Block 7",avg_pool=False, real_in=False,
        lr=1E-4, batch_size=2**8, activation=binarize_STE,
         pool_by_stride=False, pool_before=False, pool_after=True,
         skip=True, pool_skip=True,
         bn_before=False, bn_after=True, ind_scaling=False
         )
```

### Block 8
```python
network("Block 8",avg_pool=False, real_in=False,
        lr=1E-4, batch_size=2**8, activation=binarize_STE,
         pool_by_stride=False, pool_before=False, pool_after=True,
         skip=True, pool_skip=True,
         bn_before=False, bn_after=True, ind_scaling=True
         )
```
