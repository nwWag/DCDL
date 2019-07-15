# DCDL
Deep Convolutional Rule Learner

The DCDL is split into three parts. First, training a DCDL-Network. Second, generating training sets. 
Third, running SLS rule extraction.

## DCDL-Network
This repo contains a DCDL-18 network for MNIST/Fashion-MNIST (mnist_pipeline.py) and CIFAR10 (cifar_pipeline.py) respectively. In general a network is build by adding

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

into a pipeline. For the shorter DCDL-10 layers have to be disabled, no particular file is provided. In the following we describe the function call of network() that corresponds to each validated/tested block.


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
         bn_before=True, bn_after=False, ind_scaling=False
         )
```

### Block 3
```python
network("Block 3",avg_pool=False, real_in=False,
        lr=1E-4, batch_size=2**8, activation=binarize_STE,
         pool_by_stride=False, pool_before=False, pool_after=False,
         skip=False, pool_skip=False,
          bn_before=True, bn_after=False, ind_scaling=False
         )
```

### Block 4
```python
network("Block 4",avg_pool=False, real_in=False,
        lr=1E-4, batch_size=2**8, activation=binarize_BetterSTE,
         pool_by_stride=False, pool_before=False, pool_after=False,
         skip=False, pool_skip=False,
         bn_before=True, bn_after=False, ind_scaling=False
         )
```

### Block 5
```python
network("Block 5",avg_pool=False, real_in=False,
        lr=1E-4, batch_size=2**8, activation=binarize_STE,
         pool_by_stride=False, pool_before=True, pool_after=False,
         skip=False, pool_skip=False,
         bn_before=True, bn_after=False, ind_scaling=False
         )
```

### Block 6
```python
network("Block 6",avg_pool=False, real_in=False,
        lr=1E-4, batch_size=2**8, activation=binarize_STE,
         pool_by_stride=False, pool_before=False, pool_after=True,
         skip=False, pool_skip=False,
         bn_before=True, bn_after=False, ind_scaling=False
         )
```

### Block 7
```python
network("Block 7",avg_pool=False, real_in=False,
        lr=1E-4, batch_size=2**8, activation=binarize_STE,
         pool_by_stride=False, pool_before=True, pool_after=False,
         skip=True, pool_skip=True,
         bn_before=True, bn_after=False, ind_scaling=False
         )
```

### Block 8
```python
network("Block 8",avg_pool=False, real_in=False,
        lr=1E-4, batch_size=2**8, activation=binarize_STE,
         pool_by_stride=False, pool_before=True pool_after=False,
         skip=True, pool_skip=True,
         bn_before=True, bn_after=False, ind_scaling=True
         )
```

### ResNet
```python
network("ResNet",avg_pool=True, real_in=True,
        lr=1E-4, batch_size=2**8, activation=tf.nn.relu,
         pool_by_stride=False, pool_before=False, pool_after=True,
         skip=True, pool_skip=True,
         bn_before=True, bn_after=False, ind_scaling=False
         )
```

### ResNet Binary Input
```python
network("ResNet-Binary",avg_pool=True, real_in=False,
        lr=1E-4, batch_size=2**8, activation=tf.nn.relu,
         pool_by_stride=False, pool_before=False, pool_after=True,
         skip=True, pool_skip=True,
         bn_before=True, bn_after=False, ind_scaling=False
         )
```

### DCDL-Network Real-Valued Input
```python
network("DCDL-Real",avg_pool=False, real_in=True,
        lr=1E-4, batch_size=2**8, activation=binarize_STE,
         pool_by_stride=False, pool_before=True, pool_after=False,
         skip=False, pool_skip=False,
         bn_before=True, bn_after=False, ind_scaling=False
         )
```

## Datasets
Datasets per perceptron can be extracted by generate_datasets.py. The aforementioned training stores the final model, to access this model the same network() call is needed. One can determine for which layers a dataset is created, as well as the corresponding resolutions of the layers.

## SLS Rule Extraction
The extraction (extracting_pipeline.py) utilizes the file created in the previous step. The same parameters as used for generating the datasets need to be set. One can set $k$ (the number of terms) in the config section. The SLS algorithm call has the follwing parameters.

```python
   clauses_n,  # k
   maxSteps,  # # of Updates
   p_g1,  # Prob of rand term in candidate
   p_g2,  # Prob of rand literal in candidate
   p_s,  # Prob of rand term in candidate
   data,  # Training set
   label,  # Training label
   data_val,  # Validation set
   label_val,  # Validation label
   data_test,  # Test set
   label_test,  # Test label
   pos_neg,  # Storage for candidates
   on_off,  # Mask for candidates
   pos_neg_to_store,  # Storage for current BEST candidate
   on_off_to_store,  # Mask for current BEST candidate
   vector_n,  # of training instances !!!!NEEDS TO BE BIGGER THEN BATCH_SIZE!!!!
   vector_n_val,  # # of validation instances !!!!NEEDS TO BE BIGGER THEN BATCH_SIZE!!!!
   vector_n_test,  # # of data instances !!!!NEEDS TO BE BIGGER THEN BATCH_SIZE!!!!
   features_n,  # # of Features
   batch,  # If score calculation should be done batchwise
   cold_restart,  # Restart if stuck in bad local minimum
   decay,  # Decay factor, could be zero. Up to min_prob
   min_prob,  # No decay below this threshold
   zero_init  # Alternativ initialization
```

The batch size as well as the trailing stop need to be set as Makros in the
DCDL/parallel_sls/src/sls_mult_core.cpp
file.

