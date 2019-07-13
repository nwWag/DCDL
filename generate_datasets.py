import fileinput
import numpy as np
import tensorflow as tf
import data.mnist_dataset as md
from model.example_model import stored_network, network
from model.rule_layer import *
import sys


def load_network(name):
    model_l = stored_network(name)
    return model_l


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CONFIG +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
size_train_nn = 3000
num_blocks, start_block = 2, 2
pooling_at = [1, 2, 3]
start_res_in = 14
start_num_ex_in = 10 * 40 * 4

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# DATA +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("Dataset processing", flush=True)
dataset = md.data()
num_var = md.num_variables()

dataset.get_iterator()
train_nn, label_train_nn = dataset.get_chunk(size_train_nn)


def generate_set(arch, tensor_name, size):
    print("Generating ", tensor_name, flush=True)
    restored = load_network(arch)
    size_train_nn = size

    counter = 0  # biased
    acc_sum = 0
    for i in range(0, size_train_nn, 512):
        start = i
        end = min(start + 512, size_train_nn)
        restored.print_to_stderr(train_nn[start:end], tensor_name)
        counter += 1


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PIPELINE +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

dcdl_network = network("baseline-bn_before-pool_before", avg_pool=False, real_in=False,
                     lr=1E-4, batch_size=2**8, activation=binarize_STE,
                     pool_by_stride=False, pool_before=True, pool_after=False,
                     skip=False, pool_skip=False,
                     bn_before=True, bn_after=False, ind_scaling=False
                     )

start_res = start_res_in
start_num_ex = start_num_ex_in

for i in range(start_block, num_blocks + 1):
    generate_set(dcdl_network, "dcdl_conv_" + str(i) + "/_out1", start_num_ex)
    generate_set(dcdl_network, "dcdl_conv_" + str(i) + "/_out2", start_num_ex)
    if i in pooling_at:
        start_num_ex *= 4

with fileinput.FileInput("datasets.txt", inplace=True) as file:
    for line in file:
        print(line.replace("[", ""), end='')

with fileinput.FileInput("datasets.txt", inplace=True) as file:
    for line in file:
        print(line.replace("]", ""), end='')

for line in fileinput.FileInput("datasets.txt", inplace=1):
    if line.strip():
        print(line.strip())
