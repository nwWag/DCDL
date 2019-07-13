from model.cifar_model import stored_network, network
from model.rule_layer import *
import data.cifar_dataset as md


def load_network(name):
    model_l = stored_network(name)
    return model_l


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CONFIG +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
size_train_nn = 45000
size_valid_nn = 5000

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# DATA +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("Dataset processing", flush=True)
dataset = md.data()
num_var = md.num_variables()

dataset.get_iterator()
train_nn, label_train_nn = dataset.get_chunk(size_train_nn)
val, label_val = dataset.get_chunk(size_valid_nn)
test, label_test = dataset.get_test()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# TEST +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def evaluate(arch):
    print("Evaluating", flush=True)
    restored = load_network(arch)
    size_test_nn = test.shape[0]

    counter = 0  # biased
    acc_sum = 0
    for i in range(0, size_test_nn, 512):
        start = i
        end = min(start + 512, size_test_nn)
        acc = restored.evaluate(test[start:end], label_test[start:end])
        acc_sum += acc
        counter += 1

    print("Test Accuracy", arch.name, acc_sum / counter)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PIPELINE +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("Training", flush=True)
archs = []

archs.append(network("baseline-bn_before-pool_before-ind",avg_pool=False, real_in=False,
                    lr=1E-4, batch_size=2**8, activation=binarize_STE,
                     pool_by_stride=False, pool_before=True, pool_after=False,
                     skip=True, pool_skip=True,
                     bn_before=True, bn_after=False, ind_scaling=True
                     ))
archs[-1].training(train_nn, label_train_nn, val, label_val)
evaluate(archs[-1])
