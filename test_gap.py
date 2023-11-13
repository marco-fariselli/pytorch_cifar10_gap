# %%
import numpy as np
from nntool.api import NNGraph
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import sys

# %%
transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
calibset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
calibset = torch.utils.data.Subset(calibset, range(0, 1, 1))
calloader = torch.utils.data.DataLoader(calibset, batch_size=1, shuffle=False, num_workers=8)

# %%
onnx_path = sys.argv[1]
# onnx_path = "checkpoint/mobilenet.onnx"

# %%
G = NNGraph.load_graph(onnx_path, load_quantization=False)
#G.draw(filepath="draw", view=True)
max_activ_size, total_params = G.total_memory_usage
ops = G.total_ops

print(f"{G.name}:")
print(f"\tMax Active Size:\t{max_activ_size} elements")
print(f"\tTotal # Parameters:\t{total_params} elements")
print(f"\tTotal # Operations:\t{ops / 1e6:.2f} MOps")
G.adjust_order()
G.fusions('scaled_match_group')
G.fusions('scaled_match_group')
# G.draw()
def representative_dataset(hwc=False):
    for input_tensor, _ in tqdm(calloader):
        input_tensor = input_tensor.numpy()
        if hwc:
            input_tensor = input_tensor.transpose(2, 0, 1)
        yield input_tensor

print("Calibrating...")
stats = G.collect_statistics(representative_dataset(hwc=False))
G.quantize(
    statistics=stats,
    graph_options={
        'use_ne16': True,
        'hwc': True
    },
)
# G.draw()

# %%
# Autotiler options: make the autotiler allocate the input of the network and reuse that space after the first layer
# more L2 for the rest of the network
G[0].at_options.allocate = 1
cal_input = np.random.uniform(0, 1, (32, 32, 3))
qout = G.execute([cal_input], quantize=True, dequantize=False)
res = G.execute_on_target(
    platform="gvsoc",
    directory="test_run",
    input_tensors=[cal_input],
    check_on_target=True,
    print_output=False,
    do_clean=False,
    settings={
        'l1_size': 128000,
        'l2_size': 1200000, 
        'tensor_directory': './tensors',
        'graph_const_exec_from_flash': True,
    },
    at_loglevel=1,
)
if res.returncode:
    print(res.stdout)

print(res.at_log)
print(res.pretty_performance())
