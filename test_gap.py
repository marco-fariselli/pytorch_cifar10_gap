
import numpy as np
from nntool.api import NNGraph
from nntool.api.utils import quantization_options, model_settings
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--onnx_path', default="checkpoint/mobilenet.onnx", type=str, help='onnx path')
parser.add_argument('--mode', default="accuracy", type=str, choices=["accuracy", "accuracy_onnx", "performance"])
parser.add_argument('--quant_mode', default="ne16", type=str, choices=["ne16", "int8", "fp16"])
parser.add_argument('--fast', action="store_true")
parser.add_argument('--print_output', action="store_true")
args = parser.parse_args()

np.random.seed(12345)
random_input = np.random.uniform(-1, 1, (3, 32, 32))
if args.fast:
    def representative_dataset():
        yield random_input
else:
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    calibset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
    calibset = torch.utils.data.Subset(calibset, range(0, 20, 1))
    calloader = torch.utils.data.DataLoader(calibset, batch_size=1, shuffle=False, num_workers=1)
    def representative_dataset(hwc=False):
        for input_tensor, _ in tqdm(calloader):
            input_tensor = input_tensor.numpy()
            if hwc:
                input_tensor = input_tensor.transpose(2, 0, 1)
            yield input_tensor


G = NNGraph.load_graph(args.onnx_path, load_quantization=False)
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

print("Calibrating...")
stats = G.collect_statistics(representative_dataset())
if args.quant_mode == "fp16":
    G.quantize(graph_options=quantization_options(scheme="FLOAT", float_type="bfloat16"))
else:
    G.quantize(
        statistics=stats,
        graph_options={
            'use_ne16': args.quant_mode == "ne16",
            'hwc': True
        },
    )

# G.draw()

if args.mode.startswith("accuracy"):
    # test with just 1000 samples for timing reasons
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testset = torch.utils.data.Subset(testset, range(0, 1000, 1))
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

    quantize = True
    onnx = args.mode == "accuracy_onnx"
    if onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.onnx_path)
        input_name = sess.get_inputs()[0].name

    acc = 0
    count = 0
    for i, (test_image, test_label) in enumerate(testloader):
        test_in = test_image.numpy()
        test_lab = test_label.numpy()[0]
        if onnx:
            output = sess.run(None, {input_name: test_in})
            output = np.asarray(output)
        else:
            output = G.execute([test_in[0].transpose((1, 2, 0))], dequantize=quantize)

        pred = output[-1][0].argmax()
        acc += 1 if pred == test_lab else 0
        count += 1
        progress_bar(i, len(testloader), f'Acc: {acc / count * 100:.2f}% ({acc}/{count})')
    print(f"Final accuracy: {acc / count * 100:.2f}%")

else:
    # Autotiler options: make the autotiler allocate the input of the network and reuse that space after the first layer
    # more L2 for the rest of the network
    G[0].at_options.allocate = 1
    if args.quant_mode == "fp16":
        # No HWC in fp16
        qout = G.execute([random_input], quantize=True, dequantize=False)
    else:
        qout = G.execute([random_input.transpose(1, 2, 0)], quantize=True, dequantize=False)
    res = G.execute_on_target(
        platform="gvsoc",
        directory="test_run",
        input_tensors=qout[0],
        tolerance=0.2 if args.quant_mode == "fp16" else 0,
        check_on_target=True,
        print_output=args.print_output,
        do_clean=False,
        settings=model_settings(
            l1_size=128000,
            l2_size=1200000, 
            tensor_directory='./tensors',
            graph_const_exec_from_flash=True,
        ),
        at_loglevel=1,
    )
    if res.returncode:
        print(res.stdout)

    print(res.at_log)
    if res.performance:
        print(res.pretty_performance())
