
import numpy as np
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
from utils import progress_bar
import onnxruntime as ort

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--onnx_path', default="checkpoint/mobilenet.onnx", type=str, help='onnx path')
args = parser.parse_args()

print(f"Testing model {args.onnx_path}")
transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# test with just 1000 samples for timing reasons
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testset = torch.utils.data.Subset(testset, range(0, 1000, 1))
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

sess = ort.InferenceSession(args.onnx_path)
input_name = sess.get_inputs()[0].name

acc = 0
count = 0
for i, (test_image, test_label) in enumerate(testloader):
    test_in = test_image.numpy()
    test_lab = test_label.numpy()[0]
    output = sess.run(None, {input_name: test_in})
    output = np.asarray(output)

    pred = output[-1][0].argmax()
    acc += 1 if pred == test_lab else 0
    count += 1
    progress_bar(i, len(testloader), f'Acc: {acc / count * 100:.2f}% ({acc}/{count})')
print(f"Final accuracy: {acc / count * 100:.2f}%")
