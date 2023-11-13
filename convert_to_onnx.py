import torch
import argparse
from models import *
from utils import get_model
from torchsummary import summary
import os

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--net', default='vit')
parser.add_argument('--resume', action='store_true')

args = parser.parse_args()

onnx_path = f"checkpoint/{args.net}.onnx"

# Model factory..
print(f'==> Building model.. {args.net}')
net = get_model(args.net)
if args.net != "dla":
    summary(net, (3, 32, 32), device="cpu")

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/{args.net}.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

print(onnx_path)
torch_input = torch.randn(1, 3, 32, 32).to("cpu")
torch.onnx.export(
    net,
    torch_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    export_params=True
)
