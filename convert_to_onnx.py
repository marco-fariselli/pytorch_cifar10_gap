import torch
import argparse
from models import *
from utils import get_model
from torchsummary import summary
import os
import torch.backends.cudnn as cudnn

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

print('==> Building model..')
net = get_model(args.net)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Load checkpoint.
ckpt_path = f'./checkpoint/{args.net}.pth'
print(f'==> Resuming from checkpoint.. {ckpt_path}')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(ckpt_path)
net.load_state_dict(checkpoint['net'])

net = net.to("cpu")
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
