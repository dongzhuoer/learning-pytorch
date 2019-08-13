import os, sys, shutil, glob, re, time, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from PIL import Image
import utility
torch.set_printoptions(sci_mode = False)


# command line options -------------
parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter, description = "calculate pre-convoluted features and train an ad-hoc classifier", epilog = "python3 -u preconvolute.py --gpus 6 7 8 9\npython3 -u preconvolute.py --gpus 6 7 8 9 -m vgg\npython3 -u preconvolute.py --gpus 6 7 8 9 -m inception\npython3 -u preconvolute.py --gpus 6 7 8 9 -m densenet\n")
parser.add_argument("--data-dir", metavar = "PATH", default= "data/dogs-vs-cats/data", help = "data directory, contains input images")
parser.add_argument("-m", dest = "model", choices = ["resnet", "vgg", "inception", "densenet"], default= "resnet", help = "which model to use (default: %(default)s)") 
parser.add_argument("-b", dest = "train_batch", type = int, metavar = 'N', default = 32, help = "training batch size (default: %(default)s)")
parser.add_argument("-B", dest = "valid_batch", type = int, metavar = 'N', default = 64, help = "validation  batch size (default: 128)")
parser.add_argument("--epochs", type = int, metavar = 'N', default = 16, help = "number of epochs to train (default: %(default)s)")
parser.add_argument("-l", dest = "learning_rate", type = float, metavar = 'R', default = 0.001, help = "SGD learning rate (default: %(default)s)")
parser.add_argument("--gpus", type = int, metavar = "GPU", default = [0], nargs = '+', help = "ordinal of GPUs to use, such as \"0 1\" (default: 0)")
parser.add_argument("--seed", default = int(time.time()), help = "random seed (default: time)")
args = parser.parse_args()

args.train_batch *= len(args.gpus)
args.valid_batch *= len(args.gpus)
args.data_dir = os.path.expanduser(args.data_dir)
torch.cuda.set_device(args.gpus[0])
torch.manual_seed(args.seed)


# preconvolute resnet -----------------
class layer_activition():
    def __init__(self, model):
        super().__init__()
        self.handle = model.register_forward_hook(self.hook)
        self.values = []
    
    def hook(self, model, input, output):
        feature = output.view(1, output.shape[0], -1).cpu().numpy()
        self.values.extend(feature)

    def remove(self):
        self.handle.remove()

def pre_convolute(model, layer, data_loader):
    model.eval()
    feature, labels, device = layer_activition(layer), [], next(model.parameters()).device
    for input, target in data_loader:
        with torch.no_grad(): model( input.to(device) ).view( len(input), -1 ) # bacth is unchaged
        labels.append(   target.cpu().numpy() )
    return np.concatenate(feature.values), np.concatenate(labels)

imagenet_mean, imagenet_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((299, 299)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(imagenet_mean, imagenet_std)
])
train_data = torchvision.datasets.ImageFolder(f'{args.data_dir}/train', transform)
valid_data = torchvision.datasets.ImageFolder(f'{args.data_dir}/test',  transform)
train_loader = torch.utils.data.DataLoader(train_data, args.train_batch, num_workers = len(args.gpus), shuffle = False)
valid_loader = torch.utils.data.DataLoader(valid_data, args.valid_batch, num_workers = len(args.gpus), shuffle = False)

if args.model == "resnet":
    model = torchvision.models.resnet34(pretrained = True)
    layer = model.layer4
if args.model == "inception":
    model = torchvision.models.inception_v3(pretrained = True)
    model.aux_logits = False
    layer = model.Mixed_7c
if args.model == "densenet":
    model = torchvision.models.densenet121(pretrained = True)
    layer = model.features
if args.model == "vgg":
    model = torchvision.models.vgg16(pretrained = True)
    layer = model.features
model = torch.nn.DataParallel(model.cuda(), args.gpus)
train_feature, train_label = pre_convolute(model, layer, train_loader) 
valid_feature, valid_label = pre_convolute(model, layer, valid_loader)

# Linear model -----------------------
train_data2 = torch.utils.data.TensorDataset(torch.tensor(train_feature), torch.tensor(train_label))
valid_data2 = torch.utils.data.TensorDataset(torch.tensor(valid_feature), torch.tensor(valid_label))
train_loader2 = torch.utils.data.DataLoader(train_data2, args.train_batch, num_workers = len(args.gpus), shuffle = True)
valid_loader2 = torch.utils.data.DataLoader(valid_data2, args.valid_batch, num_workers = len(args.gpus), shuffle = False)
model2 = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(train_data2[0][0].shape[0], 2)
)
model2 = torch.nn.DataParallel(model2.cuda(), args.gpus)
optimizer = torch.optim.Adam(model2.parameters(), lr = args.learning_rate)

_, accuracy = utility.valid(model2, valid_loader2, f"Epoch {0:03d} validation")
for epoch in range(args.epochs): #pass
    _, _        = utility.train(model2, train_loader2, f"\nEpoch {epoch+1:03d} training  ", optimizer)
    _, accuracy = utility.valid(model2, valid_loader2, f"Epoch {epoch+1:03d} validation")


