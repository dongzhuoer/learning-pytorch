import os, sys, shutil, glob, re, time, argparse
from IPython import get_ipython
import numpy as np
import torch, torchvision
from torchvision import transforms
import utility
torch.set_printoptions(sci_mode = False)


# command line options -------------
parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter, description = "dogs vs cats classfication by deep learning", epilog = "python3 -u image-classify.py --gpus 6 7 8 9\npython3 -u image-classify.py --gpus 6 7 8 9 -m vgg\npython3 -u image-classify.py --gpus 6 7 8 9 -m inception\npython3 -u image-classify.py --gpus 6 7 8 9 -m densenet\npython3 -u image-classify.py --gpus 6 7 8 9 --data-augmentation\npython3 -u image-classify.py --gpus 6 7 8 9 --only-fc\npython3 -u image-classify.py --gpus 6 7 8 9 --epochs 32 --no-pretrain\npython3 -u image-classify.py --gpus 6 7 8 9 --epochs 64 --no-pretrain -m vgg -l 0.0001\npython3 -u image-classify.py --gpus 6 7 8 9 -s 1000 -S 500\n")
parser.add_argument("--data-dir", metavar = "PATH", default= "data/dogs-vs-cats/data", help = "data directory, contains input images")
parser.add_argument("-m", dest = "model", choices = ["resnet", "vgg", "inception", "densenet"], default= "resnet", help = "which model to use (default: %(default)s)") 
parser.add_argument("-b", dest = "train_batch", type = int, metavar = 'N', default = 32, help = "training batch size (default: %(default)s)")
parser.add_argument("-B", dest = "valid_batch", type = int, metavar = 'N', default = 64, help = "validation  batch size (default: 128)")
parser.add_argument("--gpus", type = int, metavar = "GPU", default = [0], nargs = '+', help = "ordinal of GPUs to use, such as \"0 1\" (default: 0)")
parser.add_argument("-s", dest = "train_subset", type = int, metavar = 'N', help = "subset how many images for training, use all images (20000) by default")
parser.add_argument("-S", dest = "valid_subset", type = int, metavar = 'N', help = "subset how many images for validation, use all images (5000) by default")
parser.add_argument("-l", dest = "learning_rate", type = float, metavar = 'R', default = 0.001, help = "SGD learning rate (default: %(default)s)")
parser.add_argument("--no-pretrain", action = "store_true", help = "when use well-known models, not use pre-trained weights")
parser.add_argument("--only-fc", action = "store_true", help = "only change parameters in last fully connected layer, ignored when --no-pretrain")
parser.add_argument("--data-augmentation", action = "store_true", help = "perfrom data augmentation")
parser.add_argument("--epochs", type = int, metavar = 'N', default = 16, help = "number of epochs to train (default: %(default)s)")
parser.add_argument("--seed", default = int(time.time()), help = "random seed (default: time)")
args = parser.parse_args()
#args = parser.parse_args("--epochs 4 -s 100 -S 400 -b 1 -B 4".split())
#args = parser.parse_args("--gpus 6 7 8 9".split())
#args = parser.parse_args("-b 2 -B 4".split())


args.train_batch *= len(args.gpus)
args.valid_batch *= len(args.gpus)
args.data_dir = os.path.expanduser(args.data_dir)
torch.cuda.set_device(args.gpus[0])
if args.no_pretrain: args.only_fc = False
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# main workflow ---------
normalization = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet mean and std
augmentation  = torchvision.transforms.Compose([ torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomRotation(0.2) ])

transform = torchvision.transforms.Compose([            torchvision.transforms.Resize((224, 224)) ])    
if args.data_augmentation: transform = torchvision.transforms.Compose([ transform, augmentation  ])
transform = torchvision.transforms.Compose([ transform, torchvision.transforms.ToTensor()         ])    
if not args.no_pretrain:   transform = torchvision.transforms.Compose([ transform, normalization ])    


train_data = torchvision.datasets.ImageFolder(f'{args.data_dir}/train', transform)
valid_data = torchvision.datasets.ImageFolder(f'{args.data_dir}/test',  transform)
train_loader = torch.utils.data.DataLoader(train_data, args.train_batch, num_workers = len(args.gpus), shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_data, args.valid_batch, num_workers = len(args.gpus))
if args.train_subset is not None: train_data.imgs, train_data.targets, train_data.samples = train_data.imgs[:args.train_subset], train_data.targets[:args.train_subset], train_data.samples[:args.train_subset]
if args.valid_subset is not None: valid_data.imgs, valid_data.targets, valid_data.samples = valid_data.imgs[:args.valid_subset], valid_data.targets[:args.valid_subset], valid_data.samples[:args.valid_subset]

if args.model == "resnet":
    model = torchvision.models.resnet34(pretrained = not args.no_pretrain)
    for p in model.parameters(): p.requires_grad = not args.only_fc
    model.fc = torch.nn.Linear(in_features = model.fc.in_features, out_features = 2, bias=True)
if args.model == "inception":
    model = torchvision.models.inception_v3(pretrained = not args.no_pretrain)
    model.aux_logits = False
    for p in model.parameters(): p.requires_grad = not args.only_fc
    model.fc = torch.nn.Linear(in_features = model.fc.in_features, out_features = 2, bias=True)
if args.model == "densenet":
    model = torchvision.models.densenet121(pretrained = not args.no_pretrain)
    for p in model.parameters(): p.requires_grad = not args.only_fc
    model.classifier = torch.nn.Linear(in_features = model.classifier.in_features, out_features = 2, bias=True)
if args.model == "vgg":
    model = torchvision.models.vgg19(pretrained = not args.no_pretrain) # vgg 16
    for p in model.parameters(): p.requires_grad = not args.only_fc
    model.classifier[6] = torch.nn.Linear(in_features = model.classifier[6].in_features, out_features = 2, bias=True)
model = torch.nn.DataParallel(model.cuda(), args.gpus)
optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
if get_ipython(): sys.exit()
_, accuracy = utility.valid(model, valid_loader, f"Epoch {0:03d} validation")

for epoch in range(args.epochs): #pass
    _, _        = utility.train(model, train_loader, f"\nEpoch {epoch+1:03d} training  ", optimizer)
    _, accuracy = utility.valid(model, valid_loader, f"Epoch {epoch+1:03d} validation")
    # model.cpu()
    # torch.save(model.module.state_dict(), f'{args.model}-{accuracy:4.2f}.pt')
    # model.cuda()




