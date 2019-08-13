# setup -------------
import os, sys, shutil, glob, re, time, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torchvision import transforms
from PIL import Image

plt.ion()
torch.set_printoptions(sci_mode = False)


# command line options -------------
parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter, description = "when use others' models, the default approach is to use and freeze pre-trained convolutional weights", epilog = "Typical usage (-w is omitted): \n  -n 200 --epochs 32 \t# MX 150\n  --data-augmentation --epochs 32\n  -m vgg --no-pretrain --epochs 64\n  -m vgg --no-freeze-convolute\n  -m vgg\n  -m vgg --data-augmentation")
parser.add_argument("-w", dest = "path", default= "/media/computer/work/data/dogs-vs-cats/", help = "working directory, contains input images")
parser.add_argument("-m", dest = "model", choices = ["scratch", "vgg"], default= "scratch", help = "which model to use, scratch (default) is a toy CNN") # resnet
parser.add_argument("-d", dest = "device", default = "cuda", help = "cpu, cuda (default), cuda: 0, etc")
parser.add_argument("-b", dest = "train_batch", type = int, metavar = 'N', default = 64, help = "training batch size (default: 64)")
parser.add_argument("-B", dest = "test_batch", type = int, metavar = 'N', default = 128, help = "testing  batch size (default: 128)")
parser.add_argument("--epochs", type = int, metavar = 'N', default = 16, help = "number of epochs to train (default: 16)")
parser.add_argument("--no-pretrain", action = "store_true", help = "when use others' models, not use pre-trained weights and not normalize data with ImageNet mean and std. implying --no-freeze-convolute")
parser.add_argument("--no-freeze-convolute", action = "store_true", help = "when use others' models, not freeze weights in convolutional layers")
parser.add_argument("--data-augmentation", action = "store_true", help = "perfrom data augmentation")
parser.add_argument("--preconvolute", action = "store_true", help = "use pre-convoluted features as input, disabled by --no-freeze-convolute and --data-augmentation")
parser.add_argument("--save", action = "store_true", help = "save model weights")
parser.add_argument("--logs", type = int, metavar = 'N', default = 10, help = "number of times to log training status per epoch (default: 10)")
parser.add_argument("-n", "--train-size", type = int, metavar = 'N', default = 20000, help = "number of images used for training (default: 20000). For now we use all test dataset, 25000*0.2)")
parser.add_argument("-l", dest = "learning_rate", type = float, metavar = 'R', default = 0.0001, help = "SGD learning rate (default: 0.0001)")
parser.add_argument("--momentum", type = float, metavar = 'R', default = 0.5, help = "SGD momentum (default: 0.9)")
parser.add_argument("--step-size", type = int, metavar = 'N', default = 20, help = "schedular step size (default: 20)")
parser.add_argument("--reinitialize-dataset", action = "store_true", help = "use a new permutation to split train and test dataset (80%% vs 20%% of 25000 images). Need write permission to working directory.")
parser.add_argument("-s", dest = "seed", default = int(time.time()), help = "random seed (default: time)")
args = parser.parse_args()

try:
    device = torch.device(args.device)
    torch.randn(1, device = device)
except RuntimeError as err:
    print('Invalid device: "{}". Use CPU instead.'.format(args.device))
    device = torch.device("cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.path = os.path.expanduser(args.path)
args.normalize = not args.no_pretrain and not args.model == "scratch"
if args.no_pretrain: args.no_freeze_convolute = True
if args.data_augmentation or args.no_freeze_convolute: args.preconvolute = False
n = args.train_size



# split train and test set ------
if args.reinitialize_dataset:
    files = glob.glob(os.path.join(args.path, "train", "*.jpg"))
    assert len(files) != 0, "Aborted! Can't find any images under `" + args.path + "`"

    def mk_dir(path):
        """make directory if not exist"""
        if not os.path.exists(path): os.mkdir(path)    

    shutil.rmtree(os.path.join(args.path, "data"), ignore_errors = True)
    os.mkdir(os.path.join(args.path, "data"))
    for data_set in ["train", "test"]:
        os.mkdir(os.path.join(args.path, "data", data_set))
        for animal in ["dog", "cat"]:
            os.mkdir(os.path.join(args.path, "data", data_set, animal))
    
    shuffle = np.random.permutation(len(files))
    test_size = int(len(files) * 0.2)    # arg test ratio
    for phase, file_indexs in {"test": shuffle[:test_size], "train": shuffle[test_size:]}.items():
        for i in file_indexs:
            animal = files[i].split('/')[-1].split('.')[0]
            image = files[i].split('/')[-1]
            src = os.path.join("..", "..", "..", "train", image)
            dest = os.path.join(args.path, "data", phase, animal, image)
            if not os.path.exists(dest): os.symlink(src, dest)
    print("Dataset splitting finished, now you can train and test your model, have fun!")
    sys.exit()



# helper ------------------
imagenet_mean, imagenet_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def imshow_tensor(img_tensor):
    img_np = img_tensor.cpu().numpy().transpose((1, 2, 0))
    if args.normalize: img_np = img_np*imagenet_std + imagenet_mean
    plt.imshow(img_np.clip(0, 1))

def plot_model(model, data):
    classes = ("cat", "dog")
    img_tensor, label = data
    input = img_tensor.view(1, 3, 224, 224).to(next(model.parameters()).device)
    target = torch.argmax(model(input), dim = 1).item()
    imshow_tensor(img_tensor)
    plt.title("A {} is predicted as a {}".format(classes[label], classes[target]))

transform = transforms.Resize((224, 224))
if args.data_augmentation: transform = transforms.Compose([transform, transforms.RandomHorizontalFlip(), transforms.RandomRotation(0.2)])
transform = transforms.Compose([transform, transforms.ToTensor()])
if args.normalize: transform = transforms.Compose([transform, transforms.Normalize(imagenet_mean, imagenet_std)])



# network class and train function ---------------

class Net(torch.nn.Module):
    def __init__(self, n_class, n_pixel):
        super().__init__()
        self.input_n_pixel = n_pixel
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size = 5)
        n_pixel = (n_pixel - (self.conv1.kernel_size[0] -1))//2   
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size = 6)
        self.conv2_drop = torch.nn.Dropout2d()
        n_pixel = (n_pixel - (self.conv2.kernel_size[0] -1))//2       
        self.fc1 = torch.nn.Linear(n_pixel*n_pixel*20, 500)
        self.fc2 = torch.nn.Linear(500, 50)
        self.fc3 = torch.nn.Linear(50, n_class)

    def forward(self, x):
        F = torch.nn.functional
        x = x.view(-1, 3, self.input_n_pixel, self.input_n_pixel) # support a single image
        x = F.relu(F.max_pool2d(                self.conv1(x),  2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0], -1)
        x = F.dropout(F.relu(self.fc1(x)), training = self.training)
        x = F.dropout(F.relu(self.fc2(x)), training = self.training)
        x = self.fc3(x)
        return x

def train(model, train_loader, scheduler, epoch, logs):
    loss, correct, device = 0, 0, next(model.parameters()).device
    model.train()
    scheduler.step()
    log_batches = np.ceil(np.linspace(0, len(train_loader)-1, logs))
    for batch_i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        batch_loss = torch.nn.CrossEntropyLoss()(outputs, targets)
        batch_correct = torch.sum(torch.argmax(outputs, dim = 1) == targets).item()
        scheduler.optimizer.zero_grad()
        batch_loss.backward()
        scheduler.optimizer.step()
        if batch_i in log_batches: 
            print("Epoch {} [{}-{}/{}], average loss: {:.3f}, correct: {}/{} ({:.2f}%)".format(epoch+1, len(inputs)*batch_i+1, len(inputs)*(batch_i+1), len(train_loader.dataset), batch_loss.item(), batch_correct, len(targets), batch_correct/len(targets)*100), flush = True)
        loss += batch_loss.item()
        correct += batch_correct
    print("Summary of Epoch {}, average loss: {:.3f}, correct: {}/{} ({:.2f}%)".format(epoch+1, loss/len(train_loader), correct, len(train_loader.dataset), correct/len(train_loader.dataset)*100), flush = True)


def test(model, test_loader):
    loss, correct, device = 0, 0, next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss += torch.nn.CrossEntropyLoss()(outputs, targets).item()
            correct += torch.sum(torch.argmax(outputs, dim = 1) == targets).item()
    print("Testing, average loss: {:.3f}, correct: {}/{} ({:.2f}%)\n".format(loss/len(test_loader), correct, len(test_loader.dataset), correct/len(test_loader.dataset)*100), flush = True)
    return loss, correct



# main workflow ------------
train_data = torchvision.datasets.ImageFolder(os.path.join(args.path, "data", "train"), transform)
train_data.imgs, train_data.targets, train_data.samples = train_data.imgs[:n], train_data.targets[:n], train_data.samples[:n]
test_data  = torchvision.datasets.ImageFolder(os.path.join(args.path, "data", "test"), transform)
train_loader = torch.utils.data.DataLoader(train_data, args.train_batch, shuffle = True)
test_loader  = torch.utils.data.DataLoader(test_data, args.test_batch)


## vgg 
if args.model == "vgg":
    vgg = torchvision.models.vgg16(not args.no_pretrain)
    if not args.no_freeze_convolute:
        for param in vgg.features.parameters(): param.requires_grad = False
    for layer in vgg.classifier.children(): # a trick to imporve accuracy a little bit
        if type(layer) == torch.nn.Dropout: layer.p = 0.2 
    vgg.classifier[-1] = torch.nn.Linear(vgg.classifier[-1].in_features, 2, not vgg.classifier[-1].bias is None)
    vgg = vgg.to(device)
    if not args.preconvolute:
        if not args.no_freeze_convolute:
            optimizer = torch.optim.SGD(vgg.classifier.parameters(), args.learning_rate, args.momentum)
        else:
            optimizer = torch.optim.SGD(vgg.parameters(), args.learning_rate, args.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size)
        for epoch in range(args.epochs):
            train(vgg, train_loader, scheduler, epoch, args.logs)
            test(vgg, test_loader)
            if args.save: torch.save(vgg.cpu().state_dict(), "dogs-vs-cats.pt")  # we save per eppch since it's very slow
    else:
        sys.exit("preconvolute not supported yet")
        train_preconv_feature = torch.cat([vgg.features(batch[0].to(device)) for batch in train_loader])
        test_preconv_feature  = torch.cat([vgg.features(batch[0].to(device)) for batch in test_loader])
        torch.save(train_preconv_feature.cpu(), "train_preconv_feature.pt")
        torch.save(test_preconv_feature.cpu(),  "test_preconv_feature.pt")
        train_preconv_feature = torch.load("train_preconv_feature.pt")
        test_preconv_feature  = torch.load("test_preconv_feature.pt")
        train_preconv_data = torch.utils.data.TensorDataset(train_preconv_feature.view(train_preconv_feature.shape[0], -1), torch.tensor(train_data.targets))
        test_preconv_data  = torch.utils.data.TensorDataset(test_preconv_feature.view( test_preconv_feature.shape[0], -1),  torch.tensor(test_data.targets))
        train_preconv_loader = torch.utils.data.DataLoader(train_preconv_data, args.train_batch, shuffle = True)
        test_preconv_loader  = torch.utils.data.DataLoader(test_preconv_data,  args.test_batch)
        optimizer = torch.optim.SGD(vgg.classifier.parameters(), args.learning_rate, args.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size)
        for epoch in range(args.epochs):
            train(vgg.classifier, train_preconv_loader, scheduler, epoch, args.logs)
            test(vgg.classifier, test_preconv_loader)
        if args.save: torch.save(vgg.cpu().state_dict(), "dogs-vs-cats.pt")
    sys.exit()

## from scratch
if args.model == "scratch":
    model = Net(2, 224).to(device) # torchvision.models.resnet18(pretrained = False, num_classes = 2)
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size)
    for epoch in range(args.epochs):
        train(model, train_loader, scheduler, epoch, args.logs)
        test(model, test_loader)
    if args.save: torch.save(model.cpu().state_dict(), "dogs-vs-cats.pt")
    sys.exit()

# plot_model(model, test_data[2434])
# vgg two part weight size
