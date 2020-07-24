import os, sys, io, time, argparse, collections
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import torch, torch.utils.data, torchvision
import IPython
# [Understanding Variational Autoencoders (VAEs)](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73), one post is enough to understand VAE
# [Variational autoencoders](https://www.jeremyjordan.me/variational-autoencoders/), very comprehensive using example of image generating
# [Variational Autoencoders Explained](https://anotherdatum.com/vae.html) emphasis that network learns a probability density function
# 可视化，还原 paper https://github.com/hwalsuklee/tensorflow-mnist-VAE 
# to do
# if batch: run many, use pandoc to produce html


class vae_encoder(torch.nn.Module):
    def forward(self, x):
        mean, lvar = torch.chunk( self.module(x), 2, dim = 1 )
        z = mean + (0.5*lvar).exp() * torch.randn_like(mean) # reparameterization trick
        return z, mean, lvar

class vae_encoder_mlp(vae_encoder):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.module = torch.nn.Sequential(
            torch.nn.Sequential( torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU() ),
            torch.nn.Linear(hidden_dim, latent_dim*2),
        )

class vae_decoder(torch.nn.Module): 
    def forward(self, x): return self.module(x)

class vae_decoder_mlp(vae_decoder):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()    
        self.module = torch.nn.Sequential(
            torch.nn.Sequential( torch.nn.Linear(latent_dim, hidden_dim), torch.nn.ReLU() ),
            torch.nn.Sequential( torch.nn.Linear(hidden_dim, output_dim), torch.nn.Sigmoid() ),   
        )

class vae_mlp(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """assume input in the range of [0, 1]"""
        super().__init__()
        self.encoder = vae_encoder_mlp(input_dim, hidden_dim, latent_dim)
        self.decoder = vae_decoder_mlp(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        z, mean, lvar = self.encoder(x)
        return self.decoder(z), mean, lvar



# set up workspace
def command_line(arg_list = []):
    parser = argparse.ArgumentParser(description = "simplest VAE: MNIST, 2D latent", epilog = "--name test if don't want to overwrite saved model\nvisualize training stat never needs batch detial (print or discard)\n", formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", type = str, metavar = "str", default = "simple-vae", help = "name of experiment, default is %(default)s")
    parser.add_argument("-s", "--seed", type = int, metavar = 'N', default = int(time.time()), help = "random seed, default is current time")
    parser.add_argument("-b", "--batch-size", type = int, metavar = 'N', default = 512, help = "input batch size for training, default is %(default)s")
    parser.add_argument("--dist", choices = ["Gaussian", "Bernoulli"], default = "Gaussian", help = "distribution type of p(x|z), default is %(default)s")
    parser.add_argument("-c", type = float, metavar = 'R', default = 1, help = "constant parameter for p(x|z) (variance if Gaussian), default is %(default)s")
    parser.add_argument("--beta", type = float, metavar = 'R', default = 1, help = "weight of KL divergence loss, default is %(default)s")
    parser.add_argument("--scale", type = float, metavar = 'R', default = 1, help = "multiple loss before backward, default is %(default)s")
    parser.add_argument("--lr", type = float, metavar = 'R', default = 0.001, help = "learning rate, default is %(default)s")
    parser.add_argument("--only-KL", action = "store_true", help = "only backward KL divergence loss (implies --beta 1)")
    parser.add_argument("--resume", action = "store_true", help = "resume from previous experiment (--name must match)")
    parser.add_argument("--load-model", type = str, metavar = 'PATH', help = "load saved model's state_dict before training (ignores --resume)")
    parser.add_argument("-e", "--epochs", type = int, metavar = 'N', default = 32, help = "number of epochs to train (save model after each epoch), default is %(default)s")
    parser.add_argument("--logs", type = int, metavar = 'N', default = 10, help = "number of times to log training status per epoch, default is %(default)s")
    parser.add_argument("--gui", action = "store_true", help = "plot 1*3 figure each epoch,  (to do: save image)")
    parser.add_argument("--jupyter-theme", action = "store_true", help = "customize plot theme for jupyter")
    return parser.parse_args(arg_list)
    
def initialize(arg_list):
    args = command_line(arg_list)
    args.model_file = f"models/{args.name}.pt"
    if args.resume and args.load_model is None: args.load_model = args.model_file
    torch.manual_seed(args.seed)
    if args.gui: plt.ion()
    return args



# train and test model ------------------------------ 
def batch(model, datas, beta = 1, dist = "Gaussian", c = None, optimizer = None, scale = 1): 
    """`c` is constant for decoder distribution"""
    # datas, beta, dist, c, scale = next(iter(train_loader))[0], 1, "Gaussian", 1, 1
    inputs = datas.flatten(1).to( next(model.parameters()).device )
    outputs, mean, lvar = model(inputs)
    m = len(datas[0].flatten()) # logP(x) = sum(logP(xi)) = nxi * mean(logP(xi))
    if dist == "Gaussian": nll = m*torch.nn.MSELoss()(outputs, inputs)/2/c + m*np.log(2*np.pi*c)/2 # MSE/2var + log(sqrt(2*pi*var))
    if dist == "Bernoulli": nll = m*torch.nn.BCELoss()(outputs, inputs) # negative log likehood
    kld = -0.5*torch.sum( 1 + lvar - mean.pow(2) - lvar.exp() )/len(inputs)  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2) in https://arxiv.org/abs/1312.6114 Appendix.B
    loss = kld if np.isinf(beta) else (nll + beta * kld) # *scale
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item(), nll.item(), kld.item()

def train(model, data_loader, args, optimizer):
    # data_loader = train_loader
    test, beta = optimizer is None, np.inf if args.only_KL else args.beta 
    (t_loss, t_nll, t_kld), N, bs = [0]*3, len(data_loader.dataset), data_loader.batch_size
    model.eval() if test else model.train()
    for batch_i, (datas, _) in enumerate(data_loader): #pass
        to_log = np.floor(np.linspace(0, len(data_loader), args.logs, endpoint = False))
        echo = not test and batch_i in to_log
        loss, nll, kld = batch(model, datas, beta, args.dist, args.c, optimizer, args.scale)   
        if echo: print(f"[{batch_i*bs:5d}/{N}]\t{loss:.2f}\t{nll:.2f}\t{kld:.4f}")            
        t_loss, t_nll, t_kld = [t_loss, t_nll, t_kld] + np.array([loss, nll, kld])*len(datas) # last batch may be imcomplete
    print("Test" if test else "Train", f"   \t{t_loss/N:.2f}\t{t_nll/N:.2f}\t{t_kld/N:.4f}", end = '')
    if not test or not args.gui: print()  # remove empty line between test line and plot



# visualize performance -------------
def image_grid(datas, nrow):
    # datas, nrow = original, 100
    """datas is N*784 tensor, return for plt.show()"""
    images = datas.view(-1, 1, 28, 28)
    big_image = torchvision.utils.make_grid(images, nrow)    
    return big_image.permute(1, 2, 0).numpy()

def plot_latent_space(model, original, label, psize, ax):
    """scatterplot of 2D latent space with digit labelled by color\npoint `size`"""
    # psize, ax = 15, None
    palette = sns.color_palette( plt.cm.get_cmap("jet")(np.linspace(0, 1, 10)) )
    with torch.no_grad(): latent = model.encoder( original.cuda() )[0].cpu()
    z1, z2, digit = latent[:, 0].numpy(), latent[:, 1].numpy(), label.numpy().astype(str)
    df = pd.DataFrame({"z1": z1, "z2": z2, "digit": digit}) # not use df -> no legend title
    ax2 = sns.scatterplot(z1, z2, digit, palette = palette, ax = ax, s = psize)
    for text, x, y in df.groupby("digit").mean().reset_index().values: #pass
        ax2.text(x, y, text, fontsize = psize, fontweight = "bold")

def plot_manifold(model, breaks, end, ax):
    # breaks, end, ax = 21, 3, plt
    ticks = np.linspace(-end, end, breaks)
    grid_coord = np.array(np.meshgrid(ticks, ticks)).T.reshape(-1, 2)
    grid_latent = torch.tensor(grid_coord).float()
    with torch.no_grad(): manifold = model.decoder(grid_latent.cuda()).cpu()
    manifold2 = manifold.view(breaks, breaks, -1).transpose(0, 1).flip(0).reshape(breaks**2, -1)
    ax.imshow( image_grid(manifold2, breaks) )  # manifold2 for cartesian system
    ax.imshow( image_grid(manifold2, breaks) )

def plot_compare(model, original, nrow, ncol, ax): # maybe use model
    # nrow, ncol, ax = 20, 10, plt
    """alternately plot nrow input image and nrow output image for ncol times"""
    with torch.no_grad(): reconstruct = model( original.cuda() )[0].cpu()
    row_l = [None]*(ncol*2)
    for i in range(ncol): #pass
        row_l[2*i]   = original[nrow*i : nrow*(i+1)]
        row_l[2*i+1] = reconstruct[nrow*i : nrow*(i+1)]
    result = torch.cat(row_l)
    ax.imshow(image_grid( result, nrow ))

def visualize(model, data_loader, args):
    # data_loader = test_loader
    original = torch.cat([ x for x, _ in data_loader ]).flatten(1)
    label = torch.cat([ y for _, y in data_loader ])
    psize = 20 if args.jupyter_theme else 15  
    if args.jupyter_theme:
        fig, axes = plt.subplots(1, 3, figsize = [18, 6], dpi = 300)
    else:
        fig, axes = plt.subplots(1, 3, figsize = [13.2, 4.4])
    fig.subplots_adjust(hspace = 0, wspace = 0)             
    plot_latent_space(model, original, label, psize, axes[0])
    plot_manifold(model, 21, 3, axes[1])
    plot_compare(model, original, 20, 10, axes[2])
    axes[1].set_xlabel("z1"), axes[1].set_ylabel("z2"), axes[2].set_xlabel("odd row original, even row reconstruction")
    plt.setp(axes[1:], xticks = [], yticks = [], xticklabels = [], yticklabels = [])
    return fig



# workflow ------------
def main(args_str = ""): pass
    # args_str = ' '.join(sys.argv[1:])
    # arg_list = args_str.split(' ')

if __name__ == "__main__": 
    args = initialize(sys.argv[1:])
    transform_img = torchvision.transforms.RandomAffine(3, [0.02, 0.02]) # P=30%, let change as small as possible
    transform = torchvision.transforms.Compose([ transform_img, torchvision.transforms.ToTensor() ]) 
    train_dataset = torchvision.datasets.MNIST("~/data", train = True , transform = transform)
    test_dataset  = torchvision.datasets.MNIST("~/data", train = False, transform = transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle = True )
    test_loader  = torch.utils.data.DataLoader(test_dataset,  args.batch_size, shuffle = False)
    model = vae_mlp(28**2, 512, 2).cuda()   # keras intermediate is 512
    if args.load_model is not None: model.load_state_dict( torch.load(args.load_model) )
    optimizer = torch.optim.Adam(model.parameters(), args.lr) 
    for epoch in range(args.epochs+1): #pass
        print('='*23 + f" Epoch {epoch:02d} " + '='*23 + '\n' + ' '*13 +"\tloss\t-logP\tKL")
        if epoch > 0:         train(model, train_loader, args, optimizer)
        with torch.no_grad(): train(model, test_loader,  args, None) 
        if epoch > 0: torch.save( model.state_dict(), args.model_file )
        fig = visualize(model, test_loader, args)
        if args.epochs > 0: fig.savefig(f"output/{args.name}-epoch{epoch:02d}.png", bbox_inches = 'tight', pad_inches = 0)
        plt.show() if args.gui else plt.close(fig) 
    