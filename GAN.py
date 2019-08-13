
import os, argparse
import matplotlib.pyplot as plt
import torch, torch.utils.data, torchvision

# ----------
parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("--epochs", type = int, metavar = 'N', default = 100, help = "number of total epochs to run (default: 100)")
parser.add_argument("--gpus", type = int, metavar = "GPU", default = 15, nargs = '+', help = "ordinal of GPUs to use, such as \"0 1\", \"0 1 2 3\" (default: 15)")
args = parser.parse_args()

torch.manual_seed(0)
torch.cuda.set_device(args.gpus[0])
if not os.path.exists('output'):     os.mkdir('output')
if not os.path.exists('output/GAN'): os.mkdir('output/GAN')


# -----------
image_size = 64
batch_size = 64*len(args.gpus)
latent_size = 100
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(image_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
# input for generator
input = torch.randn(batch_size, latent_size, 1, 1).normal_(0, 1).cuda()
# one of input for discriminator (the other is output of generator)
dataset = torchvision.datasets.CIFAR10(root = '~/data/cifar-10', transform = transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle = True)


# -------------------
generator_filter = 64
discriminator_filter = 64
output_channels = 3
# custom weights initialization called on generator and discriminator
def weights_init(module):
    classname = type(module).__name__
    if classname.startswith("Conv"):
        module.weight.data.normal_(0.0, 0.02)
    if classname.startswith("BatchNorm"):
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.zero_()

# Generator
generator = torch.nn.Sequential(                 #  (in - 1)*stride + (kernel - 1) + 1 - padding*2 = out
    # (latent_size)        x  1 x  1
    torch.nn.ConvTranspose2d(latent_size,        generator_filter*8, 4, 1, 0, bias = False),
    torch.nn.BatchNorm2d(generator_filter * 8),  #  (1 - 1)*1 + (4 -1) + 1 - 0*2 = 4
    torch.nn.ReLU(),  
    # (generator_filter*8) x  4 x  4
    torch.nn.ConvTranspose2d(generator_filter*8, generator_filter*4, 4, 2, 1, bias = False),
    torch.nn.BatchNorm2d(generator_filter*4),    #  (4 - 1)*2 + (4 -1) + 1 - 1*2 = 8
    torch.nn.ReLU(),  
    # (generator_filter*4) x  8 x  8
    torch.nn.ConvTranspose2d(generator_filter*4, generator_filter*2, 4, 2, 1, bias = False),
    torch.nn.BatchNorm2d(generator_filter*2),    #  (8 - 1)*2 + (4 -1) + 1 - 1*2 = 16
    torch.nn.ReLU(),  
    # (generator_filter*2) x 16 x 16
    torch.nn.ConvTranspose2d(generator_filter*2, generator_filter,   4, 2, 1, bias = False),
    torch.nn.BatchNorm2d(generator_filter),      # (16 - 1)*2 + (4 -1) + 1 - 1*2 = 32
    torch.nn.ReLU(),  
    # (generator_filter)   x 32 x 32
    torch.nn.ConvTranspose2d(generator_filter,   output_channels,    4, 2, 1, bias = False),
    torch.nn.Tanh()                              # (32 - 1)*2 + (4 -1) + 1 - 1*2 = 64
    # (output_channels)    x 64 x 64
)
generator.apply(weights_init)
generator.cuda()

# Discriminator
discriminator = torch.nn.Sequential(
    # (output_channels)        x 64 x 64
    torch.nn.Conv2d(output_channels,        discriminator_filter,   4, 2, 1, bias = False),
    # (discriminator_filter)   x 32 x 32
    torch.nn.LeakyReLU(0.2),
    torch.nn.Conv2d(discriminator_filter,   discriminator_filter*2, 4, 2, 1, bias = False),
    # (discriminator_filter*2) x 16 x 16
    torch.nn.BatchNorm2d(discriminator_filter * 2),
    torch.nn.LeakyReLU(0.2),
    torch.nn.Conv2d(discriminator_filter*2, discriminator_filter*4, 4, 2, 1, bias = False),
    # (discriminator_filter*4) x  8 x  8
    torch.nn.BatchNorm2d(discriminator_filter * 4),
    torch.nn.LeakyReLU(0.2),
    torch.nn.Conv2d(discriminator_filter*4, discriminator_filter*8, 4, 2, 1, bias = False),
    # (discriminator_filter*8) x  4 x  4
    torch.nn.BatchNorm2d(discriminator_filter * 8),
    torch.nn.LeakyReLU(0.2),
    torch.nn.Conv2d(discriminator_filter*8, 1,                      4, 1, 0, bias = False),
    # (discriminator_filter)   x  1 x  1
    torch.nn.Sigmoid()
)
discriminator.apply(weights_init)
discriminator.cuda()

discriminator = torch.nn.DataParallel(discriminator, args.gpus)
generator     = torch.nn.DataParallel(generator,     args.gpus)
# loss and optimizer
criterion = torch.nn.BCELoss().cuda()
optimizerD = torch.optim.Adam(discriminator.parameters(), 0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(generator.parameters(),     0.0002, betas=(0.5, 0.999))

for epoch in range(args.epochs):
    for i, data in enumerate(data_loader):

        # (1) Update Discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
        discriminator.zero_grad()
        real = data[0].cuda()                         # real image
        output_real = discriminator(real).view(-1)
        lossD_real = criterion( output_real, torch.full_like(output_real, 1) )
        with torch.no_grad(): fake = generator(input) # fake image
        output_fake = discriminator(fake).view(-1)
        lossD_fake = criterion( output_fake, torch.full_like(output_fake, 0) )
        # backward()
        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # (2) Update Generator network: maximize log(D(G(z)))
        generator.zero_grad()
        fake = generator(input)  # graph a second time, but the buffers have already been freed
        output_fake2 = discriminator(fake).view(-1)
        lossG = criterion( output_fake2, torch.full_like(output_fake2, 1) )
        lossG.backward()
        optimizerG.step()

        # (3) print statistics & save images
        if (i+1) % int(len(data_loader)/3) == 0:
            print(f"Epoch{epoch:2d}, batch [{i+1:3d}/{len(data_loader)}], lossD: {lossD:6.3f}, lossG: {lossG:6.3f}" +  
                ", accD: {:5.3f}, accG: {:5.3f} -> {:5.3f}".format( *[x.mean().item() for x in [output_real, output_fake, output_fake2]] ))
            torchvision.utils.save_image(real[:64],  'output/GAN/real_samples.png', normalize = True)
            torchvision.utils.save_image(fake[:64], f'output/GAN/fake_samples_epoch_{epoch:02d}.png', normalize = True)
