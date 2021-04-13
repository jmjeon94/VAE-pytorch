import torch
import matplotlib.pyplot as plt
from model import VAE
from torchvision import datasets, transforms
import numpy as np
import matplotlib.cm as cm
from tqdm import tqdm

def show(imgs, cmap='gray'):
    for i, img in enumerate(imgs):
        img = np.squeeze(img)
        plt.subplot(1, len(imgs), i+1)
        plt.imshow(img, cmap)
    plt.show()


def show_imgs(model, dataloader):
    for img, target in dataloader:

        output, _, _ = model(img)
        show([img[0].detach().numpy(), output[0].detach().numpy()])

        if input('key:')=='x':
            break

def show_2D_feature(model, dataloader):
    colors = cm.rainbow(np.linspace(0, 1, 10))

    for i, (img, target) in enumerate(tqdm(dataloader)):

        output, _, _ = model.encoder(img)
        output = torch.squeeze(output).detach().numpy()
        plt.scatter(output[:,0], output[:,1], color=colors[target], s=4)

    plt.show()

def random_sample(model, cmap='gray'):
    n_to_show = 20
    znew = torch.randn((n_to_show, 2))
    recon = model.decoder(znew)

    for i in range(n_to_show):
        plt.subplot(4, 5, i+1)
        plt.imshow(torch.squeeze(recon[i]).detach().numpy(), cmap)
        plt.xticks([]); plt.yticks([])
    plt.show()

if __name__=='__main__':
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

    model = VAE()
    model.load_state_dict(torch.load('./checkpoints/epoch_050.pth'))

    show_2D_feature(model, test_loader)
    random_sample(model)
    # show_imgs(model, test_loader)
