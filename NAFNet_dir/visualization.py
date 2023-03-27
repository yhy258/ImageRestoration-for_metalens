import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import utils
from datasets import SIDDDataset, UDCDatasetPair, GoProDatasetPair

import matplotlib.pyplot as plt

@torch.no_grad()
def visualization(config, nums=4):
    # image가 저장되어있다고 가정. (evaluation을 먼저 할거라서..)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'
    model = utils.define_models(config, device)
    model, loss_info = utils.load_models(config.model_load_path, model)

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((280, 280)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    patch_size = config.patch_size


    dataset = UDCDatasetPair(path=config.test_path, train=False, transform=transform, patch_size=patch_size)

    dataloader = DataLoader(dataset=dataset, batch_size=nums, shuffle=True)
    x, y = next(iter(dataloader))

    x = x.to(device)
    y = y.to(device)

    x_prime = model(y)

    plt.figure(figsize=(20,20))

    x_prime = x_prime.cpu().permute(0, 2, 3, 1)
    x = x.cpu().permute(0, 2, 3, 1)

    x_prime = (x_prime + 1) / 2
    x = (x + 1) / 2

    # subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(nums+1, 2)
    for i in range(nums):
        axarr[i, 0].imshow(x[i])
        axarr[i, 0].grid(False)
        axarr[i, 0].axis(False)

        axarr[i, 1].imshow(x_prime[i])
        axarr[i, 1].grid(False)
        axarr[i, 1].axis(False)


    if config.edge_information :
        axarr[-1, 0].plot(loss_info["Reconstruction Loss"])
        axarr[-1, 0].title.set_text('Reconstruction Loss Plot')
        axarr[-1, 1].plot(loss_info["Edge Loss"])
        axarr[-1, 1].title.set_text('Edge Loss Plot')
    else :
        axarr[-1, 0].plot(loss_info["Total Loss"])
        axarr[-1, 0].title.set_text('Reconstruction Loss Plot')

    plt.show()


if __name__ == "__main__":
    from configure import Config
    config = Config()
    visualization(config, 4)

