import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pyiqa

import datasets
from datasets import SIDDDataset, UDCDatasetPair, GoProDatasetPair

import utils

def minus_one_one_to8bit(img, clip=True):
    if clip :
        return torch.clip((img + 1) / 2., 0., 1.) * 255.
    else :
        return (img + 1) / 2. * 255.
def minus_one_one_to_zero_one(img):
    return torch.clip((img + 1) / 2., 0., 1.)


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2): # bs, c, h, w
        img1, img2 = list(map(minus_one_one_to8bit, [img1, img2]))
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        # 입력 경계의 반사를 사용하여 상/하/좌/우에 입력 텐서를 추가로 채웁니다.
        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x, y = list(map(minus_one_one_to_zero_one, [x, y]))
        print(x.max(), y.max())
        # shape : (xh, xw) -> (xh + 2, xw + 2)
        x = self.refl(x)
        # shape : (yh, yw) -> (yh + 2, yw + 2)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        # SSIM score
        return torch.mean(torch.clamp((SSIM_n / SSIM_d) / 2, 0, 1))

        # Loss function
        # return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

def pnsr_ssim_eval(Xs, Ys):
    pnsr_class = PSNR()
    ssim_class = SSIM()

    # pnsr = pnsr_class(Xs, Ys)
    # ssim = ssim_class(Xs, Ys)
    x, y = list(map(minus_one_one_to_zero_one, [Xs, Ys]))
    psnr_metric = pyiqa.create_metric('psnr')
    ssim_metric = pyiqa.create_metric('ssim')
    psnr = psnr_metric(x, y)
    ssim = ssim_metric(x, y)
    return torch.mean(psnr), torch.mean(ssim)

"""
    사전학습된 모델로 test 이미지 쌍을 만들어준다.
"""
def prepare_and_save_images(config, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    model = utils.define_models(config, device)
    model, loss_info, _, _ = utils.load_models(config.model_load_path, model)

    model.eval()

    gt_samples = []
    recon_samples = []

    ep = int(config.model_load_path.split('/')[-1][7:-3])

    print("LOG : Save Image Data.npy")
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i * config.batch_size > config.sample_num:
                break
            x, y = x.to(device), y.to(device)
            x_prime = model(y)

            gt_samples.append(x.detach().cpu())
            recon_samples.append(x_prime.detach().cpu())

    gt_samples = torch.cat(gt_samples, dim=0)[:config.sample_num]
    recon_samples = torch.cat(recon_samples, dim=0)[:config.sample_num]


    gt_path = os.path.join(config.image_save_path, f"gt_imgs_{ep}.npy")
    recon_path = os.path.join(config.image_save_path, f"recon_imgs_{ep}.npy")
    utils.save_image_bunch_npy(gt_path, gt_samples)
    utils.save_image_bunch_npy(recon_path, recon_samples)

    return gt_samples, recon_samples


def evaluation(config):

    if os.path.isfile(config.image_load_paths[0]) and os.path.isfile(config.image_load_paths[1]):
        print("Just Evaluation")
        gt_path = config.image_load_paths[0]
        recon_path = config.image_load_paths[1]
        gt_samples = utils.load_image_bunch_npy(gt_path)
        recon_samples = utils.load_image_bunch_npy(recon_path)

    else :
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        dataset = UDCDatasetPair(path=config.test_path, train=False, transform=transform, patch_size=0)
        dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True)
        gt_samples, recon_samples = prepare_and_save_images(config, dataloader)



    pnsr, ssim = pnsr_ssim_eval(gt_samples, recon_samples)
    print(f"-- PNSR : {pnsr} \t SSIM : {ssim} --")



# import copy
# import matplotlib.pyplot as plt
# @torch.no_grad()
# def compare_to_NAFmodel(config, model1_path, model2_path, imgs):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model1 = utils.define_models(config, device)
#     model2 = copy.deepcopy(model1)
#     model1, loss_info1 = utils.load_models(model1_path, model1)
#     model2, loss_info2 = utils.load_models(model2_path, model2)
#
#     fig, axes = plt.subplots(2,3)
#
#     img, sharp_img = imgs
#
#     sharp_img = (sharp_img + 1) / 2
#     sharp_img = sharp_img.permute(0, 2, 3, 1)[0]
#
#     if len(img.shape) == 3 :
#         img = img.unsqueeze(0)
#     img = img.to(device)
#     x1_prime = model1(img)
#     x1_prime = (x1_prime + 1) / 2
#     x2_prime = model2(img)
#     x2_prime = (x2_prime + 1) / 2
#
#
#     axes[0, 1].imshow(x1_prime.cpu().permute(0, 2, 3, 1)[0])
#     axes[0, 0].plot(loss_info1)
#     axes[1, 1].imshow(x2_prime.cpu().permute(0, 2, 3, 1)[0])
#     axes[1, 0].plot(loss_info2["Total Loss"])
#     axes[0, 2].imshow(sharp_img)
#     axes[1, 2].imshow(sharp_img)
#     plt.show()
#
#
# if __name__ == "__main__":
#     from configure import Config
#     from PIL import Image
#     import numpy as np
#     config =Config()
#     model1_path = "save_model/gopro/patch_gopro_NAFNet_500.pt"
#     model2_path = "save_model/gopro/patch_gopro_NAFNet_edgeinfo_500.pt"
#     img_path = "C:/Users/Computer/Desktop/Datasets/GOPRO_Large/test/GOPR0385_11_01/blur/003101.png"
#     sharp_img_path = "C:/Users/Computer/Desktop/Datasets/GOPRO_Large/test/GOPR0385_11_01/sharp/003101.png"
#     blur_img = np.array(Image.open(img_path))
#     sharp_img = np.array(Image.open(sharp_img_path))
#
#     blur_img, sharp_img = datasets.get_patch([blur_img, sharp_img], 256)
#     blur_img = torch.tensor(np.transpose(blur_img, axes=[2, 0, 1]).astype('float32'))
#     blur_img = datasets.bit_to_minus_one_one(blur_img)
#     blur_img = blur_img.view(1, 3, 256, 256)
#
#     sharp_img = torch.tensor(np.transpose(sharp_img, axes=[2, 0, 1]).astype('float32'))
#     sharp_img = datasets.bit_to_minus_one_one(sharp_img)
#     sharp_img = sharp_img.view(1, 3, 256, 256)
#
#     compare_to_NAFmodel(config, model1_path, model2_path, [blur_img, sharp_img])



