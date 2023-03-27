# TODO! : 졸작 기준. VDIR 모델 구축. (잘 되는지 확인..)
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import SIDDDataset, UDCDatasetPair, GoProDatasetPair

import utils
from utils import define_models
import evaluation

from tqdm import tqdm

# # # torch version으로 바꾸장
# class Base2FourierFeatures(nn.Module):
#     def __init__(self, start=0, stop=8): # stop을 제외하고 이전까지
#         super(Base2FourierFeatures, self).__init__()
#         self.freqs = torch.arange(start, stop, 1)
#
#         w = 2 ** self.freqs * 2 * math.pi
#
#
# class Base2FourierFeatures(nn.Module):
#   start: int = 0
#   stop: int = 8
#   step: int = 1
#
#   @nn.compact
#   def __call__(self, inputs):
#     freqs = range(self.start, self.stop, self.step)
#
#     # Create Base 2 Fourier features
#     w = 2.**(jnp.asarray(freqs, dtype=inputs.dtype)) * 2 * jnp.pi
#     w = jnp.tile(w[None, :], (1, inputs.shape[-1]))
#
#     # Compute features
#     h = jnp.repeat(inputs, len(freqs), axis=-1)
#     h = w * h
#     h = jnp.concatenate([jnp.sin(h), jnp.cos(h)], axis=-1)
#     return h

class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

        # 결과에 대해 3x3 conv2d filter (grouped conv)
        # fourier feature는


    def forward(self, x):
        # x_list = []
        # for i in range(x.shape[1]):
        #     x_i = x[:, i]
        #     x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
        #     x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
        #     x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
        #     x_list.append(x_i)

        # x = torch.cat(x_list, dim=1)


        x_v = F.conv2d(x, self.weight_v, padding=1, groups=3)
        x_h = F.conv2d(x, self.weight_h, padding=1, groups=3)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x


def edge_loss_func(x, y, grad_filter):

    x = evaluation.minus_one_one_to8bit(x, clip=False)
    y = evaluation.minus_one_one_to8bit(y, clip=False)
    x_edge = grad_filter(x)
    y_edge = grad_filter(y)
    return (torch.abs(x_edge - y_edge) / 255.).mean()


def train(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print("Define Models")
    NAF_model = define_models(config, device=device)


    optimizer = torch.optim.AdamW(params=NAF_model.parameters(), lr=config.lr, betas=[0.9, 0.9])
    # 15를 곱해주는 이유 : UDCDataset 기준, 1epoch당 15 iteration이 돌아갔기 때문.

    cosine_annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.iteration, eta_min=1e-7, last_epoch=-1 , verbose=False)

    # Dataset 정의

    transform = transforms.Compose([
        transforms.Resize((280, 280)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = UDCDatasetPair(path=config.path, transform=transform, patch_size=config.patch_size)

    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True)


    all_loss_dict = None
    if config.start_epoch > config.model_save_period :
        print(f"Load EPOCH : {config.start_epoch} Model")
        ep = config.start_epoch
        load_path = os.path.join(config.save_root, f"NAFNet_{ep}.pt")
        NAF_model, all_loss_dict, optimizer, cosine_annealing_scheduler = utils.load_models(load_path, NAF_model, optimizer, cosine_annealing_scheduler)


    # lossfunc = utils.extra_loss('ssim')
    lossfunc = evaluation.PSNR()

    if config.edge_information :
        grad_filter = GradLayer().to(device)


    """
    
       GAN쪽 코딩이 맞는지 모르겟다. 
    """
    print("-- Start Train Loop --")

    this_iter = 0
    if all_loss_dict == None:
        all_loss_dict = {"Total Loss" : [], "PSNR Loss" : [], "Edge Loss" : []}

    for ep in range(config.start_epoch, config.epochs):
        print(f"-- Now Epoch : {ep+1} / {config.epochs} --")
        loss_dict = {"Total Loss" : [], "PSNR Loss" : [], "Edge Loss" : []}

        for i, (x, y) in enumerate(tqdm(dataloader)): # x : clean image, y : degraded image
            x, y = x.to(device), y.to(device)

            x_prime = NAF_model(y)

            x, x_prime = list(map(evaluation.minus_one_one_to_zero_one, [x, x_prime]))
            edge_loss = None
            psnr_loss = -lossfunc(x_prime, x)
            tot_loss = psnr_loss
            if config.edge_information :
                edge_loss = edge_loss_func(x, x_prime, grad_filter)
                tot_loss = psnr_loss + edge_loss
            loss_dict = utils.iter_loss_info_save(loss_dict, tot_loss.detach().cpu(), psnr_loss.detach().cpu(), edge_loss)

            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()
            cosine_annealing_scheduler.step()

            this_iter += 1

            if this_iter > config.iteration :

                break

        this_PSNR_Loss = np.mean(loss_dict["PSNR Loss"])
        this_loss = np.mean(loss_dict['Total Loss'])
        if config.edge_information :
            this_Edge_Loss = np.mean(loss_dict["Edge Loss"])
            print(f"Total Loss : {this_loss}\t PSNR Loss : {this_PSNR_Loss}\t Edge Loss : {this_Edge_Loss}")
        else :
            print(f"Total Loss : {this_loss}")
        all_loss_dict["Total Loss"].append(this_loss)
        all_loss_dict["PSNR Loss"].append(this_PSNR_Loss)

        if config.edge_information :
            all_loss_dict["Edge Loss"].append(this_Edge_Loss)


        if (ep > 0) and ((ep + 1) % config.model_save_period == 0) :
            utils.save_models(config.save_root, NAF_model, optimizer, cosine_annealing_scheduler, all_loss_dict, ep+1)

        if this_iter >= config.iteration:
            utils.save_models(config.save_root, NAF_model, optimizer, cosine_annealing_scheduler, all_loss_dict, ep+1)
            break

    return NAF_model
