import os
import torch
import numpy as np
from model import NAFNet
import pyiqa

def define_models(config, device):
    net = NAFNet(img_channel=config.img_channel, width=config.width, middle_blk_num=config.middle_blk_num,
                 enc_blk_nums=config.enc_blk_nums, dec_blk_nums=config.dec_blk_nums).to(device)
    return net


def iter_loss_info_save(loss_dict, tot_loss, psnr_loss, edge_loss):
    if edge_loss == None :
        loss_dict["Total Loss"].append(tot_loss)
        loss_dict["PSNR Loss"].append(psnr_loss)
    else :
        edge_loss = edge_loss.detach().cpu()
        loss_dict["Total Loss"].append(tot_loss)
        loss_dict["PSNR Loss"].append(psnr_loss)
        loss_dict["Edge Loss"].append(edge_loss)
    return loss_dict

"""
    Loss Info : epoch마다 어떻게 loss가 변화했는지
"""
def save_models(save_path, model, optimizer, scheduler, loss_info, epoch):
    if isinstance(model, list):
        NAFNet, vae = model
        torch.save({
            "model": NAFNet.state_dict(),
            "VAE" : vae.state_dict(),
            "optimizer" : optimizer.state_dict(),
            "scheduler" : scheduler.state_dict(),
            "loss_info": loss_info
        }, os.path.join(save_path, f"NAFNet_{epoch}.pt"))
    else :
        NAFNet = model
        torch.save({
            "model": NAFNet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss_info": loss_info
        }, os.path.join(save_path, f"NAFNet_{epoch}.pt"))

def load_models(load_path, model, optimizer=None, scheduler=None):
    if isinstance(model, list):
        NAFNet, VAE = model
        checkpoint = torch.load(load_path)
        NAFNet.load_state_dict(checkpoint["model"])
        VAE.load_state_dict(checkpoint["VAE"])
        loss_info = checkpoint["loss_info"]

    else :
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model"])
        loss_info = checkpoint["loss_info"]

    if not optimizer == None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    return model, loss_info, optimizer, scheduler

def save_image_bunch_npy(save_path, datas): # shape : BS, C, H, W
    if isinstance(datas, torch.Tensor):
        datas = datas.cpu().numpy()
    np.save(save_path, datas)

def load_image_bunch_npy(load_path):
    return torch.from_numpy(np.load(load_path))


# def extra_loss(mode="ssim"):
#     if mode == "ssim":
#         ExtraLoss = pyiqa.create_metric('ssim', as_loss=True)
#     else :
#         ExtraLoss = pyiqa.create_metric("psnr", as_loss=True)
#     return ExtraLoss

# 4090 쓰는데 일단 720 x 1280 안돌아가;;


