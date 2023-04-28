import os
import torch
import numpy as np
from model import NAFNet, NAFNetLocal
from vae_model import VanillaVAE, PowerVAE
from light_NAFNet import Light_NAFNet, LightNAFNetLocal
import pyiqa

def define_models(config, device, local=False):

    if config.model_type == "Light" :
        net = Light_NAFNet(img_channel=config.img_channel, width=config.width, middle_blk_num=config.middle_blk_num,
                     enc_blk_nums=config.enc_blk_nums, dec_blk_nums=config.dec_blk_nums, latend_dim=config.latent_dim,
                     head_num=config.head_num, mode=config.mode).to(device)
        if local :
            net = LightNAFNetLocal(img_channel=config.img_channel, width=config.width, middle_blk_num=config.middle_blk_num,
                         enc_blk_nums=config.enc_blk_nums, dec_blk_nums=config.dec_blk_nums,
                         latend_dim=config.latent_dim,
                         head_num=config.head_num, mode=config.mode).to(device)


    else :
        net = NAFNet(img_channel=config.img_channel, width=config.width, middle_blk_num=config.middle_blk_num,
                     enc_blk_nums=config.enc_blk_nums, dec_blk_nums=config.dec_blk_nums, latend_dim=config.latent_dim,
                     head_num=config.head_num, mode=config.mode).to(device)
        if local:
            net = NAFNetLocal(img_channel=config.img_channel, width=config.width, middle_blk_num=config.middle_blk_num,
                              enc_blk_nums=config.enc_blk_nums, dec_blk_nums=config.dec_blk_nums,
                              latend_dim=config.latent_dim, prior=config.latent_dim * config.head_num,
                              head_num=config.head_num, mode=config.mode).to(device)

    if config.mode == "Prior_NAFNet" :
        if config.vae_type == "Power":
            vae = PowerVAE(in_channels=config.img_channel, latent_dim=config.latent_dim, head_nums=config.head_num).to(device)
        else :
            vae = VanillaVAE(in_channels=config.img_channel, latent_dim=config.latent_dim, head_nums=config.head_num).to(device)
        return [net, vae]

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
def save_models(config, model, optimizer, scheduler, loss_info, epoch):


    if isinstance(model, list):
        if config.vae_type == "Power":
            save_path = f"Power_{config.mode}{config.latent_dim}_alpha{config.alpha}_beta{config.kld_weight}_{epoch}.pt"
        else:
            save_path = f"{config.mode}{config.latent_dim}_alpha{config.alpha}_beta{config.kld_weight}_{epoch}.pt"

        NAFNet, vae = model
        torch.save({
            "model": NAFNet.state_dict(),
            "VAE" : vae.state_dict(),
            "optimizer" : optimizer.state_dict(),
            "scheduler" : scheduler.state_dict(),
            "loss_info": loss_info
        }, os.path.join(config.save_root, f"{config.mode}{config.latent_dim}_alpha{config.alpha}_beta{config.kld_weight}_{epoch}.pt"))
    else :
        NAFNet = model
        torch.save({
            "model": NAFNet.state_dict(),
            "loss_info": loss_info
        }, os.path.join(config.save_root, f"{config.mode}_{config.model_type}_{epoch}.pt"))

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


import sys
import math
import time
import os
import shutil
import torch
import torch.distributions as dist
import numpy as np

# Classes
class Constants(object):
    eta = 1e-6
    log2 = math.log(2)
    log2pi = math.log(2 * math.pi)
    logceilc = 88                # largest cuda v s.t. exp(v) < inf
    logfloorc = -104             # smallest cuda v s.t. exp(v) > 0
    invsqrt2pi = 1. / math.sqrt(2 * math.pi)

# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)
        print('====> [{}] Time: {:7.3f}s or {}'
              .format(self.name,
                      self.elapsed,
                      time.strftime("%H:%M:%S", self.elapsedH)))


# Functions
def save_vars(vs, filepath):
    """
    Saves variables to the given filepath in a safe manner.
    """
    if os.path.exists(filepath):
        shutil.copyfile(filepath, '{}.old'.format(filepath))
    torch.save(vs, filepath)


def save_model(model, filepath):
    """
    To load a saved model, simply use
    `model.load_state_dict(torch.load('path-to-saved-model'))`.
    """
    save_vars(model.state_dict(), filepath)


def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


def has_analytic_kl(type_p, type_q):
    return (type_p, type_q) in torch.distributions.kl._KL_REGISTRY

def kl_divergence(p, q, samples=None):
    if has_analytic_kl(type(p), type(q)):
        return dist.kl_divergence(p, q)
    else:
        if samples is None:
            K = 10
            samples = p.rsample(torch.Size([K])) if p.has_rsample else p.sample(torch.Size([K]))
        # ent = p.entropy().unsqueeze(0) if hasattr(p, 'entropy') else -p.log_prob(samples)
        ent = -p.log_prob(samples)
        return (-ent - q.log_prob(samples)).mean(0)

def normal_kl_divergence(mu1, sig1, mu2, sig2):
    return torch.mean(torch.log(sig2/(sig1+1e-6)) + (sig1**2 + (mu1 - mu2)**2) / (2*sig2**2) - 1/2)



# def extra_loss(mode="ssim"):
#     if mode == "ssim":
#         ExtraLoss = pyiqa.create_metric('ssim', as_loss=True)
#     else :
#         ExtraLoss = pyiqa.create_metric("psnr", as_loss=True)
#     return ExtraLoss

# 4090 쓰는데 일단 720 x 1280 안돌아가;;


