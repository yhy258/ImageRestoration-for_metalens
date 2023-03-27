# https://github.com/megvii-research/NAFNet/blob/50cb1496d630dbc4165e0ff8f4b5893ca5fa00a1/options/train/SIDD/Baseline-width32.yml
# Model Configure

class Config():
    # data
    # path = "C:/Users/Computer/Desktop/Datasets/SIDD_Medium_Srgb_Parts/SIDD_Medium_Srgb/Data"
    path = "C:/Users/Computer/Desktop/Datasets/UDC_Dataset/Train/Poled"
    test_path = "C:/Users/Computer/Desktop/Datasets/UDC_Dataset/UDC_val_test/poled"
    patch_size = 256

    # train
    iteration = 200000
    epochs = 20000
    start_epoch = 5000 # epoch ---> scheduler last_epoch 조정. epoch * iteration_per_epoch
    lr = 1e-3
    batch_size = 16
    edge_information = False
    head_num = 1 # my idea

    img_channel = 3

    width = 32
    middle_blk_num = 12
    enc_blk_nums = [2, 2, 4, 8]
    dec_blk_nums = [2, 2, 2, 2]
    dw_expand = 1
    ffn_expand = 2


    save_root = "save_model/udc/poled"
    model_save_period = 200

    model_load_path = "save_model/udc/poled/NAFNet_3000.pt"

    # eval
    image_save_path = "eval_images/udc/poled"
    image_load_paths = ["eval_images/udc/poled/gt_imgs_3000.npy", "eval_images/udc/poled/recon_imgs_3000.npy"]
    sample_num = 500


import torch
from configure import Config

from utils import define_models

from tqdm import tqdm
if __name__ == "__main__":

    config =Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print("Define Models")
    NAF_model = define_models(config, device=device)
    optimizer = torch.optim.AdamW(params=NAF_model.parameters(), lr=config.lr, betas=[0.9, 0.9])
    cosine_annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.iteration, eta_min=1e-7, last_epoch=-1 , verbose=False)
    cosine_annealing_scheduler.last_epoch = config.start_epoch
    print(cosine_annealing_scheduler.get_last_lr())

