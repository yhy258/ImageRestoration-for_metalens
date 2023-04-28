# https://github.com/megvii-research/NAFNet/blob/50cb1496d630dbc4165e0ff8f4b5893ca5fa00a1/options/train/SIDD/Baseline-width32.yml
# Model Configure


class Config():
    # data
    # path = "C:/Users/Computer/Desktop/Datasets/SIDD_Medium_Srgb_Parts/SIDD_Medium_Srgb/Data"
    mode = "Prior_NAFNet" # Prior or Plain
    model_type = "Plain"
    vae_type = "Plain"

    path = "/home/eidl/바탕화면/Datasets/UDC_Dataset/Train/Poled"
    test_path = "/home/eidl/바탕화면/Datasets/UDC_Dataset/UDC_val_test/poled"
    patch_size = 256

    # train
    iteration = 200000
    epochs = 15000
    start_epoch = 400
    lr = 1e-3
    batch_size = 16
    edge_information = False
    latent_dim = 64 # 그 다음 64로 늘려보기 이후에는 multi head
    head_num = 1 # my idea
    kld_weight = 5e-3
    # 원래 1e-2

    # decomposition property
    # alpha > 0 이면 decomposition. -> debugging 필요.
    alpha = 5e-3

    img_channel = 3

    width = 32
    middle_blk_num = 12
    enc_blk_nums = [2, 2, 4, 8]
    dec_blk_nums = [2, 2, 2, 2]
    dw_expand = 1
    ffn_expand = 2


    save_root = "/home/eidl/바탕화면/Joon/HYU_EIDL/Project/ImageRestoration/MyIdeas/save_model/udc/poled"
    model_save_period = 100

    model_load_path = f"/home/eidl/바탕화면/Joon/HYU_EIDL/Project/ImageRestoration/MyIdeas/save_model/udc/poled/{mode}64_alpha{alpha}_beta{kld_weight}_14000.pt"

    # eval
    image_save_path = "/home/eidl/바탕화면/Joon/HYU_EIDL/Project/ImageRestoration/MyIdeas/eval_images/udc/poled"
    image_load_paths = ["eval_images/udc/poled/prior_gt_imgs_11000.npy", "eval_images/udc/poled/prior_recon_imgs_11000.npy"]
    sample_num = 500

