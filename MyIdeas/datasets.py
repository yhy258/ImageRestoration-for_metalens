# TODO! : Dataset 구축.


import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from scipy.io.matlab.mio import savemat, loadmat


def get_patch(imgs, patch_size):

    H = imgs[0].shape[0]
    W = imgs[0].shape[1]

    ps_temp = min(H, W, patch_size)

    xx = np.random.randint(0, W-ps_temp) if W > ps_temp else 0
    yy = np.random.randint(0, H-ps_temp) if H > ps_temp else 0

    for i in range(len(imgs)):
        imgs[i] = imgs[i][yy:yy+ps_temp, xx:xx+ps_temp, :]

    if np.random.randint(2, size=1)[0] == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)
    if np.random.randint(2, size=1)[0] == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=0)
    if np.random.randint(2, size=1)[0] == 1:
        for i in range(len(imgs)):
            imgs[i] = np.transpose(imgs[i], (1, 0, 2))

    return imgs

def bit_to_minus_one_one(img):
    img /= 255.
    return img * 2 - 1

def open_images(path_pair):
    x, y = path_pair
    gt_img = Image.open(x).convert("RGB")
    n_img = Image.open(y).convert("RGB")
    return gt_img, n_img

class SIDDDataset(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.transform = transform
        self.train = train
        folders = os.listdir(path)
        self.test_scale = 1/2

        def root_path_join(folder):
            return os.path.join(path, folder)

        self.all_paths = list(map(root_path_join, folders))

        self.path_pairs = []
        for a in self.all_paths:
            fs = os.listdir(a)
            path_pair1 = [os.path.join(a, fs[0]), os.path.join(a, fs[2])]
            path_pair2 = [os.path.join(a, fs[1]), os.path.join(a, fs[3])]
            self.path_pairs.append(path_pair1)
            self.path_pairs.append(path_pair2)

    def __len__(self):
        return len(self.path_pairs)

    def __getitem__(self, idx):
        x, y = open_images(self.path_pairs[idx])

        if not self.train:  # full resolution으로 돌리고싶긴 하다만, 안돌아감.
            H, W = x.size
            new_H, new_W = list(map(int, [self.test_scale * H, self.test_scale * W]))
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((new_H, new_W)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(0.5, 0.5)
            ])
            return transform(x), transform(y)

        else:
            if self.patch_size > 0:
                x, y = np.array(x), np.array(y)
                [x, y] = get_patch([x, y], self.patch_size)
                x = torch.tensor(np.transpose(x, axes=[2, 0, 1]).astype('float32'))
                y = torch.tensor(np.transpose(y, axes=[2, 0, 1]).astype('float32'))

                x = bit_to_minus_one_one(x)
                y = bit_to_minus_one_one(y)


            elif self.transform is not None:
                x = self.transform(x)
                y = self.transform(y)

            return x, y





class GoProDatasetPair(Dataset):
    def __init__(self, path, train=True, gamma=False, transform=None, patch_size=256): # train or test
        super().__init__()
        self.transform = transform
        dir_list = os.listdir(path) # List
        self.patch_size = patch_size
        self.train = train
        self.test_scale = 1/2

        if gamma :
            pair_dir = "blur_gamma"
        else :
            pair_dir = "blur"

        def root_path_join(file):
            return "/".join([path, file])
            # return os.path.join(path, file)

        def blur_post_path_join(dir):
            return "/".join([dir, pair_dir])
            # return os.path.join(dir, pair_dir)

        def sharp_post_path_join(dir):
            return "/".join([dir, "sharp"])
            # return os.path.join(dir, "sharp")



        this_blur = list(map(blur_post_path_join, dir_list))
        this_sharp = list(map(sharp_post_path_join, dir_list))

        blur_paths = list(map(root_path_join, this_blur))
        sharp_paths = list(map(root_path_join, this_sharp))

        self.path_pairs = []
        for bp, sp in zip(blur_paths, sharp_paths) :
            bpf_list = os.listdir(bp)
            spf_list = os.listdir(sp)

            for bpf, spf in zip(bpf_list, spf_list):
                self.path_pairs.append([ "/".join([sp, spf]), "/".join([bp, bpf])])
                # self.path_pairs.append([os.path.join(bp, bpf), os.path.join(sp, spf)])

    def __len__(self):
        return len(self.path_pairs)

    def __getitem__(self, idx):
        x, y = open_images(self.path_pairs[idx])
        if not self.train : # full resolution으로 돌리고싶긴 하다만, 안돌아감.
            H, W = x.size
            new_H, new_W = list(map(int, [self.test_scale * H, self.test_scale * W]))
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((new_H, new_W)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(0.5, 0.5)
            ])
            return transform(x), transform(y)

        else :
            if self.patch_size > 0 :
                x, y = np.array(x), np.array(y)
                [x, y] = get_patch([x, y], self.patch_size)
                x = torch.tensor(np.transpose(x, axes=[2, 0, 1]).astype('float32'))
                y = torch.tensor(np.transpose(y, axes=[2, 0, 1]).astype('float32'))

                x = bit_to_minus_one_one(x)
                y = bit_to_minus_one_one(y)


            elif self.transform is not None:
                x = self.transform(x)
                y = self.transform(y)

            return x, y



class UDCDatasetPair(Dataset):
    def __init__(self, path, train=True, transform=None, patch_size=256):
        super().__init__()
        # path = "C:/Users/Computer/Desktop/Datasets/UDC_Dataset/Train/Poled" -> T, P냐에 따라 Poled, Toled
        # test_path = "C:/Users/Computer/Desktop/Datasets/UDC_Dataset/UDC_val_test/poled"
        # test path의 경우 matlab file
        self.transform = transform
        self.patch_size = patch_size
        self.train = train
        self.test_scale = 1 / 2


        if self.train :
            high_qual = "HQ"
            low_qual = "LQ"

            high_path = os.path.join(path, high_qual)
            low_path = os.path.join(path, low_qual)
            self.path_pairs = []

            for i in range(1, 241):
                this_file_name = f"{i}.png"
                hq_path = os.path.join(high_path, this_file_name)
                lq_path = os.path.join(low_path, this_file_name)
                self.path_pairs.append([hq_path, lq_path])

        else :
            udc_fn = 'poled_test_display.mat'  # or toled_val_display.mat
            gt_udc_fn = 'poled_test_gt.mat'
            udc_key = 'test_display'
            udc_mat = loadmat(os.path.join(path, udc_fn))[udc_key]
            gt_udc_mat = loadmat(os.path.join(path, gt_udc_fn))["test_gt"]

            # restoration
            n_im, h, w, c = udc_mat.shape
            self.n_im = n_im
            self.blur_imgs = np.array(udc_mat).astype("float32") / 255. * 2 - 1
            self.gt_imgs = np.array(gt_udc_mat).astype("float32") / 255. * 2 - 1

            self.blur_imgs = torch.tensor(self.blur_imgs).permute(0, 3, 1, 2)
            self.gt_imgs = torch.tensor(self.gt_imgs).permute(0, 3, 1, 2)


    def __len__(self):
        if self.train :
            return len(self.path_pairs)
        else :
            return self.n_im

    def __getitem__(self, idx):
        # Test 의 경우, 앞에서 언급했듯이, mat file에 데이터가 전체로 들어가있따.
        # 그러므로, path 단위로 꺼내오는게 아니라, 우선 mat file에서 데이터를 모두 불러오고 이 데이터를 indexing해서 배치화 하자.
        if not self.train : # full resolution으로 돌리고싶긴 하다만, 안돌아감.
            N, C, H, W = self.blur_imgs.shape
            new_H, new_W = list(map(int, [self.test_scale * H, self.test_scale * W]))
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((new_H, new_W))
            ])

            return transform(self.gt_imgs[idx]), transform(self.blur_imgs[idx])

        else :
            x, y = open_images(self.path_pairs[idx])
            if self.patch_size > 0 :
                x, y = np.array(x), np.array(y)
                [x, y] = get_patch([x, y], self.patch_size)
                x = torch.tensor(np.transpose(x, axes=[2, 0, 1]).astype('float32'))
                y = torch.tensor(np.transpose(y, axes=[2, 0, 1]).astype('float32'))

                x = bit_to_minus_one_one(x)
                y = bit_to_minus_one_one(y)


            elif self.transform is not None:
                x = self.transform(x)
                y = self.transform(y)

            return x, y



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from evaluation import pnsr_ssim_eval

    path = "C:/Users/Computer/Desktop/Datasets/GOPRO_Large/train"
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256,256)),
        torchvision.transforms.ToTensor()])
    dataset = GoProDatasetPair(path=path, transform=transform)
    # dl = DataLoader(dataset, batch_size=4, shuffle=True)
    # print(len(dataset))
    # it = iter(dl)
    # img = next(it)
    # print(len(img), img[0].shape)
    # print(next(it)[0].shape)
    #
    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(img[0].permute(0, 2, 3, 1)[0])
    # axes[1].imshow(img[1].permute(0, 2, 3, 1)[0])
    #
    # plt.show()
    #
    # psnr, ssim = pnsr_ssim_eval(img[0], img[1])
    # print(psnr, ssim)
    # for i in range(len(dataset)):
    #     if dataset[i].shape

    for i in range(len(dataset)):
        if dataset[0][1].shape[0] == 1 :
            print(i)
