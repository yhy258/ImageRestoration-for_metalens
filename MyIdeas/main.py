import os

import torch

from train import train
from configure import Config
from evaluation import evaluation
from visualization import visualization

# TODO! : Evaluation, Visualization, Save Utils
# TODO! : Idea : Fourier Feature, NAF + Prior. (복잡한 Degradation Factor.)
if __name__ == "__main__":
    config = Config()

    print(torch.cuda.is_available())
    print("---------Train---------")
    NAF_model = train(config)

    print("---------Evaluation---------")
    evaluation(config)

    print("Visualization")
    visualization(config, nums=2)
