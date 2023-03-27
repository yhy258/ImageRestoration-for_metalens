import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
from utils import define_models
from configure import Config
from train import GradLayer

if __name__ == "__main__":
    with torch.cuda.device(0):
        config = Config()
        device = 'cpu'
        NAF_model = define_models(config, device=device)
        edge_extractor = GradLayer()

        macs, params = get_model_complexity_info(NAF_model, (3, 256, 256), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
        edge_macs, edge_params = get_model_complexity_info(edge_extractor, (3, 256, 256), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('(NAFNet) Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('(NAFNet) Number of parameters: ', params))
        print('{:<30}  {:<8}'.format('(Edge Extractor) Computational complexity: ', edge_macs))
        print('{:<30}  {:<8}'.format('(Edge Extractor) Number of parameters: ', edge_params))