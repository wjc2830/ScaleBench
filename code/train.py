import os
import numpy as np
import torch
import datasets
from importlib import import_module
import argparse
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='PyTorch ScaleBench Training')

parser.add_argument('--tmp_root', default='ScaleBenchDataRootHere', 
                    type=str)
parser.add_argument('--DGAlg', default='Mixup', type=str, 
                    help='Mixup, DANN, CORAL, EFDM, IRM, MMD, RSC, VRex')
parser.add_argument('--model_name', default='HR_Net', type=str, 
                    help='[HR_Net] [ViTUNet] [Res18UNet]')
parser.add_argument('--seed', default=1, type=int)


parser.add_argument('--only_test', default='None', type=str)
parser.add_argument('--ifdebug', default=False, type=bool)


parser.add_argument('--Pre_HR_Net', default="HRNet.pth", type=str, help='train batchsize') 
parser.add_argument('--batch_size', default=16, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float, help='initial learning rate')
parser.add_argument('--num_iter', default=3e4, type=int)
parser.add_argument('--gpuid', default='0', type=str)
parser.add_argument('--val_freq', default=1e3, type=int)
parser.add_argument('--val_start', default=1e4, type=int)

parser.add_argument('--dataset', default='data1', type=str)
parser.add_argument('--mode', default='DG', type=str)
parser.add_argument('--save_results', default='', type=str)
parser.add_argument('--penalty_param', default=.3, type=float)
parser.add_argument('--sup_param', default=1., type=float)
parser.add_argument('--num_domains', default=3, type=int)

parser.add_argument('--mixup_content', default='img', type=str)
parser.add_argument('--mixupalpha', default=.2, type=float)

parser.add_argument('--DANN_disc_alpha', default=.2, type=float)
parser.add_argument('--DANN_feat_dim', default=768, type=int, 
                    help="512 for UNet; 720 for HRNet; 768 for ViT")

parser.add_argument('--anneal_iters', default=1e4, type=int)

parser.add_argument('--SagNet_pixel_recor', default=False, type=bool)
parser.add_argument('--SagNet_eps', default=1e-5, type=float)
parser.add_argument('--SAGM_alpha', default=.001, type=float)
parser.add_argument('--SAGM_rho', default=.05, type=float)

parser.add_argument('--GAM_grad_beta_0', default=.5, type=float)
parser.add_argument('--GAM_grad_beta_1', default=.6, type=float)
parser.add_argument('--GAM_grad_beta_2', default=.5, type=float)
parser.add_argument('--GAM_grad_beta_3', default=.4, type=float)
parser.add_argument('--GAM_grad_gamma', default=.03, type=float)
parser.add_argument('--GAM_grad_rho', default=.02, type=float)
parser.add_argument('--GAM_grad_norm_rho', default=.2, type=float)
parser.add_argument('--GAM_adaptive', default=False, type=bool)

parser.add_argument('--HGP_penalty_alpha', default=1e-4, type=float)
parser.add_argument('--HGP_penalty_beta', default=5e-3, type=float)

parser.add_argument('--DomainDrop_drop_percent', default=.33, type=float)
parser.add_argument('--DomainDrop_lambd', default=.25, type=float)
parser.add_argument('--DomainDrop_consis_param', default=1.5, type=float)
parser.add_argument('--DomainDrop_layer_wise_prob', default=.8, type=float)
parser.add_argument('--DomainDrop_discriminator_layers', default=[1, 2, 3, 4], type=list)


parser.add_argument('--InfoBot_mode', default='ERM', type=str)
parser.add_argument('--InfoBot_penalty_param', default=1., type=float)


parser.add_argument('--CausalIRL_mode', default='gaussian', type=str)

parser.add_argument('--SemanticHook_l0', default=.5, type=float)
parser.add_argument('--SemanticHook_alpha', default=10, type=float)
parser.add_argument('--SemanticHook_beta', default=.75, type=float)
parser.add_argument('--SemanticHook_NoiseScale', default=0.1, type=float)







args = parser.parse_args()



np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
torch.backends.cudnn.benchmark = True


#------------prepare data loader------------
data_mode = 'data1'

datasetting = import_module(f'datasets.setting.{data_mode}')
cfg_data = datasetting.cfg_data


        #------------Prepare Trainer------------domain
from trainer import Trainer

        #------------Start Training------------
pwd = os.path.split(os.path.realpath(__file__))[0]
cc_trainer = Trainer(cfg_data, pwd, args)
cc_trainer.forward()




