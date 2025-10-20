import numpy as np
import torch, sys
import datasets
import cv2
from torch import optim
from torch.utils.data import ConcatDataset
from algorithms.alg import DGAlg
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
from model.locator import Crowd_locator
from misc.utils import *
from tqdm import tqdm
from misc.compute_metric import eval_metrics
from torch.utils.data import DataLoader
from algorithms.GAM import ProportionScheduler, GAM
from algorithms.SAGM import LinearScheduler, SAGM
from algorithms.SAM import SAMOptim


class Trainer():
    def __init__(self, cfg_data, pwd, args):
        self.args = args
        self.save_results = args.save_results
        self.cfg_data = cfg_data
        train_lst, val_lst, self.test_lst, self.restore_transform = datasets.loading_data(args.dataset, args.mode, args.tmp_root)
        
        concat_dataset=ConcatDataset([loader.dataset for loader in val_lst])
        self.val_loader=DataLoader(
            dataset=concat_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )
        
        self.train_loaders = init_loaders(train_lst, args.batch_size)
            
        self.train_iters = [iter(self.train_loaders[i]) for i in range(len(self.train_loaders))]
        print([f'Length of Source Domain {i}: {len(self.train_loaders[i].dataset)}' for i in range(len(self.train_loaders))])
        print(f'Length of Sets: [ Val: {len(self.val_loader.dataset)}| Test: {sum([len(loader.dataset) for loader in self.test_lst.values()] )} ]')
        self.data_mode = args.dataset
        
        
        
        self.pwd = pwd
        self.net_name = args.model_name
        self.net = Crowd_locator(self.net_name, args.gpuid, sag_net=True if args.DGAlg == 'SagNet' else False, 
                                 args=args)

        self.DGAlgName = args.DGAlg
        self.DG = DGAlg(args)
            
        if self.DGAlgName != 'SAM':
            self.optimizer = optim.Adam([{'params':self.net.Extractor.parameters(), 'lr':args.lr, 'weight_decay':1e-5},
                                            ])
        else:
            base_optimizer = torch.optim.Adam  # define an optimizer for the "sharpness-aware" update
            self.optimizer = SAMOptim(self.net.parameters(), base_optimizer, lr=args.lr, weight_decay=1e-5)

        if self.DGAlgName in ['SAGM']:
            self.lr_scheduler = LinearScheduler(T_max=5e3, max_value=args.lr,
                                    min_value=args.lr, optimizer=self.optimizer)
            self.rho_scheduler = LinearScheduler(T_max=5e3, max_value=0.05,
                                            min_value=0.05)
            self.SAGM_optimizer = SAGM(params=self.net.parameters(), base_optimizer=self.optimizer, model=self.net,
                                alpha=args.SAGM_alpha, rho_scheduler=self.rho_scheduler, adaptive=False)
        elif self.DGAlgName in ['GAM']:
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.num_iter)
            grad_rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=self.lr_scheduler, max_lr=args.lr, min_lr=0.0,
                                                 max_value=args.GAM_grad_rho, min_value=args.GAM_grad_rho)

            grad_norm_rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=self.lr_scheduler, max_lr=args.lr, min_lr=0.0,
                                                        max_value=args.GAM_grad_norm_rho,
                                                        min_value=args.GAM_grad_norm_rho)

            self.GAM_optimizer = GAM(params=self.net.parameters(), base_optimizer=self.optimizer, model=self.net,
                            grad_rho_scheduler=grad_rho_scheduler, grad_norm_rho_scheduler=grad_norm_rho_scheduler,
                            adaptive=args.GAM_adaptive, args=args)

        self.train_record = {'best_F1': 0, 'best_Pre': 0,'best_Rec': 0, 'best_mae': 1e20, 'best_mse':1e20, 'best_nae':1e20, 'best_model_name': ''}
        self.timer={'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}
        self.exp_name = time.strftime("%m-%d_%H-%M", time.localtime()) + '_' + args.mode + '_' + self.data_mode + '_' + args.model_name + '_' + args.DGAlg 
        if args.only_test != 'None':
            self.Best_Record = {
                'F1': 0, 'Model': torch.load(args.only_test)
            }
                
            
            self.exp_name += '_OnlyTest'
            self.only_test = True
        else:
            self.Best_Record = {
                'F1': 0, 'Model': self.net.state_dict()
            }
            self.only_test = False
        if args.DGAlg in ['Mixup', 'EFDM']:
            self.exp_name += '_' + args.mixup_content
        elif args.DGAlg in ['CausalIRL']:
            self.exp_name += '_' + args.CausalIRL_mode
        elif args.DGAlg in ['InfoBot']:
            self.exp_name += '_' + args.InfoBot_mode
        else:
            pass
        self.epoch = 0
        self.i_tb = 0
        self.ifdebug = args.ifdebug

        if 'HR_Net' in self.net_name:
            iter_scaler = 1
        elif 'Res18UNet' in self.net_name:
            iter_scaler = 0.5
        elif 'ViTUNet' in self.net_name:
            iter_scaler = 1.5
        self.num_iters = int(args.num_iter*iter_scaler) if not args.ifdebug else 200
        self.val_freq = args.val_freq if not args.ifdebug else 50
        self.val_start = args.val_start if not args.ifdebug else 0
        

        self.log_txt = logger(args, './exp', self.exp_name, self.pwd, ['ViTexp', 'exp', 'figure', 'oldexp', 'img', 'vis', 'scripts', 'new_exp', 'scripts_line2'], resume=False)
        
    def load_data(self):
        img, gt_map = [], []
        for NumOfTrainIter in range(len(self.train_iters)):
            try:
                tmp_img, tmp_gt = next(self.train_iters[NumOfTrainIter])
            except StopIteration:
                self.train_iters[NumOfTrainIter] = iter(self.train_loaders[NumOfTrainIter])
                tmp_img, tmp_gt = next(self.train_iters[NumOfTrainIter])
            img.append(tmp_img)
            gt_map.append(tmp_gt)
        img = torch.cat(img, dim=0)
        gt_map = torch.cat(gt_map, dim=0)
        return img, gt_map

    def test(self):
        self.net.load_state_dict(self.Best_Record['Model'])
        for test_k in self.test_lst.keys():
            print("#######################", test_k, "#######################")
            self.validate(self.test_lst[test_k], 'test', test_k)
            print("#######################", test_k, "#######################\n\n\n")

    def forward(self):
        if self.ifdebug:
            F1 = self.validate(self.val_loader)[0]
            self.Best_Record = {
                'F1': F1, 'Model': self.net.state_dict()
            }
            self.test()
        if self.only_test:
            self.test()
            return
        self.net.train()
        while self.i_tb < self.num_iters:
            self.i_tb+=1  
            self.timer['train time'].tic()
            img, gt_map = self.load_data()
                
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()

            
            all_loss, pre_map, head_map_loss, penalty_loss, img, gt_map = eval(f'self.DG.{self.DGAlgName}_forward(self.net, img, gt_map)')
            
            if self.DGAlgName in ['SAGM']:
                
                self.SAGM_optimizer.set_closure(img, gt_map)
                pre_map, head_map_loss = self.SAGM_optimizer.step()
                self.lr_scheduler.step()
                self.SAGM_optimizer.update_rho_t()
            elif self.DGAlgName in ['GAM']:
                self.GAM_optimizer.zero_grad()
                self.GAM_optimizer.set_closure(img, gt_map)
                pre_map, head_map_loss = self.GAM_optimizer.step()
                self.lr_scheduler.step()
                with torch.no_grad():
                    self.GAM_optimizer.update_rho_t()
            elif self.DGAlgName in ['SAM']:
                self.optimizer.zero_grad()
                self.DG.SAM_backward(all_loss, self.optimizer, self.net, img, gt_map)
                lr = adjust_learning_rate(self.optimizer,
                                self.args.lr,
                                self.num_iters,
                                self.i_tb)
            else:
                self.optimizer.zero_grad()
                all_loss.backward()
                self.optimizer.step()
                lr = adjust_learning_rate(self.optimizer,
                                self.args.lr,
                                self.num_iters,
                                self.i_tb)
            
            sys.stdout.write('\r')
            sys.stdout.write('[Proceeding %s... %d|%d] [loss %.4f | %.4f] PMean [%.4f | %.4f]'
                    %(self.DGAlgName, self.i_tb, self.num_iters, head_map_loss.item(), penalty_loss,
                        (pre_map * gt_map).mean().detach().cpu().item(), (pre_map * (torch.ones_like(gt_map).to(gt_map.device) - gt_map)).mean().detach().cpu().item()))
            sys.stdout.flush()         
            self.timer['train time'].toc(average=False)

            
            if self.i_tb%self.val_freq==0 and self.i_tb>self.val_start:

                self.timer['val time'].tic()
                F1 = self.validate(self.val_loader)[0]
                if F1 >= self.Best_Record['F1']:
                    self.Best_Record = {
                        'F1': F1, 'Model': self.net.state_dict()
                    }
                self.timer['val time'].toc(average=False)
                print( 'val time: {:.2f}s'.format(self.timer['val time'].diff) )
                self.net.train()

        self.test()

    def get_boxInfo_from_Binar_map(self, Binar_numpy, min_area=3):
        Binar_numpy = Binar_numpy.squeeze().astype(np.uint8)
        assert Binar_numpy.ndim == 2
        cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(Binar_numpy, connectivity=4)  # centriod (w,h)

        boxes = stats[1:, :]
        points = centroids[1:, :]
        index = (boxes[:, 4] >= min_area)
        boxes = boxes[index]
        points = points[index]
        pre_data = {'num': len(points), 'points': points}
        return pre_data, boxes

    def validate(self, loader, val_mode='val', test_env=''):
        self.net.eval()
        losses = AverageMeter()
        cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(), 'nae': AverageMeter()}
        metrics_l = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(), 'tp_c': AverageCategoryMeter(6),
                     'fn_c': AverageCategoryMeter(6)}

        gen_tqdm = tqdm(loader)
        for vi, data in enumerate(gen_tqdm, 0):
            if vi >= 20 and self.ifdebug:
                break
            _, img, mask_map, gt_data = data
            loss, tp_l, fp_l, fn_l, tp_c_l, fn_c_l, s_mae, s_mse, s_nae = self.infer_one(img, mask_map, gt_data)

            losses.update(loss.item())
            metrics_l['tp'].update(tp_l)
            metrics_l['fp'].update(fp_l)
            metrics_l['fn'].update(fn_l)
            metrics_l['tp_c'].update(tp_c_l)
            metrics_l['fn_c'].update(fn_c_l)
            cnt_errors['mae'].update(s_mae)
            cnt_errors['mse'].update(s_mse)
            if gt_data['num'].numpy().astype(float) != 0:
                cnt_errors['nae'].update(s_nae)            
        ap_l = metrics_l['tp'].sum / (metrics_l['tp'].sum + metrics_l['fp'].sum + 1e-20)
        ar_l = metrics_l['tp'].sum / (metrics_l['tp'].sum + metrics_l['fn'].sum + 1e-20)
        f1m_l = 2 * ap_l * ar_l / (ap_l + ar_l+ 1e-20)
        ar_c_l = metrics_l['tp_c'].sum / (metrics_l['tp_c'].sum + metrics_l['fn_c'].sum + 1e-20)
        loss = losses.avg
        mae = cnt_errors['mae'].avg
        mse = np.sqrt(cnt_errors['mse'].avg)
        nae = cnt_errors['nae'].avg
        self.train_record = update_model(self, [f1m_l, ap_l, ar_l,mae, mse, nae, loss, val_mode+str(test_env)])
        print_summary(self,[f1m_l, ap_l, ar_l,mae, mse, nae, loss, val_mode+str(test_env)])
        return [f1m_l, ap_l, ar_l, mae, mse, nae, loss]
    
    def infer_one(self, img, gt_map, gt_data):
        slice_h, slice_w = self.cfg_data.TRAIN_SIZE
     
        with torch.no_grad():
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()
            crop_imgs, crop_gt, crop_masks = [], [], []
            b, c, h, w = img.shape

            if h == slice_h and w == slice_w:
                [_, pred_map]= [i.cpu() for i in self.net(img, mask_gt=None, mode = 'val')]
            else:
                if h % 32 !=0:
                    pad_dims = (0,0, 0,32-h%32)
                    h = (h//32+1)*32
                    img = F.pad(img, pad_dims, "constant")
                    gt_map = F.pad(gt_map, pad_dims, "constant")

                if w % 32 !=0:
                    pad_dims = (0, 32-w%32, 0, 0)
                    w =  (w//32+1)*32
                    img = F.pad(img, pad_dims, "constant")
                    gt_map = F.pad(gt_map, pad_dims, "constant")

                assert img.size()[2:] == gt_map.size()[2:]

                for i in range(0, h, slice_h):
                    h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                    for j in range(0, w, slice_w):
                        w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)

                        crop_imgs.append(img[:, :, h_start:h_end, w_start:w_end])
                        crop_gt.append(gt_map[:, :, h_start:h_end, w_start:w_end])
                        mask = torch.zeros_like(gt_map).cpu()
                        mask[:, :,h_start:h_end, w_start:w_end].fill_(1.0)
                        crop_masks.append(mask)
                crop_imgs, crop_gt, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_gt, crop_masks))

                # forward may need repeatng
                crop_preds = []
                nz, period = crop_imgs.size(0), self.cfg_data.TRAIN_BATCH_SIZE
                for i in range(0, nz, period):
                    [_, crop_pred] = [i.cpu() for i in self.net(crop_imgs[i:min(nz, i+period)],mask_gt = None, mode='val')]
                    crop_preds.append(crop_pred)
                    

                crop_preds = torch.cat(crop_preds, dim=0)
                

                # splice them to the original size
                idx = 0
                pred_map = torch.zeros_like(gt_map).cpu().float()
                
                for i in range(0, h, slice_h):
                    h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                    for j in range(0, w, slice_w):
                        w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                        pred_map[:, :, h_start:h_end, w_start:w_end]  += crop_preds[idx]
                        
                        idx += 1

            # for the overlapping area, compute average value
                mask = crop_masks.sum(dim=0)
                pred_map = (pred_map / mask)
                
            pred_threshold = (torch.ones_like(pred_map) * .5 ).to(pred_map.device)
            # binar_map = self.net.Binar(pred_map.cuda(), pred_threshold.cuda()).cpu()
            a = torch.ones_like(pred_map)
            b = torch.zeros_like(pred_map)
            binar_map = torch.where(pred_map >= pred_threshold, a, b)

            gt_map = gt_map.cpu()
        loss = F.mse_loss(pred_map, gt_map)
        binar_map = binar_map.numpy()
        pred_data, boxes = self.get_boxInfo_from_Binar_map(binar_map)
        tp_s, fp_s, fn_s, tp_c_s, fn_c_s, tp_l, fp_l, fn_l, tp_c_l, fn_c_l = eval_metrics(6, pred_data, gt_data)
        gt_count, pred_cnt = gt_data['num'].numpy().astype(float), pred_data['num']
        s_mae = abs(gt_count - pred_cnt)
        s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
        s_nae = (abs(gt_count - pred_cnt) / gt_count) if gt_count != 0 else 0
        return loss, tp_l, fp_l, fn_l, tp_c_l, fn_c_l, s_mae, s_mse, s_nae

   
   



