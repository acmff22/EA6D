import numpy as np
import yaml
import argparse
import math
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from denoising_diffusion_pytorch.utils import *
import torchvision as tv
from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
from denoising_diffusion_pytorch.uncond_unet import Unet
from denoising_diffusion_pytorch.data import *
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from fvcore.common.config import CfgNode
from scipy import integrate
import torchvision.transforms as T
from PIL import Image
from metrics.metric_utils import *
from metrics.model import Model3D
from scipy.spatial.transform import Rotation as R
import os
from denoising_diffusion_pytorch.rotation_decoder import RAEncoder as pose_decoder
from denoising_diffusion_pytorch.main_network import mainNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="training vae configure")
    parser.add_argument("--cfg", help="experiment configure file name", type=str, required=True)
    args = parser.parse_args()
    args.cfg = load_conf(args.cfg)
    return args

def load_conf(config_file, conf={}):
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in exp_conf.items():
            conf[k] = v
    return conf

def rot2quaternion(rot):   
        rot = rot[:3,:3]
        quat = R.from_matrix(rot).as_quat()
        return quat      

def quaternion2rot(quat):    #四元数转旋转矩阵
        rot  = R.from_quat(quat).as_matrix()
        return rot        

# Colors for all 20 parts
part_colors = [[0, 0, 0], [255, 85, 0],  [255, 170, 0],
               [255, 0, 85], [255, 0, 170],
               [0, 255, 0], [85, 255, 0], [170, 255, 0],
               [0, 255, 85], [0, 255, 170],
               [0, 0, 255], [85, 0, 255], [170, 0, 255],
               [0, 85, 255], [0, 170, 255],
               [255, 255, 0], [255, 255, 85], [255, 255, 170],
               [255, 0, 255], [255, 85, 255], [255, 170, 255],
               [0, 255, 255], [85, 255, 255], [170, 255, 255]]

def main(args):
    cfg = CfgNode(args.cfg)
    torch.manual_seed(42)
    np.random.seed(42)
    model_cfg = cfg.model
    first_stage_cfg = model_cfg.first_stage
    first_stage_model = AutoencoderKL(
        ddconfig=first_stage_cfg.ddconfig,
        lossconfig=first_stage_cfg.lossconfig,
        embed_dim=first_stage_cfg.embed_dim,
        ckpt_path=first_stage_cfg.ckpt_path,
    )

    if model_cfg.model_name == 'cond_unet':
        from denoising_diffusion_pytorch.mask_cond_unet import Unet
        unet_cfg = model_cfg.unet
        unet = Unet(dim=unet_cfg.dim,
                    channels=unet_cfg.channels,
                    dim_mults=unet_cfg.dim_mults,
                    learned_variance=unet_cfg.get('learned_variance', False),
                    out_mul=unet_cfg.out_mul,
                    cond_in_dim=unet_cfg.cond_in_dim,
                    cond_dim=unet_cfg.cond_dim,
                    cond_dim_mults=unet_cfg.cond_dim_mults,
                    window_sizes1=unet_cfg.window_sizes1,
                    window_sizes2=unet_cfg.window_sizes2,
                    fourier_scale=unet_cfg.fourier_scale,
                    cfg=unet_cfg,
                    )
    else:
        raise NotImplementedError
    if model_cfg.model_type == 'const_sde':
        from denoising_diffusion_pytorch.ddm_const_sde import LatentDiffusion
    else:
        raise NotImplementedError(f'{model_cfg.model_type} is not supported!')
    ldm = LatentDiffusion(
        model=unet,
        auto_encoder=first_stage_model,
        train_sample=model_cfg.train_sample,
        image_size=model_cfg.image_size,
        timesteps=model_cfg.timesteps,
        sampling_timesteps=model_cfg.sampling_timesteps,
        loss_type=model_cfg.loss_type,
        objective=model_cfg.objective,
        scale_factor=model_cfg.scale_factor,
        scale_by_std=model_cfg.scale_by_std,
        scale_by_softsign=model_cfg.scale_by_softsign,
        default_scale=model_cfg.get('default_scale', False),
        input_keys=model_cfg.input_keys,
        ckpt_path=model_cfg.ckpt_path,
        ignore_keys=model_cfg.ignore_keys,
        only_model=model_cfg.only_model,
        start_dist=model_cfg.start_dist,
        perceptual_weight=model_cfg.perceptual_weight,
        use_l1=model_cfg.get('use_l1', True),
        cfg=model_cfg,
    )
    
    model_pose_decoder = pose_decoder().cuda()
   

    model = mainNetwork(
        diffusion_extractor=ldm,
        pose_network=model_pose_decoder,
    ).cuda()
   
 
    data_cfg = cfg.data

    if data_cfg['name'] == 'edge':
        dataset = EdgeDatasetTest(
            data_root=data_cfg.img_folder,
            image_size=model_cfg.image_size,
        )

    else:
        raise NotImplementedError

    dl = DataLoader(dataset, batch_size=cfg.sampler.batch_size, shuffle=False, pin_memory=True,
                    num_workers=data_cfg.get('num_workers', 2))

    sampler_cfg = cfg.sampler
    sampler = Sampler(
        model, dl, batch_size=sampler_cfg.batch_size,
        sample_num=sampler_cfg.sample_num,
        results_folder=sampler_cfg.save_folder, cfg=cfg,
    )
    sampler.sample()

    if data_cfg.name == 'cityscapes' or data_cfg.name == 'sr' or data_cfg.name == 'edge':
        exit()
    else:
        assert len(os.listdir(sampler_cfg.target_path)) > 0, "{} have no image !".format(sampler_cfg.target_path)
        sampler.cal_fid(target_path=sampler_cfg.target_path)
    pass

class Sampler(object):
    def __init__(
            self,
            model,
            data_loader,
            sample_num=1000,
            batch_size=16,
            results_folder='./results',
            cfg={},
    ):
        super().__init__()
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision='no',
            kwargs_handlers=[ddp_handler],
        )
        self.model = model
        self.sample_num = sample_num

        self.batch_size = batch_size
        self.batch_num = math.ceil(sample_num // batch_size)

        self.image_size = model.diffusion_extractor.image_size
        self.cfg = cfg

        dl = self.accelerator.prepare(data_loader)
        self.dl = dl
        self.results_folder = Path(results_folder)
        if self.accelerator.is_main_process:
            self.results_folder.mkdir(exist_ok=True, parents=True)

        self.model = self.accelerator.prepare(self.model)
        data = torch.load(cfg.sampler.ckpt_path, map_location=lambda storage, loc: storage)

        model = self.accelerator.unwrap_model(self.model)
        if cfg.sampler.use_ema:
            sd = data['ema']
            new_sd = {}
            for k in sd.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]  # remove ema_model.
                    new_sd[new_k] = sd[k]
            sd = new_sd
            model.load_state_dict(sd)
        else:
            state_dict = data['model']
            for key in list(state_dict.keys()):
                if key not in model.state_dict():
                    del state_dict[key]
            model.load_state_dict(state_dict)

        if 'scale_factor' in data['model']:
            model.scale_factor = data['model']['scale_factor']

    def denormalize(self, tensor):
        return tensor * 0.5 + 0.5
    
    def sample(self):
        accelerator = self.accelerator
        device = accelerator.device
#        print('device',device)

        batch_num = self.batch_num
        
        with torch.no_grad():
            self.model.eval()
            for idx, batch in tqdm(enumerate(self.dl)):
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                cond = batch['cond']
                # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
                # gt = batch['gt']  
                # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

                raw_w = batch["raw_size"][0].item()
                raw_h = batch["raw_size"][1].item()
                img_name = batch["img_name"][0]

                mask = batch['ori_mask'] if 'ori_mask' in batch else None
                if self.cfg.sampler.sample_type == 'whole':
                    batch_pred = self.whole_sample(cond, raw_size=(raw_h, raw_w), mask=mask)
                elif self.cfg.sampler.sample_type == 'slide':
                    batch_pred, quaternionlist = self.slide_sample(cond, crop_size=self.cfg.sampler.get('crop_size', [320, 320]), stride=self.cfg.sampler.stride, mask=mask)
                    quaternion=quaternionlist[0].detach().clone().cpu().numpy()
                    rotation2=quaternion2rot(quaternion)
                    print(rotation2)
                    

                    
                else:
                    raise NotImplementedError

             
                for img, input_img in zip(batch_pred, cond):
                    input_img_denormalized = self.denormalize(input_img.cpu())

                    input_img_pil = T.ToPILImage()(input_img_denormalized.squeeze(0))
                    img_pil = T.ToPILImage()(img.squeeze(0))

                    combined_img = Image.new('RGB', (input_img_pil.width + img_pil.width, input_img_pil.height))
                    combined_img.paste(input_img_pil, (0, 0))
                    combined_img.paste(img_pil, (input_img_pil.width, 0))

                    file_name = self.results_folder / img_name
                    combined_img.save(str(file_name)[:-4] + ".png")
                    
                    
            #print(self.cfg.sampler.ckpt_path) 
            #print('num_corrects',num_corrects,num_corrects/len(self.dl))        
        # accelerator.print('sampling complete')

# 保存GT！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
#                 for img, input_img, gt_img in zip(batch_pred, cond, gt):  # 添加gt_img
#                     input_img_denormalized = self.denormalize(input_img.cpu())
#                     gt_img_denormalized = self.denormalize(gt_img.cpu())
#
#                     input_img_pil = T.ToPILImage()(input_img_denormalized.squeeze(0))
#                     img_pil = T.ToPILImage()(img.squeeze(0))
#                     gt_img_pil = T.ToPILImage()(gt_img_denormalized.squeeze(0))
#
#                     combined_img = Image.new('RGB', (input_img_pil.width + img_pil.width + gt_img_pil.width, input_img_pil.height))
#                     combined_img.paste(input_img_pil, (0, 0))
#                     combined_img.paste(img_pil, (input_img_pil.width, 0))
#                     combined_img.paste(gt_img_pil, (input_img_pil.width + img_pil.width, 0))
#
#                     file_name = self.results_folder / img_name
#                     combined_img.save(str(file_name)[:-4] + ".png")
#         accelerator.print('sampling complete')




    # 只保存输入
    # def sample(self):
    #     accelerator = self.accelerator
    #     device = accelerator.device
    #     batch_num = self.batch_num
    #     with torch.no_grad():
    #         self.model.eval()
    #         for idx, batch in tqdm(enumerate(self.dl)):
    #             for key in batch.keys():
    #                 if isinstance(batch[key], torch.Tensor):
    #                     batch[key] = batch[key].to(device)
    #             cond = batch['cond']
    #             raw_w = batch["raw_size"][0].item()
    #             raw_h = batch["raw_size"][1].item()
    #             img_name = batch["img_name"][0]
    #
    #             mask = batch['ori_mask'] if 'ori_mask' in batch else None
    #             if self.cfg.sampler.sample_type == 'whole':
    #                 batch_pred = self.whole_sample(cond, raw_size=(raw_h, raw_w), mask=mask)
    #             elif self.cfg.sampler.sample_type == 'slide':
    #                 batch_pred = self.slide_sample(cond, crop_size=self.cfg.sampler.get('crop_size', [320, 320]),
    #                                                stride=self.cfg.sampler.stride, mask=mask)
    #             else:
    #                 raise NotImplementedError
    #
    #             for img in batch_pred:
    #                 img_pil = T.ToPILImage()(img.squeeze(0))
    #                 file_name = self.results_folder / img_name
    #                 img_pil.save(str(file_name)[:-4] + ".png")
    #     accelerator.print('sampling complete')

    def slide_sample(self, inputs, crop_size, stride, mask=None):
#        print("in slide_sample")
        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = 3
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        aux_out1 = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]

                if isinstance(self.model, nn.parallel.DistributedDataParallel):
                    crop_seg_logit = self.model.module.sample(batch_size=1, cond=crop_img, mask=mask)
                    aux_out = None
                elif isinstance(self.model, nn.Module):
                    #crop_seg_logit, quaternionlist = self.model.sample(batch_size=1, cond=crop_img, mask=mask)
                    crop_seg_logit, quaternionlist = self.model.sample(crop_img)
                    aux_out = None
                else:
                    raise NotImplementedError

                if crop_seg_logit.shape[1] == 1:
                    crop_seg_logit = crop_seg_logit.repeat(1, 3, 1, 1)

                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                if aux_out is not None:
                    aux_out1 += F.pad(aux_out,
                                   (int(x1), int(aux_out1.shape[3] - x2), int(y1),
                                    int(aux_out1.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat
        aux_out1 = aux_out1 / count_mat
        if aux_out is not None:
            return seg_logits, aux_out1
        return seg_logits, quaternionlist

    def whole_sample(self, inputs, raw_size, mask=None):
        inputs = F.interpolate(inputs, size=(416, 416), mode='bilinear', align_corners=True)
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            seg_logits = self.model.module.sample(batch_size=inputs.shape[0], cond=inputs, mask=mask)
        elif isinstance(self.model, nn.Module):
            seg_logits = self.model.sample(batch_size=inputs.shape[0], cond=inputs, mask=mask)
        seg_logits = F.interpolate(seg_logits, size=raw_size, mode='bilinear', align_corners=True)
        return seg_logits

    def cal_fid(self, target_path):
        command = 'fidelity -g 0 -f -i -b {} --input1 {} --input2 {}'.format(self.batch_size, str(self.results_folder), target_path)
        os.system(command)

def to_flattened_numpy(x):
    return x.detach().cpu().numpy().reshape((-1,))

def from_flattened_numpy(x, shape):
    return torch.from_numpy(x.reshape(shape))

def get_ode_sampler(rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
  def denoise_update_fn(model, x):
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    _, x = predictor_obj.update_fn(x, vec_eps)
    return x

  def drift_fn(model, x, t, model_type='const'):
    pred = model(x, t)
    if model_type == 'const':
        drift = pred
    elif model_type == 'linear':
        K, C = pred.chunk(2, dim=1)
        drift = K * t + C
    return drift

  def ode_sampler(model, shape):
    with torch.no_grad():
      x = torch.randn(*shape)
      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t * 1000
        drift = drift_fn(model, x, vec_t)
        return to_flattened_numpy(drift)

      solution = integrate.solve_ivp(ode_func, (1, eps), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
      return x, nfe

  return ode_sampler

if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass
