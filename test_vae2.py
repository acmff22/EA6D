import yaml
import argparse
import torch
import torch.nn as nn
from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
import torchvision as tv
from pathlib import Path
import cv2

from diffusers.models import AutoencoderKL
from diffusers import DiffusionPipeline
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def parse_args():
    parser = argparse.ArgumentParser(description="Testing VAE with checkpoints")
    parser.add_argument("--cfg", help="experiment configure file name", type=str, required=True)
    return parser.parse_args()


def load_conf(config_file, conf={}):
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in exp_conf.items():
            conf[k] = v
    return conf


def load_model(cfg):
    model_cfg = cfg['model']
    model = AutoencoderKL(
        ddconfig=model_cfg['ddconfig'],
        lossconfig=model_cfg['lossconfig'],
        embed_dim=model_cfg['embed_dim'],
        ckpt_path=model_cfg['ckpt_path'],
    )

    # Load checkpoint
    checkpoint = torch.load(model_cfg['ckpt_path'])
    model.load_state_dict(checkpoint['model'])

    return model


def test_model(model, cfg, device):
    # Set the model to evaluation mode
    model.eval()
    model.to(device)

    data_cfg = cfg['data']

    # Load test images
    transform = tv.transforms.Compose([
        tv.transforms.Resize((256, 256)),  # 将 list 转换为 tuple
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.5], [0.5])  # 正则化以匹配训练数据
    ])

    test_dataset = tv.datasets.ImageFolder(data_cfg['img_folder'], transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=data_cfg['batch_size'], shuffle=False)

    output_folder = Path(cfg['trainer']['results_folder'])
    output_folder.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        for i, (img, _) in enumerate(test_loader):
            img = img.to(device)

            # Forward pass through the model to get reconstructed images
            #            recon_img = model.validate_img(img)
            latents = model.encode(img, return_dict=False)
            latents = latents[0].mean

            #            print('latents',latents)
            decodeOut = model.decode(latents, return_dict=False)

            decodeOut = decodeOut[0].permute(2, 3, 1, 0).squeeze()

            recon_img = decodeOut.cpu().detach().numpy()
            recon_img = recon_img * 255

            numpy_img = cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR)

            #            print(os.path.join(output_folder, "/","1.png"))
            filename = f'image_{i:03d}.png'
            file_path = os.path.join(output_folder, filename)
            print(file_path)
            cv2.imwrite(file_path, numpy_img)
            # Normalize images to range [0, 1]
            #            recon_img = torch.clamp((recon_img + 1.0) / 2.0, 0.0, 1.0)
            original_img = torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)
            #
            #            # Save the original and reconstructed images
            #            combined_img = torch.cat([original_img, recon_img], dim=3)  # Combine side-by-side
            tv.utils.save_image(original_img, output_folder / f"reconstructed_{i}.png", nrow=1)
            if i > 9:
                break


def main():
    # 通过命令行参数获取配置文件路径
    args = parse_args()
    cfg = load_conf(args.cfg)  # 使用传入的配置文件路径

    # Load model
    #    model = load_model(cfg)

    #    model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")

    #    checkpoint=torch.load('/home/data/chenhongyao/diffusuion_pose_models/step1/vae-ft-ema-560000-ema-pruned_coco.ckpt')
    #    model.load_state_dict(checkpoint['state_dict'])
    #    pipeline = DiffusionPipeline.from_pretrained("stabilityai/sd-vae-ft-ema")
    model = AutoencoderKL.from_pretrained("/home/data/chenhongyao/diffusuion_pose_models/step1/sd-vae-ft-ema")
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Perform testing
    test_model(model, cfg, device)


if __name__ == "__main__":
    main()
