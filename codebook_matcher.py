import yaml
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os
from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
import torch.backends.cudnn as cudnn

# 设置cuDNN
cudnn.benchmark = True
cudnn.enabled = True

# 解析配置文件
def parse_args():
    parser = argparse.ArgumentParser(description="codebook generation and matching")
    parser.add_argument("--cfg", help="experiment configure file name", type=str, required=True)
    args = parser.parse_args()
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        print(f"Total images in dataset: {len(self.image_paths)}")  # 调试信息

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path)

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'filename': image_name}

# 加载自编码器模型
def load_autoencoder(cfg):
    model_cfg = cfg['model']
    autoencoder = AutoencoderKL(
        ddconfig=model_cfg['ddconfig'],
        lossconfig=model_cfg['lossconfig'],
        embed_dim=model_cfg['embed_dim'],
        ckpt_path=model_cfg.get('ckpt_path', None)
    )
    return autoencoder

# 加载codebook
def load_codebook(filepath):
    data = np.load(filepath, allow_pickle=True)
    return data['embeddings'], data['rotations'], data['filenames']

# 匹配输入图像与codebook
def match_image_to_codebook(encoder, image, codebook_embeddings, device, top_n=1):
    encoder.eval()
    with torch.no_grad():
        z = encoder(image.to(device))
        z = z.view(1, -1).cpu().numpy()  # 展平为二维数组
    similarities = cosine_similarity(z, codebook_embeddings)
    top_n_indices = np.argsort(similarities.squeeze())[::-1][:top_n]
    return top_n_indices

# 保存图像拼接结果
def save_combined_image(output_folder, input_image, test_image_name, matched_image, bg_image):
    width, height = 256, 256
    combined_image = Image.new('RGB', (width * 3, height))

    # 确保所有图像的颜色模式一致
    input_image = input_image.convert('RGB')
    bg_image = bg_image.convert('RGB')
    matched_image = matched_image.convert('RGB')

    # 带背景的图像
    combined_image.paste(bg_image, (0, 0))

    # 输入图像
    combined_image.paste(input_image, (width, 0))

    # 最相似的图像
    combined_image.paste(matched_image, (2 * width, 0))

    # 仅使用输入图像的序号作为文件名
    file_index = os.path.splitext(test_image_name)[0]
    combined_image.save(os.path.join(output_folder, f"{file_index}.png"))

if __name__ == "__main__":
    # 解析配置文件
    cfg = parse_args()

    # 打印一些调试信息
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # 设置CUDA设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # 加载和初始化自编码器
    autoencoder = load_autoencoder(cfg)
    autoencoder.to(device)

    # 提取编码器部分
    encoder = autoencoder.encoder

    # 加载codebook
    codebook_path = cfg['data']['codebook_path']
    codebook_embeddings, codebook_rotations, codebook_filenames = load_codebook(codebook_path)

    # 获取测试图像文件夹路径
    test_image_folder = cfg['data']['test_img_folder']
    matching_image_folder = cfg['data']['matching_img_folder']
    bg_image_folder = cfg['data']['bg_img_folder']
    output_folder = cfg['data']['output_folder']

    test_image_paths = [os.path.join(test_image_folder, fname) for fname in os.listdir(test_image_folder) if fname.endswith('.png')]
    print(f"Total test image paths: {len(test_image_paths)}")  # 调试信息

    # 定义图像预处理变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 保持标准化处理
    ])

    # 创建自定义数据集
    test_dataset = CustomDataset(test_image_paths, transform=transform)

    # 创建DataLoader
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 进行图像匹配
    for data in test_dataloader:
        test_image = data['image']
        test_image_name = data['filename'][0]
        top_indices = match_image_to_codebook(encoder, test_image, codebook_embeddings, device, top_n=1)

        print(f"Input image filename: {test_image_name}")
        print(f"Top matched rotation matrix index: {top_indices[0]}")
        print(f"Matched rotation matrix: {codebook_rotations[top_indices[0]]}")
        print(f"Matched filename: {codebook_filenames[top_indices[0]]}")

        # 获取最相似的图像
        matched_image = Image.open(os.path.join(matching_image_folder, codebook_filenames[top_indices[0]].replace('.png', '.jpg'))).resize((256, 256))

        # 获取输入图像和带背景的图像
        input_image = transforms.ToPILImage()(test_image.squeeze())
        bg_image_path = os.path.join(bg_image_folder, test_image_name.replace('.png', '.jpg'))
        bg_image = Image.open(bg_image_path).resize((256, 256))

        # 将标准化的图像转换回原始范围
        input_image = transforms.ToPILImage()((test_image.squeeze() * 0.5 + 0.5).clamp(0, 1))

        # 保存拼接图像
        save_combined_image(output_folder, input_image, test_image_name, matched_image, bg_image)
