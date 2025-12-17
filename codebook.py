# import yaml
# import argparse
# import torch
# import numpy as np
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from PIL import Image
# from tqdm import tqdm
# import os
# from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
# from sklearn.metrics.pairwise import cosine_similarity
#
# # 解析配置文件
# def parse_args():
#     parser = argparse.ArgumentParser(description="codebook generation and matching")
#     parser.add_argument("--cfg", help="experiment configure file name", type=str, required=True)
#     args = parser.parse_args()
#     with open(args.cfg, 'r') as f:
#         cfg = yaml.safe_load(f)
#     return cfg
#
# # 自定义数据集类
# class CustomDataset(Dataset):
#     def __init__(self, image_paths, rotation_folder, transform=None):
#         self.image_paths = image_paths
#         self.rotation_folder = rotation_folder
#         self.transform = transform
#         print(f"Total images in dataset: {len(self.image_paths)}")  # 调试信息
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         image_name = os.path.basename(image_path)
#         rotation_path = os.path.join(self.rotation_folder, image_name.replace('.png', '.txt').replace('.jpg', '.txt'))
#
#         image = Image.open(image_path).convert('RGB')
#         rotation = np.loadtxt(rotation_path)
#
#         if self.transform:
#             image = self.transform(image)
#
#         return {'image': image, 'rotation': rotation, 'filename': image_name}
#
# # 加载自编码器模型
# def load_autoencoder(cfg):
#     model_cfg = cfg['model']
#     autoencoder = AutoencoderKL(
#         ddconfig=model_cfg['ddconfig'],
#         lossconfig=model_cfg['lossconfig'],
#         embed_dim=model_cfg['embed_dim'],
#         ckpt_path=model_cfg.get('ckpt_path', None)
#     )
#     return autoencoder
#
# # 创建codebook
# def create_codebook(encoder, dataset, device, batch_size=32):
#     encoder.eval()
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#     embeddings = []
#     rotation_matrices = []
#     filenames = []
#
#     with torch.no_grad():
#         for data in tqdm(dataloader):
#             images = data['image'].to(device)
#             rotations = data['rotation'].numpy()
#             z = encoder(images)
#             z = z.view(z.size(0), -1).cpu().numpy()  # 展平为二维数组
#             embeddings.append(z)
#             rotation_matrices.append(rotations)
#             filenames.extend(data['filename'])
#
#     embeddings = np.vstack(embeddings)
#     rotation_matrices = np.vstack(rotation_matrices)
#     return embeddings, rotation_matrices, filenames
#
# # 保存codebook
# def save_codebook(filepath, embeddings, rotation_matrices, filenames):
#     np.savez(filepath, embeddings=embeddings, rotations=rotation_matrices, filenames=filenames)
#
# # 加载codebook
# def load_codebook(filepath):
#     data = np.load(filepath, allow_pickle=True)
#     return data['embeddings'], data['rotations'], data['filenames']
#
# # 匹配输入图像与codebook
# def match_image_to_codebook(encoder, image, codebook_embeddings, device, top_n=1):
#     encoder.eval()
#     with torch.no_grad():
#         z = encoder(image.to(device))
#         z = z.view(1, -1).cpu().numpy()  # 展平为二维数组
#     similarities = cosine_similarity(z, codebook_embeddings)
#     top_n_indices = np.argsort(similarities.squeeze())[::-1][:top_n]
#     return top_n_indices
#
# if __name__ == "__main__":
#     # 解析配置文件
#     cfg = parse_args()
#
#     # 获取图像文件夹和旋转矩阵文件夹路径
#     image_folder = cfg['data']['img_folder']
#     rotation_folder = cfg['data']['rotation_folder']
#
#     # 获取图像路径
#     image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith(('.png', '.jpg'))]
#     print(f"Total image paths: {len(image_paths)}")  # 调试信息
#
#     # 检查codebook文件是否存在
#     codebook_path = '/home/data/chenhongyao/codebook/output/codebook_flattened_1313.npz'
#     codebook_exists = os.path.exists(codebook_path)
#
#     # 加载和初始化自编码器
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     autoencoder = load_autoencoder(cfg)
#     autoencoder.to(device)
#     encoder = autoencoder.encoder
#
#     if codebook_exists:
#         # 加载codebook
#         codebook_embeddings, codebook_rotations, codebook_filenames = load_codebook(codebook_path)
#         print(f"Loaded codebook from {codebook_path}")
#     else:
#         # 定义图像预处理变换
#         transform = transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#
#         # 创建自定义数据集
#         dataset = CustomDataset(image_paths, rotation_folder, transform=transform)
#
#         # 生成codebook
#         embeddings, rotation_matrices, filenames = create_codebook(encoder, dataset, device, batch_size=cfg['data']['batch_size'])
#
#         # 保存codebook
#         save_codebook(codebook_path, embeddings, rotation_matrices, filenames)
#         print(f"Saved codebook to {codebook_path}")
#
#         # 设置codebook变量
#         codebook_embeddings, codebook_rotations, codebook_filenames = embeddings, rotation_matrices, filenames
#
#     # 示例：匹配输入图像与codebook
#     test_image_path = image_paths[0]  # 使用第一个图像作为测试
#     test_image_name = os.path.basename(test_image_path)  # 获取输入图像的文件名
#     print(f"Input image filename: {test_image_name}")  # 打印输入图像文件名
#
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#     test_image = transform(Image.open(test_image_path).convert('RGB')).unsqueeze(0)
#     top_indices = match_image_to_codebook(encoder, test_image, codebook_embeddings, device, top_n=5)
#
#     print(f"Top 5 matched rotation matrices indices: {top_indices}")
#     print(f"Matched rotation matrices: {codebook_rotations[top_indices]}")
#     print(f"Matched filenames: {codebook_filenames[top_indices]}")
#
#     matched_rotations = codebook_rotations[top_indices]
#     matched_filenames = codebook_filenames[top_indices]
#
#     for idx, matched_filename in enumerate(matched_filenames):
#         print(f"Matched image {idx+1}: {matched_filename}, Rotation matrix: {matched_rotations[idx]}")











import yaml
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
from sklearn.metrics.pairwise import cosine_similarity

# 解析配置文件
def parse_args():
    parser = argparse.ArgumentParser(description="codebook generation and matching")
    parser.add_argument("--cfg", help="experiment configure file name", type=str, required=True)
    args = parser.parse_args()
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

# 自定义函数来解析旋转矩阵和位移向量
def parse_rotation_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    rotation = None
    translation = None
    for line in lines:
        if line.startswith('cam_R_m2c:'):
            rotation = np.array([float(x) for x in line.split('[')[1].strip(']\n').split(',')])
        elif line.startswith('cam_t_m2c:'):
            translation = np.array([float(x) for x in line.split('[')[1].strip(']\n').split(',')])
    return rotation, translation

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_paths, rotation_folder, transform=None):
        self.image_paths = image_paths
        self.rotation_folder = rotation_folder
        self.transform = transform
        print(f"Total images in dataset: {len(self.image_paths)}")  # 调试信息

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path)
        rotation_path = os.path.join(self.rotation_folder, image_name.replace('.png', '.txt').replace('.jpg', '.txt'))

        image = Image.open(image_path).convert('RGB')
        rotation, translation = parse_rotation_file(rotation_path)

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'rotation': np.hstack((rotation, translation)), 'filename': image_name}

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

# 创建codebook
def create_codebook(encoder, dataset, device, batch_size=32):
    encoder.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    rotation_matrices = []
    filenames = []

    with torch.no_grad():
        for data in tqdm(dataloader):
            images = data['image'].to(device)
            rotations = data['rotation'].numpy()
            z = encoder(images)
            z = z.view(z.size(0), -1).cpu().numpy()  # 展平为二维数组
            embeddings.append(z)
            rotation_matrices.append(rotations)
            filenames.extend(data['filename'])

    embeddings = np.vstack(embeddings)
    rotation_matrices = np.vstack(rotation_matrices)
    return embeddings, rotation_matrices, filenames

# 保存codebook
def save_codebook(filepath, embeddings, rotation_matrices, filenames):
    np.savez(filepath, embeddings=embeddings, rotations=rotation_matrices, filenames=filenames)

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

if __name__ == "__main__":
    # 解析配置文件
    cfg = parse_args()

    # 获取图像文件夹和旋转矩阵文件夹路径
    image_folder = cfg['data']['img_folder']
    rotation_folder = cfg['data']['rotation_folder']

    # 获取图像路径
    image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith(('.png', '.jpg'))]
    print(f"Total image paths: {len(image_paths)}")  # 调试信息

    # 检查codebook文件是否存在
    codebook_path = '/home/data/chenhongyao/codebook/output/codebook_flattened_1313.npz'
    codebook_exists = os.path.exists(codebook_path)

    # 加载和初始化自编码器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder = load_autoencoder(cfg)
    autoencoder.to(device)
    encoder = autoencoder.encoder

    if codebook_exists:
        # 加载codebook
        codebook_embeddings, codebook_rotations, codebook_filenames = load_codebook(codebook_path)
        print(f"Loaded codebook from {codebook_path}")
    else:
        # 定义图像预处理变换
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # 创建自定义数据集
        dataset = CustomDataset(image_paths, rotation_folder, transform=transform)

        # 生成codebook
        embeddings, rotation_matrices, filenames = create_codebook(encoder, dataset, device, batch_size=cfg['data']['batch_size'])

        # 保存codebook
        save_codebook(codebook_path, embeddings, rotation_matrices, filenames)
        print(f"Saved codebook to {codebook_path}")

        # 设置codebook变量
        codebook_embeddings, codebook_rotations, codebook_filenames = embeddings, rotation_matrices, filenames

    # 示例：匹配输入图像与codebook
    test_image_path = image_paths[0]  # 使用第一个图像作为测试
    test_image_name = os.path.basename(test_image_path)  # 获取输入图像的文件名
    print(f"Input image filename: {test_image_name}")  # 打印输入图像文件名

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_image = transform(Image.open(test_image_path).convert('RGB')).unsqueeze(0)
    top_indices = match_image_to_codebook(encoder, test_image, codebook_embeddings, device, top_n=5)

    top_indices = top_indices.astype(int)  # 将 top_indices 转换为整数数组

    print(f"Top 5 matched rotation matrices indices: {top_indices}")
    print(f"Matched rotation matrices: {codebook_rotations[top_indices]}")
    print(f"Matched filenames: {codebook_filenames[top_indices]}")

    matched_rotations = codebook_rotations[top_indices]
    matched_filenames = codebook_filenames[top_indices]

    for idx, matched_filename in enumerate(matched_filenames):
        print(f"Matched image {idx+1}: {matched_filename}, Rotation matrix: {matched_rotations[idx]}")



