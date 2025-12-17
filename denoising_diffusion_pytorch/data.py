import torch
import torchvision.transforms as T
import torch.utils.data as data
import torch.nn as nn
from pathlib import Path
from functools import partial
from denoising_diffusion_pytorch.utils import exists, convert_image_to_fn, normalize_to_neg_one_to_one
from PIL import Image, ImageDraw
import torch.nn.functional as F
import math
import torchvision.transforms.functional as F2
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from typing import Any, Callable, Optional, Tuple
import os
import pickle
import numpy as np
import copy
import albumentations
from torchvision.transforms.functional import InterpolationMode

from torchvision.utils import save_image

from torch.utils.data import Dataset



# 1111111111111111111111111111111111
def exists(val):
    return val is not None

class Identity:
    def __call__(self, x):
        return x
# 111111111111111111111111111111111111

def get_imgs_list(imgs_dir):
    imgs_list = os.listdir(imgs_dir)
    imgs_list.sort()
    return [os.path.join(imgs_dir, f) for f in imgs_list if f.endswith('.jpg') or f.endswith('.JPG')or f.endswith('.png') or f.endswith('.pgm') or f.endswith('.ppm')]


def fit_img_postfix(img_path):
    if not os.path.exists(img_path) and img_path.endswith(".jpg"):
        img_path = img_path[:-4] + ".png"
    if not os.path.exists(img_path) and img_path.endswith(".png"):
        img_path = img_path[:-4] + ".jpg"
    return img_path




class EdgeDataset(data.Dataset):
    def __init__(
        self,
        data_root,
        # mask_folder,
        image_size,
        exts = ['png', 'jpg'],
        augment_horizontal_flip = True,
        convert_image_to = None,
        normalize_to_neg_one_to_one=True,
        split='train',
        # inter_type='bicubic',
        # down=4,
        threshold=0.3, use_uncertainty=False, cfg={}
    ):
        super().__init__()
        # self.img_folder = Path(img_folder)
        # self.edge_folder = Path(os.path.join(data_root, f'gt_imgs'))
        # self.img_folder = Path(os.path.join(data_root, f'imgs'))
        # self.edge_folder = Path(os.path.join(data_root, "edge", "aug"))
        # self.img_folder = Path(os.path.join(data_root, "image", "aug"))
        self.data_root = data_root
        self.image_size = image_size
        self.exts = exts


        # self.edge_paths = [p for ext in exts for p in self.edge_folder.rglob(f'*.{ext}')]
        # self.img_paths = [(self.img_folder / item.parent.name / f'{item.stem}.jpg') for item in self.edge_paths]
        # self.img_paths = [(self.img_folder / f'{item.stem}.jpg') for item in self.edge_paths]

        self.threshold = threshold * 255
        self.use_uncertainty = use_uncertainty
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else Identity()

        self.data_list = self.build_list()

        # self.transform = Compose([
        #     Resize(image_size),
        #     RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
        #     ToTensor()
        # ])

        self.transform = Compose([
            RandomCrop(image_size),
            RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
            ToTensor()  # 确保正确转换为Tensor
        ])

        crop_type = cfg.get('crop_type') if 'crop_type' in cfg else 'rand_crop'
        if crop_type == 'rand_crop':
            self.transform = Compose([
                RandomCrop(image_size),
                RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
                ToTensor()
            ])
        elif crop_type == 'rand_resize_crop':
            self.transform = Compose([
                RandomResizeCrop(image_size),
                RandomHorizontalFlip() if augment_horizontal_flip else Identity(),
                ToTensor()
            ])

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.debug_dir = "/home/chenhongyao/DiffusionPose-main/output/test/"
        os.makedirs(self.debug_dir, exist_ok=True)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        print("crop_type:", crop_type)


    def __len__(self):
        return len(self.data_list)


    def read_img(self, image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        raw_width, raw_height = img.size


        return img, (raw_width, raw_height)

    def read_lb(self, lb_path):
        # lb_data = Image.open(lb_path).convert('L')
        lb_data = Image.open(lb_path)  # Remove .convert('L') to keep 3 channels
        lb = np.array(lb_data).astype(np.float32)

        threshold = self.threshold


        lb[lb >= threshold] = 255
        lb[lb < threshold] = 0
        lb = Image.fromarray(lb.astype(np.uint8))
        return lb
    
    def build_list(self):
        data_root = os.path.abspath(self.data_root)
        images_path = os.path.join(data_root, 'image')
        labels_path = os.path.join(data_root, 'clean')
        poses_path = os.path.join(data_root, 'RT')

        print(f"Images path: {images_path}")  # 调试信息
        print(f"Labels path: {labels_path}")  # 调试信息
        print(f"Poses path: {poses_path}")  # 调试信息

        if not os.path.exists(images_path):
            raise FileNotFoundError(f"Images path does not exist: {images_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels path does not exist: {labels_path}")
        if not os.path.exists(poses_path):
            raise FileNotFoundError(f"Poses path does not exist: {poses_path}")    

        samples = []

        for ext in self.exts:
            image_files = sorted([f for f in os.listdir(images_path) if f.endswith(ext)])
            label_files = sorted([f for f in os.listdir(labels_path) if f.endswith(ext)])
            pose_files = sorted([f for f in os.listdir(poses_path)])
            
#            print('image_files',len(image_files))
#            print('label_files',len(label_files))
            assert len(image_files) == len(label_files), "Mismatch between number of images and labels"
            if (len(image_files) !=0):
                assert len(image_files) == len(pose_files), "Mismatch between number of images and labels"
            
                for image_file, label_file, pose_file in zip(image_files, label_files, pose_files):
                    image_path = os.path.join(images_path, image_file)
                    lb_path = os.path.join(labels_path, label_file)
                    pose_path=os.path.join(poses_path, pose_file)
                    samples.append((image_path, lb_path,pose_path))
        #print(samples[0])        
        return samples
    '''
    def build_list(self):
        data_root = os.path.abspath(self.data_root)
        samples = []

        # 指定的两个文件夹名称
        specified_folders = ['camera','iron','phone']
        #['ape', 'cat']

        # 获取 data_root 下的所有子目录
        subdirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

        for subdir in subdirs:
            # 检查当前子目录是否是我们指定的两个文件夹之一
            if subdir in specified_folders:
                subdir_path = os.path.join(data_root, subdir)
                images_path = os.path.join(subdir_path, 'image')
                labels_path = os.path.join(subdir_path, 'edge')  # 修改edge为cube
                poses_path = os.path.join(subdir_path, 'RT')
                
                print(f"Images path: {images_path}")  # 调试信息
                print(f"Labels path: {labels_path}")  # 调试信息
                print(f"Poses path: {poses_path}")  # 调试信息
                
                
                # 检查图像和标签文件夹是否存在
                if not os.path.exists(images_path) or not os.path.exists(labels_path) or not os.path.exists(poses_path):
                    continue  # 如果路径不存在，跳过该子目录
                for ext in self.exts:
                    image_files = sorted([f for f in os.listdir(images_path) if f.endswith(ext)])
                    label_files = sorted([f for f in os.listdir(labels_path) if f.endswith(ext)])
                    pose_files = sorted([f for f in os.listdir(poses_path)])
                    # 确保图像和标签文件数量相同
                    assert len(image_files) == len(label_files), f"Mismatch between number of images and labels in {subdir}"
                    
                    if (len(image_files) !=0):
                        assert len(image_files) == len(pose_files), "Mismatch between number of images and labels"

                    for image_file, label_file, pose_file in zip(image_files, label_files, pose_files):
                        image_path = os.path.join(images_path, image_file)
                        lb_path = os.path.join(labels_path, label_file)
                        pose_path=os.path.join(poses_path, pose_file)
                        samples.append((image_path, lb_path,pose_path))

        return samples
    '''
    def __getitem__(self, index):
        # img and edge path
        img_path, edge_path, pose_path = self.data_list[index]
        pose=np.loadtxt(pose_path)
        pose=torch.tensor(pose, dtype=torch.float32)
        img_name = os.path.basename(img_path)
        img, raw_size = self.read_img(img_path)
        # !!!!!!!!!!
        #edge = self.read_lb(edge_path)
        edge, raw_size0 = self.read_img(edge_path)
        # !!!!!!!!!!!!!!!!!!!!!!

# ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # 保存原始图像以检查读取的正确性
        #img.save(os.path.join(self.debug_dir, f'raw_img_{img_name}'))
        #edge.save(os.path.join(self.debug_dir, f'raw_edge_{img_name}'))

        # 逐步检查和保存每一步变换后的图像
        img_transformed = img
        edge_transformed = edge

        # RandomCrop裁剪
        img_transformed, edge_transformed = RandomCrop(self.image_size)(img_transformed, edge_transformed)
        #img_transformed.save(os.path.join(self.debug_dir, f'crop_img_{img_name}'))
        #edge_transformed.save(os.path.join(self.debug_dir, f'crop_edge_{img_name}'))

        # RandomHorizontalFlip旋转50%
        #img_transformed, edge_transformed = RandomHorizontalFlip()(img_transformed, edge_transformed)
        #img_transformed.save(os.path.join(self.debug_dir, f'flip_img_{img_name}'))
        #edge_transformed.save(os.path.join(self.debug_dir, f'flip_edge_{img_name}'))

        # ToTensor转换数据类型
        img_transformed = ToTensor()(img_transformed)
        edge_transformed = ToTensor()(edge_transformed)
        # save_image(img_transformed, os.path.join(self.debug_dir, f'tensor_img_{img_name}.png'))
        # save_image(edge_transformed, os.path.join(self.debug_dir, f'tensor_edge_{img_name}.png'))

        # 检查值范围和数据类型
        # print(
        #     f'img_transformed range: {img_transformed.min()} - {img_transformed.max()}, dtype: {img_transformed.dtype}')
        # print(
        #     f'edge_transformed range: {edge_transformed.min()} - {edge_transformed.max()}, dtype: {edge_transformed.dtype}')

        # 保存 transform 之后的图像以检查处理过程中的正确性
        # save_image(img_transformed, os.path.join(self.debug_dir, f'transformed_before_norm_img_{img_name}.png'))
        # save_image(edge_transformed, os.path.join(self.debug_dir, f'transformed_before_norm_edge_{img_name}.png'))

        # img_transformed, edge_transformed = self.transform(img, edge)
        if self.normalize_to_neg_one_to_one:  # 检查是否需要归一化
            img_transformed = self.apply_normalize_to_neg_one_to_one(img_transformed)
            edge_transformed = self.apply_normalize_to_neg_one_to_one(edge_transformed)



        # 保存转换后的图像以检查处理过程中的正确性
        #save_image(self.apply_denormalize_to_zero_to_one(img_transformed),
        #            os.path.join(self.debug_dir, f'transformed_img_{img_name}'))
        return {'image': edge_transformed, 'cond': img_transformed, 'img_name': img_name , 'pose': pose}

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # 重写归一化和反归一化函数
    def apply_normalize_to_neg_one_to_one(self, tensor):
        return tensor * 2 - 1  # Normalize tensor values from [0, 1] to [-1, 1]


    def apply_denormalize_to_zero_to_one(self, tensor):
        return (tensor + 1) / 2  # Denormalize tensor values from [-1, 1] to [0, 1]





# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 数据集类（test）,不包括GT

class EdgeDatasetTest(data.Dataset):
    def __init__(
        self,
        data_root,
        # mask_folder,
        image_size,
        exts = ['png', 'jpg'],
        convert_image_to = None,
        normalize_to_neg_one_to_one=True,
    ):
        super().__init__()
        
        self.exts = exts
        self.data_root = data_root
        self.image_size = image_size
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else Identity()

        self.data_list = self.build_list()
        
        self.transform = Compose([
            ToTensor()
        ])
    


    def __len__(self):
        return len(self.data_list)


    def read_img(self, image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        raw_width, raw_height = img.size


        return img, (raw_width, raw_height)

    def read_lb(self, lb_path):
        lb_data = Image.open(lb_path).convert('L')
        lb = np.array(lb_data).astype(np.float32)

        threshold = self.threshold


        lb[lb >= threshold] = 255
        lb = Image.fromarray(lb.astype(np.uint8))
        return lb

    def build_list(self):
        data_root = os.path.abspath(self.data_root)
        # print(f"Checking directory: {data_root}")  # Debug line
        images_path = os.path.join(data_root, 'image')
        poses_path = os.path.join(data_root, 'RT')
#        samples = [os.path.join(images_path, file) for file in os.listdir(images_path) if file.endswith('.jpg')]  #.jpg linemod数据集用jpg
#
#        samples.sort()  # 确保图像按顺序排列
#        poses=
#        # print(f"Found files: {samples}")  # Debug line
#
#        return samples
#
#
#
#        samples.sort()  # 确保图像按顺序排列
#        return samples
    
        samples = []

        for ext in self.exts:
            image_files = sorted([f for f in os.listdir(images_path) if f.endswith(ext)])
            pose_files = sorted([f for f in os.listdir(poses_path)])
            
            if (len(image_files) !=0):
                assert len(image_files) == len(pose_files), "Mismatch between number of images and labels"
            
                for image_file, pose_file in zip(image_files, pose_files):
                    image_path = os.path.join(images_path, image_file)
                    pose_path=os.path.join(poses_path, pose_file)
                    samples.append((image_path, pose_path))
      
        return samples


    def __getitem__(self, index):
        img_path,pose_path = self.data_list[index]
        # edge_path = self.edge_paths[index]
        # img_path = self.img_paths[index]
        img_name = os.path.basename(img_path)

        img, raw_size = self.read_img(img_path)

        img = self.transform(img)
        if self.normalize_to_neg_one_to_one:   # transform to [-1, 1]
            img = normalize_to_neg_one_to_one(img)
        
        pose=np.loadtxt(pose_path)
        pose=torch.tensor(pose, dtype=torch.float32)



        return {'cond': img, 'raw_size': raw_size, 'img_name': img_name, 'pose': pose}
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# # 保存GT的数据集测试类
# class EdgeDatasetTest(Dataset):
#     def __init__(
#         self,
#         data_root,
#         gt_folder,  # 新增的groundtruth文件夹路径
#         image_size,
#         exts=['png', 'jpg'],
#         convert_image_to=None,
#         normalize_to_neg_one_to_one=True,
#     ):
#         super().__init__()
# 
#         self.data_root = data_root
#         self.gt_folder = gt_folder  # 初始化groundtruth文件夹路径
#         self.image_size = image_size
#         self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
# 
#         self.data_list = self.build_list()
# 
#         self.transform = Compose([
#             ToTensor()
#         ])
# 
#     def __len__(self):
#         return len(self.data_list)
# 
#     def read_img(self, image_path):
#         with open(image_path, 'rb') as f:
#             img = Image.open(f)
#             img = img.convert('RGB')
#         raw_width, raw_height = img.size
#         return img, (raw_width, raw_height)
# 
#     def build_list(self):
#         data_root = os.path.abspath(self.data_root)
#         images_path = data_root
#         samples = [os.path.join(images_path, file) for file in os.listdir(images_path) if file.endswith(('png', 'jpg'))]
#         samples.sort()  # 确保图像按顺序排列
#         return samples
# 
#     def __getitem__(self, index):
#         img_path = self.data_list[index]
#         img_name = os.path.basename(img_path)
#         gt_path = os.path.join(self.gt_folder, img_name)  # 构造groundtruth图像的路径
# 
#         img, raw_size = self.read_img(img_path)
#         gt_img, _ = self.read_img(gt_path)  # 读取groundtruth图像
# 
#         img = self.transform(img)
#         gt_img = self.transform(gt_img)  # 转换groundtruth图像
# 
#         if self.normalize_to_neg_one_to_one:  # 归一化到[-1, 1]
#             img = normalize_to_neg_one_to_one(img)
#             gt_img = normalize_to_neg_one_to_one(gt_img)
# 
#         return {'cond': img, 'raw_size': raw_size, 'img_name': img_name, 'gt': gt_img}  # 返回包含groundtruth的字典





class Identity(nn.Identity):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__(*args, **kwargs)

    def forward(self, input, target):
        return input, target

class Resize(T.Resize):
    def __init__(self, size, interpolation2=None, **kwargs):
        super().__init__(size, **kwargs)
        if interpolation2 is None:
            self.interpolation2 = self.interpolation
        else:
            self.interpolation2 = interpolation2

    def forward(self, img, target=None):
        if target is None:
            img = F2.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
            return img
        else:
            img = F2.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
            target = F2.resize(target, self.size, self.interpolation2, self.max_size, self.antialias)
            return img, target

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)

    def forward(self, img, target=None):
        if target is None:
            if torch.rand(1) < self.p:
                img = F2.hflip(img)
            return img
        else:
            if torch.rand(1) < self.p:
                img = F2.hflip(img)
                target = F2.hflip(target)
            return img, target

class CenterCrop(T.CenterCrop):
    def __init__(self, size):
        super().__init__(size)

    def forward(self, img, target=None):
        if target is None:
            img = F2.center_crop(img, self.size)
            return img
        else:
            img = F2.center_crop(img, self.size)
            target = F2.center_crop(target, self.size)
            return img, target

class RandomCrop(T.RandomCrop):
    def __init__(self, size, **kwargs):
        super().__init__(size, **kwargs)

    def single_forward(self, img, i, j, h, w):
        if self.padding is not None:
            img = F2.pad(img, self.padding, self.fill, self.padding_mode)
        width, height = F2.get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F2.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F2.pad(img, padding, self.fill, self.padding_mode)

        return F2.crop(img, i, j, h, w)

    def forward(self, img, target=None):
        i, j, h, w = self.get_params(img, self.size)
        if target is None:
            img = self.single_forward(img, i, j, h, w)
            return img
        else:
            img = self.single_forward(img, i, j, h, w)
            target = self.single_forward(target, i, j, h, w)
            return img, target

class RandomResizeCrop(T.RandomResizedCrop):
    def __init__(self, size, scale=(0.25, 1.0), **kwargs):
        super().__init__(size, scale, **kwargs)

    # def single_forward(self, img, i, j, h, w):
    #     if self.padding is not None:
    #         img = F2.pad(img, self.padding, self.fill, self.padding_mode)
    #     width, height = F2.get_image_size(img)
    #     # pad the width if needed
    #     if self.pad_if_needed and width < self.size[1]:
    #         padding = [self.size[1] - width, 0]
    #         img = F2.pad(img, padding, self.fill, self.padding_mode)
    #     # pad the height if needed
    #     if self.pad_if_needed and height < self.size[0]:
    #         padding = [0, self.size[0] - height]
    #         img = F2.pad(img, padding, self.fill, self.padding_mode)
    #
    #     return F2.crop(img, i, j, h, w)

    def single_forward(self, img, i, j, h, w, interpolation=InterpolationMode.BILINEAR):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        # i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F2.resized_crop(img, i, j, h, w, self.size, interpolation)

    def forward(self, img, target=None):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if target is None:
            img = self.single_forward(img, i, j, h, w)
            return img
        else:
            img = self.single_forward(img, i, j, h, w)
            target = self.single_forward(target, i, j, h, w, interpolation=InterpolationMode.NEAREST)
            return img, target

class ToTensor(T.ToTensor):
    def __init__(self):
        super().__init__()

    def __call__(self, img, target=None):
        if target is None:
            img = F2.to_tensor(img)
            return img
        else:
            img = F2.to_tensor(img)
            target = F2.to_tensor(target)
            return img, target

class Lambda(T.Lambda):
    """Apply a user-defined lambda as a transform. This transform does not support torchscript.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        super().__init__(lambd)

    def __call__(self, img, target=None):
        if target is None:
            return self.lambd(img)
        else:
            return self.lambd(img), self.lambd(target)

class Compose(T.Compose):
    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, img, target=None):
        if target is None:
            for t in self.transforms:
                img = t(img)
            return img
        else:
            for t in self.transforms:
                img, target = t(img, target)
            return img, target


# if __name__ == '__main__':
#     dataset = CIFAR10(
#         img_folder='/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/data/cifar-10-python',
#         augment_horizontal_flip=False
#     )
#     # dataset = CityscapesDataset(
#     #     # img_folder='/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/data/CelebAHQ/celeba_hq_256',
#     #     data_root='/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/data/Cityscapes/',
#     #     # data_root='/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/data/ADEChallengeData2016/',
#     #     image_size=[512, 1024],
#     #     exts = ['png'],
#     #     augment_horizontal_flip = False,
#     #     convert_image_to = None,
#     #     normalize_to_neg_one_to_one=True,
#     #     )
#     # dataset = SRDataset(
#     #     img_folder='/media/huang/ZX3 512G/data/DIV2K/DIV2K_train_HR',
#     #     image_size=[512, 512],
#     # )
#     # dataset = InpaintDataset(
#     #     img_folder='/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/data/CelebAHQ/celeba_hq_256',
#     #     image_size=[256, 256],
#     #     augment_horizontal_flip = True
#     # )
#     dataset = EdgeDataset(
#         data_root='/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/data/BSDS',
#         image_size=[256, 256],
#     )
#     for i in range(len(dataset)):
#         d = dataset[i]
#         mask = d['cond']
#         print(mask.max())
#     dl = data.DataLoader(dataset, batch_size=2, shuffle=False, pin_memory=True, num_workers=0)
#
#
#     dataset_builder = tfds.builder('cifar10')
#     split = 'train'
#     dataset_options = tf.data.Options()
#     dataset_options.experimental_optimization.map_parallelization = True
#     dataset_options.experimental_threading.private_threadpool_size = 48
#     dataset_options.experimental_threading.max_intra_op_parallelism = 1
#     read_config = tfds.ReadConfig(options=dataset_options)
#     dataset_builder.download_and_prepare()
#     ds = dataset_builder.as_dataset(
#         split=split, shuffle_files=True, read_config=read_config)
#     pause = 0





# if __name__ == "__main__":
#     dataset = EdgeDatasetTest(data_root='/home/data/chenhongyao/traindata/image', image_size=256)
#     print(f"Total images: {len(dataset)}")
#     for i in range(min(5, len(dataset))):
#         sample = dataset[i]
#         print(f"Sample {i}: {sample['img_name']}, shape: {sample['cond'].shape}, raw size: {sample['raw_size']}")
