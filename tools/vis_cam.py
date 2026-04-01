import os
import sys

sys.path.insert(0, os.getcwd())
import argparse
import copy
import math
import pkg_resources
import re
from pathlib import Path
import torch
from PIL import Image, ImageOps
import cv2
import numpy as np
import datetime

# 添加PDF相关导入
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 保持高DPI设置以确保清晰度
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300

# 紧凑设置，删除所有标题和标签
matplotlib.rcParams['figure.subplot.left'] = 0.0
matplotlib.rcParams['figure.subplot.right'] = 1.0
matplotlib.rcParams['figure.subplot.bottom'] = 0.0
matplotlib.rcParams['figure.subplot.top'] = 1.0

from models.build import BuildNet
from utils.version_utils import digit_version
from utils.train_utils import file2dict
from utils.misc import to_2tuple
from utils.inference import init_model
from core.datasets.compose import Compose
from torch.nn import BatchNorm1d, BatchNorm2d, GroupNorm, LayerNorm

try:
    from pytorch_grad_cam import (EigenCAM, EigenGradCAM, GradCAM,
                                  GradCAMPlusPlus, LayerCAM, XGradCAM)
    from pytorch_grad_cam.activations_and_gradients import \
        ActivationsAndGradients
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    raise ImportError('Please run `pip install "grad-cam>=1.3.6"` to install '
                      '3rd party package pytorch_grad_cam.')

# set of transforms, which just change data format, not change the pictures
FORMAT_TRANSFORMS_SET = {'ToTensor', 'Normalize', 'ImageToTensor', 'Collect'}

# Supported grad-cam type map
METHOD_MAP = {
    'gradcam': GradCAM,
    'gradcam++': GradCAMPlusPlus,
    'xgradcam': XGradCAM,
    'eigencam': EigenCAM,
    'eigengradcam': EigenGradCAM,
    'layercam': LayerCAM,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CAM for multiple categories')
    parser.add_argument('--alpha', type=Path, required=True, help='Alpha category image')
    parser.add_argument('--delta', type=Path, required=True, help='Delta category image')
    parser.add_argument('--omicron', type=Path, required=True, help='Omicron category image')
    parser.add_argument('--else-cat', type=Path, required=True, help='Else category image')
    parser.add_argument('config', help='Config file')
    parser.add_argument(
        '--target-layers',
        default=[],
        nargs='+',
        type=str,
        help='The target layers to get CAM, if not set, the tool will '
             'specify the norm layer in the last block.')
    parser.add_argument(
        '--preview-model',
        default=False,
        action='store_true',
        help='To preview all the model layers')
    parser.add_argument(
        '--method',
        default='GradCAM',
        help='Type of method to use, supports '
             f'{", ".join(list(METHOD_MAP.keys()))}.')
    parser.add_argument(
        '--eigen-smooth',
        default=False,
        action='store_true',
        help='Reduce noise by taking the first principle componenet of '
             '``cam_weights*activations``')
    parser.add_argument(
        '--aug-smooth',
        default=False,
        action='store_true',
        help='Wether to use test time augmentation, default not to use')
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='The directory to save individual PDF files.')
    parser.add_argument(
        '--dpi',
        type=int,
        default=600,
        help='DPI for PDF output, default is 600')
    parser.add_argument(
        '--crop-pixels',
        type=int,
        default=100,
        help='Number of pixels to crop from each side for original and overlay, default is 100')
    parser.add_argument(
        '--target-size',
        type=int,
        default=1000,
        help='Target size for all output images (will be resized to target_size x target_size), default is 1000')
    parser.add_argument(
        '--detail-size',
        type=int,
        default=300,
        help='Original detail size before resizing to target_size, default is 300')
    parser.add_argument(
        '--use-original-resolution',
        action='store_true',
        default=True,
        help='Use original high-resolution PNG image for visualization')
    parser.add_argument('--device', default='cpu', help='Device to use cpu')
    parser.add_argument(
        '--vit-like',
        action='store_true',
        help='Whether the network is a ViT-like network.')
    parser.add_argument(
        '--num-extra-tokens',
        type=int,
        help='The number of extra tokens in ViT-like backbones. Defaults to'
             ' use num_extra_tokens of the backbone.')
    parser.add_argument(
        '--cam-alpha',
        type=float,
        default=0.6,
        help='Alpha blending factor for CAM overlay, default is 0.6')
    parser.add_argument(
        '--interpolation',
        type=str,
        default='lanczos',
        choices=['nearest', 'bilinear', 'bicubic', 'lanczos'],
        help='Interpolation method for resizing, default is lanczos')
    args = parser.parse_args()
    if args.method.lower() not in METHOD_MAP.keys():
        raise ValueError(f'invalid CAM type {args.method},'
                         f' supports {", ".join(list(METHOD_MAP.keys()))}.')

    return args


def validate_image_paths(args):
    """验证所有图像路径是否存在"""
    categories = {
        'alpha': args.alpha,
        'delta': args.delta,
        'omicron': args.omicron,
        'else_cat': args.else_cat
    }

    for cat_name, path in categories.items():
        if not path.exists():
            raise FileNotFoundError(f"{cat_name.capitalize()} category image not found: {path}")
        if path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            print(f"Warning: {cat_name} image is not a standard image format: {path.suffix}")


def build_reshape_transform(model, args):
    """Build reshape_transform for `cam.activations_and_gradients`, which is
    necessary for ViT-like networks."""
    if not args.vit_like:
        def check_shape(tensor):
            assert len(tensor.size()) != 3, \
                (f"The input feature's shape is {tensor.size()}, and it seems "
                 'to have been flattened or from a vit-like network. '
                 "Please use `--vit-like` if it's from a vit-like network.")
            return tensor

        return check_shape

    if args.num_extra_tokens is not None:
        num_extra_tokens = args.num_extra_tokens
    elif hasattr(model.backbone, 'num_extra_tokens'):
        num_extra_tokens = model.backbone.num_extra_tokens
    else:
        num_extra_tokens = 1

    def _reshape_transform(tensor):
        """reshape_transform helper."""
        assert len(tensor.size()) == 3, \
            (f"The input feature's shape is {tensor.size()}, "
             'and the feature seems not from a vit-like network?')
        tensor = tensor[:, num_extra_tokens:, :]
        heat_map_area = tensor.size()[1]
        height, width = to_2tuple(int(math.sqrt(heat_map_area)))
        assert height * height == heat_map_area, \
            (f"The input feature's length ({heat_map_area + num_extra_tokens}) "
             f'minus num-extra-tokens ({num_extra_tokens}) is {heat_map_area},'
             ' which is not a perfect square number. Please check if you used '
             'a wrong num-extra-tokens.')
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

        result = result.transpose(2, 3).transpose(1, 2)
        return result

    return _reshape_transform


def load_original_image(img_path, target_size=None):
    """直接加载原始高分辨率PNG图像，支持多种格式"""
    try:
        pil_img = Image.open(img_path)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        # 获取图像信息
        img_format = pil_img.format
        original_size = pil_img.size

        # 如果需要，调整图像大小到目标尺寸
        if target_size:
            # 保持宽高比的调整
            pil_img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

        current_size = pil_img.size

        original_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        print(
            f"Original image loaded: {original_size[0]}x{original_size[1]} -> {current_size[0]}x{current_size[1]}, format: {img_format}")
        return original_img
    except Exception as e:
        raise ValueError(f"Error loading image {img_path}: {e}")


def apply_transforms(img_path, pipeline_cfg, use_original_resolution=False, target_size=None):
    """Apply transforms pipeline and get both formatted data and the image
    without formatting."""
    data = dict(img_info=dict(filename=img_path), img_prefix=None)

    def split_pipeline_cfg(pipeline_cfg):
        """to split the transfoms into image_transforms and
        format_transforms."""
        image_transforms_cfg, format_transforms_cfg = [], []
        if pipeline_cfg[0]['type'] != 'LoadImageFromFile':
            pipeline_cfg.insert(0, dict(type='LoadImageFromFile'))
        for transform in pipeline_cfg:
            if transform['type'] in FORMAT_TRANSFORMS_SET:
                format_transforms_cfg.append(transform)
            else:
                image_transforms_cfg.append(transform)
        return image_transforms_cfg, format_transforms_cfg

    image_transforms, format_transforms = split_pipeline_cfg(pipeline_cfg)
    image_transforms = Compose(image_transforms)
    format_transforms = Compose(format_transforms)

    intermediate_data = image_transforms(data)

    inference_img = copy.deepcopy(intermediate_data['img'])
    format_data = format_transforms(intermediate_data)

    if use_original_resolution:
        try:
            original_img = load_original_image(img_path, target_size)
            return format_data, original_img
        except Exception as e:
            print(f"Warning: Could not load original image: {e}")
            print("Using inference image instead")
            return format_data, inference_img
    else:
        return format_data, inference_img


class MMActivationsAndGradients(ActivationsAndGradients):
    """Activations and gradients manager for mmcls models."""

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(
            x, return_loss=False, softmax=False, post_process=False)


def init_cam(method, model, target_layers, use_cuda, reshape_transform):
    """Construct the CAM object once."""

    GradCAM_Class = METHOD_MAP[method.lower()]

    try:
        cam = GradCAM_Class(
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform)
    except TypeError:
        cam = GradCAM_Class(
            model=model,
            target_layers=target_layers,
            use_cuda=use_cuda,
            reshape_transform=reshape_transform)

    cam.activations_and_grads.release()
    cam.activations_and_grads = MMActivationsAndGradients(
        cam.model, cam.target_layers, reshape_transform)

    return cam


def get_layer(layer_str, model):
    """get model layer from given str."""
    cur_layer = model
    layer_names = layer_str.strip().split('.')

    def get_children_by_name(model, name):
        try:
            return getattr(model, name)
        except AttributeError as e:
            raise AttributeError(
                e.args[0] +
                '. Please use `--preview-model` to check keys at first.')

    def get_children_by_eval(model, name):
        try:
            return eval(f'model{name}', {}, {'model': model})
        except (AttributeError, IndexError) as e:
            raise AttributeError(
                e.args[0] +
                '. Please use `--preview-model` to check keys at first.')

    for layer_name in layer_names:
        match_res = re.match('(?P<name>.+?)(?P<indices>(\\[.+\\])+)',
                             layer_name)
        if match_res:
            layer_name = match_res.groupdict()['name']
            indices = match_res.groupdict()['indices']
            cur_layer = get_children_by_name(cur_layer, layer_name)
            cur_layer = get_children_by_eval(cur_layer, indices)
        else:
            cur_layer = get_children_by_name(cur_layer, layer_name)

    return cur_layer


def create_cam_overlay_with_jet(grayscale_cam, original_img, alpha=0.6):
    """创建CAM叠加图，使用jet colormap（红色表示高值，蓝色表示低值）"""
    if original_img is None:
        raise ValueError("Original image is None")

    grayscale_cam = grayscale_cam[0, :]

    original_height, original_width = original_img.shape[:2]
    cam_height, cam_width = grayscale_cam.shape

    # 调整CAM大小以匹配原始图像
    if (original_height != cam_height) or (original_width != cam_width):
        resized_cam = cv2.resize(grayscale_cam,
                                 (original_width, original_height),
                                 interpolation=cv2.INTER_LANCZOS4)
    else:
        resized_cam = grayscale_cam

    # 将CAM归一化到0-1
    cam_normalized = (resized_cam - resized_cam.min()) / (resized_cam.max() - resized_cam.min() + 1e-8)

    # 使用jet colormap生成彩色CAM
    # OpenCV的COLORMAP_JET是：蓝色(低值) -> 青色 -> 黄色 -> 红色(高值)
    cam_colored = cv2.applyColorMap(np.uint8(255 * cam_normalized), cv2.COLORMAP_JET)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

    # 将原始图像转换为RGB浮点数
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_rgb_float = original_rgb.astype(np.float32) / 255.0

    # 将彩色CAM转换为浮点数
    cam_colored_float = cam_colored.astype(np.float32) / 255.0

    # 使用alpha混合
    overlay_rgb = (cam_colored_float * alpha + original_rgb_float * (1 - alpha))
    overlay_rgb = np.clip(overlay_rgb * 255, 0, 255).astype(np.uint8)

    return overlay_rgb, resized_cam, cam_colored


def create_cam_colored_with_jet(cam_array):
    """使用jet colormap创建彩色CAM图像"""
    # 归一化CAM
    cam_normalized = (cam_array - cam_array.min()) / (cam_array.max() - cam_array.min() + 1e-8)

    # 使用jet colormap
    cam_colored = cv2.applyColorMap(np.uint8(255 * cam_normalized), cv2.COLORMAP_JET)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

    return cam_colored


def crop_and_resize_image(image_array, crop_pixels=100, target_size=1000, interpolation='lanczos'):
    """
    裁剪图像四周并调整到目标尺寸

    Args:
        image_array: numpy数组格式的图像 (H, W, C)
        crop_pixels: 从上下左右裁剪的像素数
        target_size: 目标尺寸，会调整为target_size x target_size
        interpolation: 插值方法

    Returns:
        processed_image: 处理后的图像数组
    """
    height, width = image_array.shape[:2]

    print(f"    Original size: {width}x{height}")
    print(f"    Crop pixels: {crop_pixels} from each side")

    # 检查裁剪是否会导致图像太小
    new_height = height - 2 * crop_pixels
    new_width = width - 2 * crop_pixels

    if new_height <= 0 or new_width <= 0:
        print(f"    Warning: Crop too large, using minimum crop")
        crop_pixels = min(height // 4, width // 4)
        new_height = height - 2 * crop_pixels
        new_width = width - 2 * crop_pixels

    print(f"    After cropping: {new_width}x{new_height}")

    # 裁剪图像
    if crop_pixels > 0:
        cropped_image = image_array[crop_pixels:height - crop_pixels,
                        crop_pixels:width - crop_pixels]
    else:
        cropped_image = image_array

    # 转换为PIL图像以便使用高质量重采样
    pil_image = Image.fromarray(cropped_image)

    # 映射插值方法
    interpolation_map = {
        'nearest': Image.Resampling.NEAREST,
        'bilinear': Image.Resampling.BILINEAR,
        'bicubic': Image.Resampling.BICUBIC,
        'lanczos': Image.Resampling.LANCZOS
    }

    interpolation_method = interpolation_map.get(interpolation, Image.Resampling.LANCZOS)

    # 调整到目标尺寸
    if target_size:
        resized_image = pil_image.resize((target_size, target_size),
                                         interpolation_method)
        print(f"    After resizing: {target_size}x{target_size}")
    else:
        resized_image = pil_image

    return np.array(resized_image)


def resize_to_target(image_array, target_size=1000, interpolation='lanczos'):
    """
    将图像调整到目标尺寸

    Args:
        image_array: numpy数组格式的图像 (H, W, C)
        target_size: 目标尺寸，会调整为target_size x target_size
        interpolation: 插值方法

    Returns:
        processed_image: 处理后的图像数组
    """
    height, width = image_array.shape[:2]

    print(f"    Original size: {width}x{height}")

    # 转换为PIL图像以便使用高质量重采样
    pil_image = Image.fromarray(image_array)

    # 映射插值方法
    interpolation_map = {
        'nearest': Image.Resampling.NEAREST,
        'bilinear': Image.Resampling.BILINEAR,
        'bicubic': Image.Resampling.BICUBIC,
        'lanczos': Image.Resampling.LANCZOS
    }

    interpolation_method = interpolation_map.get(interpolation, Image.Resampling.LANCZOS)

    # 调整到目标尺寸
    resized_image = pil_image.resize((target_size, target_size),
                                     interpolation_method)
    print(f"    After resizing: {target_size}x{target_size}")

    return np.array(resized_image)


def extract_detail_region_from_overlay(overlay_rgb, cam_array, detail_size=300, target_size=1000,
                                       interpolation='lanczos'):
    """
    从CAM叠加图中提取细节区域并调整到目标尺寸

    Args:
        overlay_rgb: CAM叠加图
        cam_array: CAM数组
        detail_size: 原始细节区域大小
        target_size: 目标尺寸
        interpolation: 插值方法

    Returns:
        detail_region: 处理后的细节区域
    """
    # 找到CAM的最大激活位置
    max_idx = np.argmax(cam_array)
    y, x = np.unravel_index(max_idx, cam_array.shape)

    img_height, img_width = overlay_rgb.shape[:2]

    # 调整细节区域大小
    detail_size = min(detail_size, img_height // 4, img_width // 4)

    y_start = max(0, y - detail_size // 2)
    y_end = min(img_height, y + detail_size // 2)
    x_start = max(0, x - detail_size // 2)
    x_end = min(img_width, x + detail_size // 2)

    actual_height = y_end - y_start
    actual_width = x_end - x_start

    # 确保方形区域
    if actual_height > actual_width:
        diff = actual_height - actual_width
        x_start = max(0, x_start - diff // 2)
        x_end = min(img_width, x_end + diff // 2)
    elif actual_width > actual_height:
        diff = actual_width - actual_height
        y_start = max(0, y_start - diff // 2)
        y_end = min(img_height, y_end + diff // 2)

    # 从CAM叠加图中提取细节区域
    detail_region = overlay_rgb[y_start:y_end, x_start:x_end]

    if detail_region.size > 0:
        print(f"    Detail region extracted: {detail_region.shape[1]}x{detail_region.shape[0]}")
        # 调整到目标尺寸
        return resize_to_target(detail_region, target_size, interpolation)
    else:
        print(f"    Warning: Detail region empty, using full overlay")
        return resize_to_target(overlay_rgb, target_size, interpolation)


def save_single_image_pdf(image_array, output_path, dpi=600):
    """保存单个图像为PDF，无白边"""
    # 计算适当的图形大小（以英寸为单位）
    height_inches = image_array.shape[0] / dpi
    width_inches = image_array.shape[1] / dpi

    fig = plt.figure(figsize=(width_inches, height_inches), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image_array)

    # 使用tight layout并去除白边
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)

    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0, dpi=dpi)

    plt.close()


def save_cam_heatmap_pdf(cam_array, output_path, dpi=600):
    """保存CAM热力图为PDF，带colorbar"""
    # 计算适当的图形大小（以英寸为单位）
    height_inches = cam_array.shape[0] / dpi
    width_inches = cam_array.shape[1] / dpi

    fig = plt.figure(figsize=(width_inches, height_inches), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # 使用jet colormap
    im = ax.imshow(cam_array, cmap='jet', vmin=cam_array.min(), vmax=cam_array.max())

    # 添加colorbar，放在图像外部右侧
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=6)

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)

    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0, dpi=dpi)

    plt.close()


def process_and_save_images(results_list, output_dir, args):
    """为每个类别的四种图像类型生成独立的PDF"""
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Generating individual PDF files...")
    print(f"Output directory: {output_dir}")
    print(f"DPI: {args.dpi}")
    print(f"Crop pixels for original/overlay: {args.crop_pixels}")
    print(f"Target size for all images: {args.target_size}x{args.target_size}")
    print(f"CAM alpha: {args.cam_alpha}")
    print(f"Colormap: Jet (red=high, blue=low)")
    print(f"Interpolation: {args.interpolation}")
    print(f"{'=' * 60}")

    for result in results_list:
        category = result['category']
        print(f"\nProcessing {category} category...")

        # 创建类别子目录
        category_dir = output_dir / category.lower()
        category_dir.mkdir(exist_ok=True)

        # 1. 处理并保存原始图像（裁剪并调整大小）
        print(f"  1. Original image:")
        original_cropped = crop_and_resize_image(
            result['original_rgb'],
            crop_pixels=args.crop_pixels,
            target_size=args.target_size,
            interpolation=args.interpolation
        )

        original_pdf = category_dir / f"{category.lower()}_original.pdf"
        save_single_image_pdf(original_cropped, original_pdf, dpi=args.dpi)
        print(f"    ✓ Saved to: {original_pdf}")

        # 2. 处理并保存CAM热力图（直接调整大小）
        print(f"  2. CAM heatmap:")
        cam_resized = resize_to_target(
            result['resized_cam'],
            target_size=args.target_size,
            interpolation=args.interpolation
        )

        cam_pdf = category_dir / f"{category.lower()}_cam_heatmap.pdf"
        save_cam_heatmap_pdf(cam_resized, cam_pdf, dpi=args.dpi)
        print(f"    ✓ Saved to: {cam_pdf}")

        # 3. 处理并保存CAM叠加图（裁剪并调整大小）
        print(f"  3. CAM overlay:")
        overlay_cropped = crop_and_resize_image(
            result['overlay_rgb'],
            crop_pixels=args.crop_pixels,
            target_size=args.target_size,
            interpolation=args.interpolation
        )

        overlay_pdf = category_dir / f"{category.lower()}_cam_overlay.pdf"
        save_single_image_pdf(overlay_cropped, overlay_pdf, dpi=args.dpi)
        print(f"    ✓ Saved to: {overlay_pdf}")

        # 4. 处理并保存细节区域（从CAM叠加图中提取并调整大小）
        print(f"  4. Detail region (from CAM overlay):")
        # 注意：这里使用未裁剪的CAM叠加图和CAM数组来定位
        detail_region = extract_detail_region_from_overlay(
            result['overlay_rgb'],  # 使用未裁剪的叠加图
            result['resized_cam'],  # 使用CAM数组定位
            detail_size=args.detail_size,
            target_size=args.target_size,
            interpolation=args.interpolation
        )

        detail_pdf = category_dir / f"{category.lower()}_detail_region.pdf"
        save_single_image_pdf(detail_region, detail_pdf, dpi=args.dpi)
        print(f"    ✓ Saved to: {detail_pdf}")


def process_single_category(img_path, cam, val_pipeline, args, category_name):
    """处理单个类别的图像"""
    print(f"\nProcessing {category_name} category...")

    # 加载和预处理图像
    try:
        data, src_img = apply_transforms(img_path, val_pipeline, args.use_original_resolution, args.target_size)
        print(f"Loaded {category_name} image: {src_img.shape}")
    except Exception as e:
        print(f"Error loading {category_name} image: {e}")
        try:
            src_img = load_original_image(img_path, args.target_size)
            from torchvision import transforms
            preprocess = transforms.Compose([
                transforms.Resize((args.target_size, args.target_size),
                                  interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            src_img_rgb = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(src_img_rgb)

            img_tensor = preprocess(pil_img)
            data = {'img': img_tensor}
            print(f"Directly loaded {category_name} image: {src_img.shape}")
        except Exception as e2:
            print(f"Failed to load {category_name} image: {e2}")
            raise

    # 准备输入张量
    img_tensor = data['img']
    if len(img_tensor.shape) == 5:
        img_tensor = img_tensor.squeeze(1)
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)

    print(f"Input tensor shape for {category_name}: {img_tensor.shape}")

    # 计算CAM
    grayscale_cam = cam(
        img_tensor,
        None,
        eigen_smooth=args.eigen_smooth,
        aug_smooth=args.aug_smooth)

    print(f"CAM shape: {grayscale_cam.shape}")

    # 创建叠加图（使用jet colormap）
    overlay_rgb, resized_cam, cam_colored = create_cam_overlay_with_jet(
        grayscale_cam, src_img, alpha=args.cam_alpha)

    # 转换为RGB
    original_rgb = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

    return {
        'category': category_name,
        'original_rgb': original_rgb,
        'resized_cam': resized_cam,
        'overlay_rgb': overlay_rgb,
        'cam_colored': cam_colored,  # 保存彩色CAM用于其他用途
        'src_img': src_img
    }


def get_default_traget_layers(model, args):
    """get default target layers from given model."""
    norm_layers = []
    for m in model.backbone.modules():
        if isinstance(m, (BatchNorm2d, LayerNorm, GroupNorm, BatchNorm1d)):
            norm_layers.append(m)
    if len(norm_layers) == 0:
        raise ValueError(
            '`--target-layers` is empty. Please use `--preview-model`'
            ' to check keys at first and then specify `target-layers`.')

    if args.vit_like:
        if args.num_extra_tokens:
            num_extra_tokens = args.num_extra_tokens
        elif hasattr(model.backbone, 'num_extra_tokens'):
            num_extra_tokens = model.backbone.num_extra_tokens
        else:
            raise AttributeError('Please set num_extra_tokens in backbone'
                                 " or using 'num-extra-tokens'")

        if num_extra_tokens >= 1:
            print('Automatically choose the last norm layer before the '
                  'final attention block as target_layer..')
            return [norm_layers[-3]]

    print('Automatically choose the last norm layer as target_layer.')
    return [norm_layers[-1]]


def main():
    args = parse_args()

    # 验证图像路径
    validate_image_paths(args)

    # 验证输出目录
    output_dir = args.output_dir
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    model_cfg, train_pipeline, val_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(args.config)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BuildNet(model_cfg)
    model = init_model(model, data_cfg, device=device, mode='eval')

    if args.preview_model:
        print(model)
        print('\n Please remove `--preview-model` to get the CAM.')
        return

    # build target layers
    if args.target_layers:
        target_layers = [
            get_layer(layer, model) for layer in args.target_layers
        ]
    else:
        target_layers = get_default_traget_layers(model, args)

    # init a cam grad calculator
    use_cuda = ('cuda' in args.device)
    reshape_transform = build_reshape_transform(model, args)
    cam = init_cam(args.method, model, target_layers, use_cuda,
                   reshape_transform)

    # 处理四个类别的图像
    results_list = []

    # Alpha
    alpha_result = process_single_category(
        args.alpha, cam, val_pipeline, args, 'Alpha')
    results_list.append(alpha_result)

    # Delta
    delta_result = process_single_category(
        args.delta, cam, val_pipeline, args, 'Delta')
    results_list.append(delta_result)

    # Omicron
    omicron_result = process_single_category(
        args.omicron, cam, val_pipeline, args, 'Omicron')
    results_list.append(omicron_result)

    # Else
    else_result = process_single_category(
        args.else_cat, cam, val_pipeline, args, 'Else')
    results_list.append(else_result)

    # 为每个类别的四种图像类型生成独立的PDF
    process_and_save_images(results_list, output_dir, args)

    print(f"\n{'=' * 60}")
    print("SUMMARY: Generated 16 individual PDF files:")
    print("  4 categories × 4 image types = 16 PDFs")
    print(f"  Output directory: {output_dir}")
    print(f"  Categories processed: Alpha, Delta, Omicron, Else")
    print(f"\n  For each category:")
    print(
        f"    1. Original image - cropped {args.crop_pixels}px from each side, resized to {args.target_size}x{args.target_size}")
    print(f"    2. CAM heatmap - directly resized to {args.target_size}x{args.target_size} (jet colormap)")
    print(
        f"    3. CAM overlay - cropped {args.crop_pixels}px from each side, resized to {args.target_size}x{args.target_size} (jet colormap)")
    print(
        f"    4. Detail region - extracted from CAM overlay (not cropped), resized to {args.target_size}x{args.target_size} (jet colormap)")
    print(f"\n  Color scheme: Jet colormap - Red (high activation) → Yellow → Green → Cyan → Blue (low activation)")
    print(f"  Processing details:")
    print(f"    CAM alpha: {args.cam_alpha}")
    print(f"    Interpolation: {args.interpolation}")
    print(f"    DPI: {args.dpi}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()