类别激活图可视化
===========================

这是一个用于生成类别激活图（CAM）可视化的Python脚本，专门处理四类图像（Alpha、Delta、Omicron、Else）并输出高质量的PDF文件。
高质量输出：生成高DPI（默认600）的PDF文件
灵活的图像处理：支持裁剪、缩放、细节提取等操作

目前支持的方法有：

| Method     | What it does |
|:----------:|:------------:|
| GradCAM    | 使用平均梯度对 2D 激活进行加权 |
| GradCAM++  | 类似 GradCAM，但使用了二阶梯度 |
| XGradCAM   | 类似 GradCAM，但通过归一化的激活对梯度进行了加权 |
| EigenCAM   | 使用 2D 激活的第一主成分（无法区分类别，但效果似乎不错）|
| EigenGradCAM  | 类似 EigenCAM，但支持类别区分，使用了激活 \* 梯度的第一主成分，看起来和 GradCAM 差不多，但是更干净 |
| LayerCAM  | 使用正梯度对激活进行空间加权，对于浅层有更好的效果 |

核心功能模块
- 图像加载与预处理 (load_original_image, apply_transforms)
支持多种图像格式
可保持原始分辨率或调整大小
- CAM生成 (init_cam, create_cam_overlay_with_jet)
支持ViT类网络结构
使用Jet colormap（红高蓝低）
图像后处理 (crop_and_resize_image, extract_detail_region_from_overlay)
- 图像裁剪和缩放
自动提取高激活区域细节
- PDF输出 (save_single_image_pdf, save_cam_heatmap_pdf)
无白边PDF生成
支持colorbar

输出文件结构
每个类别生成4个PDF文件：
```bash
{category}_original.pdf - 原始图像（裁剪+缩放）
{category}_cam_heatmap.pdf - CAM热力图
{category}_cam_overlay.pdf - CAM叠加图
{category}_detail_region.pdf - 高激活区域细节
```

**命令行**：

```bash
python script.py \
  --alpha path/to/alpha.png \
  --delta path/to/delta.png \
  --omicron path/to/omicron.png \
  --else-cat path/to/else.png \
  config.py \
  --output-dir ./cam_results \
  --method gradcam \
  --target-layers layer1 layer2 \
  --dpi 600 \
  --crop-pixels 100 \
  --target-size 1000
```

**所有参数的说明**：
- `alpha/delta/omicron/else-cat`: 四类图像路径
- `config`: 模型配置文件
- `method`: CAM方法（gradcam, gradcam++, eigencam等）
- `target-layers`: 目标层名称
- `output-dir`: 输出目录
- `dpi`: PDF分辨率（默认600）
- `crop-pixels`: 裁剪像素数（默认100）
- `target-size`: 目标图像尺寸（默认1000）
- `cam-alpha`: CAM透明度（默认0.6）

**示例（CNN）**：使用不同方法可视化 `MobileNetV3`。
```
python tools/vis_cam.py --alpha ./images/alpha.png --delta ./images/delta.png --omicron ./images/omicron.png --else-cat ./images/else.png ./configs/resnet50.py --output-dir ./cam_results --method gradcam --dpi 600 --crop-pixels 100 --target-size 1000 --cam-alpha 0.6
```