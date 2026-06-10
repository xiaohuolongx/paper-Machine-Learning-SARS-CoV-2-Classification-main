
## 训练自己的数据集（k-fold）

### 1. 数据集准备

- 确保标签文件 `datas/annotations.txt` 已准备就绪（格式一般为：`样本路径 标签` 或 `样本路径,标签`，具体参照已有配置）。
- 确保 `datas/train.txt` 与 `datas/test.txt` 已经按照 `annotations.txt` 生成。  
  - **普通模式**：`train.txt` 为训练集，`test.txt` 为验证集。  
  - **K 折交叉验证模式**：仅使用 `train.txt`，脚本会按比例自动划分验证集。
- 所有文本文件中每行代表一个样本，内容格式需与配置文件中 `train_pipeline` 的数据读取方式一致（通常为路径和标签）。

### 2. 选择配置文件

- 进入 `models/` 目录，选择您想要训练的模型（例如 `mobilenet/mobilenet_v3_small.py`）。
- 复制该配置文件，或直接修改其中的参数（参见下节“配置文件解释”）。

### 3. 配置文件关键参数修改

| 参数 | 说明 |
|------|------|
| `model_cfg.backbone.type` | 骨干网络类型（如 `MobileNetV3Small`） |
| `data_cfg.batch_size` | 批大小（根据 GPU 显存调整） |
| `data_cfg.train.epoches` | 训练总轮数 |
| `data_cfg.train.pretrained_flag` | 是否使用预训练权重 |
| `data_cfg.train.freeze_flag` / `freeze_layers` | 是否冻结部分层 |
| `train_pipeline` | 数据预处理流水线（如归一化、随机裁剪等） |
| `optimizer_cfg.type` 及其参数 | 优化器类型（`SGD`, `Adam`）与学习率、动量等 |
| `lr_config` | 学习率调度策略（`StepLR`, `CosineAnnealing` 等） |

### 4. 运行训练

在项目根目录（`paper-Machine Learning-SARS-CoV-2-Classification-main`）打开终端，执行：

```bash
python tools/train.py models/mobilenet/mobilenet_v3_small.py
```

若需要 K 折交叉验证，请添加 `--split-validation` 参数：

```bash
python tools/train.py models/mobilenet/mobilenet_v3_small.py --split-validation --ratio 0.2
```

### 5. 命令行参数完整列表

```bash
python tools/train.py \
    ${CONFIG_FILE} \
    [--resume-from CHECKPOINT_PATH] \
    [--seed SEED] \
    [--device DEVICE] \
    [--gpu-id GPU_ID] \
    [--split-validation] \
    [--ratio RATIO] \
    [--deterministic]
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | 必填 | 模型配置文件的路径（如 `models/mobilenet/mobilenet_v3_small.py`） |
| `--resume-from` | 可选 | 恢复训练的权重文件路径，**必须指向完整的 checkpoint**（如 `Last_Epoch15.pth`） |
| `--seed` | 可选 | 随机种子，用于结果复现 |
| `--device` | 可选 | 训练设备（`cuda` 或 `cpu`），默认自动选择 GPU |
| `--gpu-id` | 可选 | 使用的 GPU 编号，默认为 `0`（单卡训练无需修改） |
| `--split-validation` | 开关 | 是否从训练集（`train.txt`）中划分验证集，开启后忽略 `test.txt` |
| `--ratio` | 可选 | 验证集占训练集的比例，默认 `0.2`（例如 `0.2` 表示 5 折交叉验证） |
| `--deterministic` | 开关 | 设置 CUDNN 确定性算法，提高可复现性（多 GPU 训练时不建议开启） |

### 6. 训练输出

- 日志与模型权重保存在 `logs/{backbone_type}/{timestamp}/` 目录下。
- 最佳模型权重：`best_val_weight.pth`
- 最后一个 epoch 的权重：`last_weight.pth`
- 训练曲线图：由 `History` 模块自动生成。

### 7. K 折交叉验证注意事项

- 开启 `--split-validation` 后，脚本会将 `train.txt` 中的所有数据打乱并按比例分成 K 份（K = 1 / ratio）。
- 每折训练独立进行，结果保存在 `logs/.../fold_0/`、`fold_1/` 等子目录下。
- 最终会打印每折的最佳验证准确率以及平均值 ± 标准差，并保存到 `cv_results.txt`。

---

## Training on Your Own Dataset （k-fold）

### 1. Dataset Preparation

- Ensure the label file `datas/annotations.txt` is ready (format: `sample_path label` or `sample_path,label` — check your configuration).
- Ensure that `datas/train.txt` and `datas/test.txt` are generated according to `annotations.txt`.  
  - **Normal mode**: `train.txt` is used for training, `test.txt` for validation.  
  - **K‑fold CV mode**: Only `train.txt` is used; the validation set is split from it automatically.
- Each line in these text files should represent one sample, with the same format expected by the `train_pipeline` in your config file.

### 2. Choose a Configuration File

- Go to the `models/` directory and pick the model you want to train (e.g., `mobilenet/mobilenet_v3_small.py`).
- Copy or directly modify the configuration file (see “Configuration Parameters” below).

### 3. Key Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `model_cfg.backbone.type` | Backbone network type (e.g., `MobileNetV3Small`) |
| `data_cfg.batch_size` | Batch size (adjust according to GPU memory) |
| `data_cfg.train.epoches` | Total number of training epochs |
| `data_cfg.train.pretrained_flag` | Whether to load pretrained weights |
| `data_cfg.train.freeze_flag` / `freeze_layers` | Whether to freeze certain layers |
| `train_pipeline` | Data preprocessing pipeline (normalization, random crop, etc.) |
| `optimizer_cfg.type` + parameters | Optimizer type (`SGD`, `Adam`) with learning rate, momentum, etc. |
| `lr_config` | Learning rate scheduler (`StepLR`, `CosineAnnealing`, etc.) |

### 4. Run Training

Open a terminal in the project root (`paper-Machine Learning-SARS-CoV-2-Classification-main`) and run:

```bash
python tools/train.py models/mobilenet/mobilenet_v3_small.py
```

To enable K‑fold cross‑validation, add `--split-validation`:

```bash
python tools/train.py models/mobilenet/mobilenet_v3_small.py --split-validation --ratio 0.2
```

### 5. Full Command Line Options

```bash
python tools/train.py \
    ${CONFIG_FILE} \
    [--resume-from CHECKPOINT_PATH] \
    [--seed SEED] \
    [--device DEVICE] \
    [--gpu-id GPU_ID] \
    [--split-validation] \
    [--ratio RATIO] \
    [--deterministic]
```

| Argument | Type | Description |
|----------|------|-------------|
| `config` | required | Path to the model configuration file (e.g., `models/mobilenet/mobilenet_v3_small.py`) |
| `--resume-from` | optional | Path to a checkpoint to resume training. **Must be a full checkpoint** (e.g., `Last_Epoch15.pth`) |
| `--seed` | optional | Random seed for reproducibility |
| `--device` | optional | Training device (`cuda` or `cpu`). Default auto‑selects GPU if available |
| `--gpu-id` | optional | GPU device ID. Default is `0` (unchanged for single‑card training) |
| `--split-validation` | flag | Whether to split a validation set from the training set (`train.txt`). If set, `test.txt` is ignored. |
| `--ratio` | optional | Ratio of validation set to training set. Default `0.2` (e.g., 0.2 → 5‑fold CV) |
| `--deterministic` | flag | Set CUDNN deterministic algorithms for better reproducibility (not recommended for multi‑GPU) |

### 6. Training Outputs

- Logs and model weights are saved under `logs/{backbone_type}/{timestamp}/`.
- Best model weight: `best_val_weight.pth`
- Last epoch weight: `last_weight.pth`
- Training curves are automatically generated by the `History` module.

### 7. K‑Fold Cross‑Validation Notes

- When `--split-validation` is enabled, all data from `train.txt` is shuffled and split into K folds (K = 1 / ratio).
- Each fold is trained independently, and results are stored in subdirectories like `fold_0/`, `fold_1/`, etc.
- After all folds, the script prints the best validation accuracy for each fold, along with the mean ± standard deviation, and saves these results to `cv_results.txt`.

---

**提示**：使用 K 折交叉验证时，每个折的训练是独立的，因此总训练时间将大约是单次训练的 K 倍。请根据您的计算资源合理选择 `ratio`。

**Note**: When using K‑fold cross‑validation, each fold is trained independently, so the total training time will be approximately K times that of a single run. Please choose `ratio` according to your computational resources.
