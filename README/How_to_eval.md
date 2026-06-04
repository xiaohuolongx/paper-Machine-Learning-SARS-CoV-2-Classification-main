模型评估
===========================

## 评估测试集
- 确认`paper-Machine Learning-SARS-CoV-2-Classification-main/datas/annotations.txt`标签准备完毕
- 确认`paper-Machine Learning-SARS-CoV-2-Classification-main/datas/`下`test.txt`与`annotations.txt`对应
- 在`paper-Machine Learning-SARS-CoV-2-Classification-main/models/`下找到对应配置文件
- 按照`配置文件解释`修改参数，主要修改权重路径
- 在`paper-Machine Learning-SARS-CoV-2-Classification-main`打开终端运行
```
python tools/evaluation.py models/mobilenet/mobilenet_v3_small.py
```

## 单张图像检测
- 在`paper-Machine Learning-SARS-CoV-2-Classification-main`打开终端运行
```
python tools/single_test.py datasets/test/Alpha/95_2020-11-13.png models/mobilenet/mobilenet_v3_small.py
```
**参数说明**：

`img` : 被测试的单张图像路径

`config` : 模型配置文件，需注意修改配置文件中`data_cfg->test->ckpt`的权重路径，将使用该权重进行预测

`--classes-map` : 数据集对应的标签文件，默认datas/annotations.txt

`--device` : 推理所用设备，默认GPU

`--save-path` : 保存路径，默认不保存


## 批量图像检测
- 在`paper-Machine Learning-SARS-CoV-2-Classification-main`打开终端运行
```
python tools/batch_test.py datasets/test/Alpha models/mobilenet/mobilenet_v3_small.py --show
```
**参数说明**：

`path` : 待批量检测的图片文件夹路径

`config` : 模型配置文件，需注意修改配置文件中`data_cfg->test->ckpt`的权重路径，将使用该权重进行预测

`--classes-map` : 数据集对应的标签文件，默认datas/annotations.txt

`--device` : 推理所用设备，默认GPU

`--save-path` : 保存路径，默认不保存

`--show'`:批量检测时是否显示图片


**Model Evaluation**
===========================

## Evaluation Test Set
- Confirm that the label file `paper-Machine Learning-SARS-CoV-2-Classification-main/datas/annotations.txt` is ready.
- Confirm that `test.txt` under `paper-Machine Learning-SARS-CoV-2-Classification-main/datas/` corresponds to `annotations.txt`.
- Locate the corresponding configuration file under `paper-Machine Learning-SARS-CoV-2-Classification-main/models/`.
- Modify parameters according to the `Configuration File Explanation`, mainly changing the weight path.
- Open a terminal in `paper-Machine Learning-SARS-CoV-2-Classification-main` and run:
```
python tools/evaluation.py models/mobilenet/mobilenet_v3_small.py
```

## Single Image Detection
- Open a terminal in `paper-Machine Learning-SARS-CoV-2-Classification-main` and run:
```
python tools/single_test.py datasets/test/Alpha/95_2020-11-13.png models/mobilenet/mobilenet_v3_small.py
```
**Parameter Description**:

`img` : Path to the single image to be tested.

`config` : Model configuration file. Note that you need to modify the `ckpt` path under `data_cfg->test` in the configuration file; this weight will be used for prediction.

`--classes-map` : Label file corresponding to the dataset. Default is `datas/annotations.txt`.

`--device` : Device used for inference. Default is GPU.

`--save-path` : Path to save the result. Default is not to save.

## Batch Image Detection
- Open a terminal in `paper-Machine Learning-SARS-CoV-2-Classification-main` and run:
```
python tools/batch_test.py datasets/test/Alpha models/mobilenet/mobilenet_v3_small.py --show
```
**Parameter Description**:

`path` : Path to the folder containing images to be tested in batch.

`config` : Model configuration file. Note that you need to modify the `ckpt` path under `data_cfg->test` in the configuration file; this weight will be used for prediction.

`--classes-map` : Label file corresponding to the dataset. Default is `datas/annotations.txt`.

`--device` : Device used for inference. Default is GPU.

`--save-path` : Path to save the results. Default is not to save.

`--show` : Whether to display images during batch detection.
