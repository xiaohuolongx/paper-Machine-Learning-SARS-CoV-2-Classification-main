制作数据集
===========================


## 1. 标签文件制作

- 本次演示以新冠病毒数据集为例，目录结构如下：

```
├─data
│  ├─Alpha
│  │      95_2020-11-13.png
│  │      95_2020-12-07.png
│  │      ...
│  ├─Delta
│  │      96_2021-04-30.png
│  │      96_2021-05-17.png
│  │      ...
│  ├─Omicron
│  │      95_2022-07-20.png
│  │      96_2021-12-21.png
│  │      ...
│  ├─Else
│  │      95_2020-10-21.png
│  │      95_2020-10-26.png
│  │      ...
```
- 在`paper-Machine Learning-SARS-CoV-2-Classification-main/datas/`中创建标签文件`annotations.txt`，按行将`类别名 索引`写入文件；
```
Alpha 0
Delta 1
Omicron 2
Else 3
```
## 2. 数据集划分
- 打开`paper-Machine Learning-SARS-CoV-2-Classification-main/tools/split_data.py`
- 修改`原始数据集路径`以及`划分后的保存路径`，强烈建议划分后的保存路径`datasets`不要改动，在下一步都是默认基于文件夹进行操作
```
init_dataset = 'A:/circos-data-set'
new_dataset = 'A:/paper-Machine Learning-SARS-CoV-2-Classification-main/datasets'
```
- 在`paper-Machine Learning-SARS-CoV-2-Classification-main/`下打开终端输入命令：
```
python tools/split_data.py
```
- 得到划分后的数据集格式如下：
```
├─...
├─datasets
│  ├─test
│  │  ├─Alpha
│  │  ├─Delta
│  │  ├─Omicron
│  │  ├─Else
│  └─train
│      ├─Alpha
│      ├─Delta
│      ├─Omicron
│      ├─Else
├─...
```
## 3. 数据集信息文件制作
- 确保划分后的数据集是在`paper-Machine Learning-SARS-CoV-2-Classification-main/datasets`下，若不在则在`get_annotation.py`下修改数据集路径；
```
datasets_path   = '你的数据集路径'
```
- 在`paper-Machine Learning-SARS-CoV-2-Classification-main/`下打开终端输入命令：
```
python tools/get_annotation.py
```
- 在`paper-Machine Learning-SARS-CoV-2-Classification-main/datas`下得到生成的数据集信息文件`train.txt`与`test.txt`
