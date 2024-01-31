# 项目介绍
该项目为华东师范大学数据学院2023年秋季学期课程《当代人工智能》的第五次实验作业，是基于文本和图像的多模态情感分析。
## 设置依赖
若要运行此代码，请在Python3下安装这些依赖：
torch==2.1.0+cu121
torchvision==0.16.0+cu121
transformers==4.31.0
pandas==2.1.1
opencv-python==4.9.0.80
numpy==1.25.2
scikit-learn==1.3.1
Pillow==10.0.0

您也可以运行这个安装：
```python
pip install -r requirements.txt
```

## 文件结构
这里我将对项目的文件结构进行说明
```python
    |-- data # 所有文本和图像数据所在
        |-- 1.txt
        |-- 1.jpg
        |-- 2.txt
        |-- 2.jpg
        ......
    |-- main.ipynb # 完整代码，运行后输出5个epoch结果、预测测试集文件以及消融实验结果
    |-- train.txt # 存放数据的guid和对应的情感标签。
    |-- test_without_label.txt # 测试集文件：数据的guid和空的情感标签。
    |-- predict.txt # 预测结果
    |-- requirements.txt # 所需依赖
    |-- README.md # readme文件
......
```
## 运行说明
1.main.ipynb文件可以直接打开后运行，需要注意的是改代码初次运行会自行下载ResNet-50和BERT预训练模型

2.main.ipynb文件最后两个代码模块是消融实验的运行结果

3.本实验设置的batch_size=32，在NVIDIA GeForce RTX 3060显卡运行成功，如若您运行时出现CUDA out of memory，请尝试适当减小参数（这样做可能会影响实验结果）。
## Attribution
此代码的某些部分基于以下库：
- [PyTroch](https://github.com/pytorch/pytorch)
- [Transformers (by Hugging Face)](https://github.com/huggingface/transformers)
- [Pandas](https://github.com/pandas-dev/pandas)
- [Pillow(PIL Fork)](https://github.com/python-pillow/Pillow)
- [torchivision](https://github.com/pytorch/vision)
