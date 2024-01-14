# ChatGLM 模型微调和训练

本文档介绍模型的微调、加载和验证方法。

## 依赖

首先需要安装基本环境，包括 CUDA，Torch。
此外还要安装模型的依赖

```
pip install -r requirements.txt
```

## 数据集

所有的数据集均位于 `myiris` 文件夹下，其中 `450_data.jsonl` 为训练集 `dev_60` 为测试集。表格的准备和可见 `myiris/excel` 文件夹。

## 模型微调

模型微调采用 P-Tuning v2 微调方法，想要进行微调，可以执行 `pt-450.sh` 脚本。
注意，原模型需要克隆到 `chatglm3-6b` 文件夹下，或者修改脚本中的模型路径变量（后同）。
```
git lfs install
git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git
```
如果相对路径存在问题，可以修改成绝对路径。
在训练时可以根据需要调整脚本中的训练步数和学习率等参数。
```
bash pt-450.sh
```

## 微调后的模型加载

要想加载微调后的模型，可以运行 `inference.bash` 脚本，注意需要调整脚本中的微调模型的 `CHECKPOINTPATH`路径和 `pt-pre-seq-len` 变量以保证模型能够正常加载。
```
bash inference.bash
```

## 批量生成推理

运行 `evaluate.bash` 脚本可以批量生成推理，可根据需要修改脚本中的验证集的路径 `DEV_PATH`，
同样需要像 `inference.bash` 一样调整参数以保证微调模型能够正常加载。
```
bash evaluate.bash
```
生成的推理会放在对应的微调后的模型文件夹下的 `evaluate.json` 文件，如 `EXCEL_pt-500-5e-3-128/evaluate.json`。