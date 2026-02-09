# 加载模型配置文件

## 概要说明

仿真计算过程中需要读取模型的`config.json`文件，这里提供了三种加载该文件的方式，通过指定`--model_path`参数来完成。

## 使用方法

### 参数说明

在执行参数上增加 `--model_path ${MODEL_PATH}` 来指定加载的config文件。

```shell
python src/main.py --model_path ${MODEL_PATH}
```

### 加载本地参数

直接在参数中指定对应文件的地址，既可以加载本地的配置文件。

```shell
python src/main.py --model_path hf_config/deepseek_671b_r1_config.json
```

### 加载 huggingface(默认) 的远程配置文件

1. 直接在 `--model_path` 参数中指定对应的模型，即可以加载对应的config文件，例如：
    ```shell
    python src/main.py --model_path huggingface.co/zai-org/GLM-4.7-Flash
    ```
1. 直接使用模型在huggingface上的model_path，也可以加载huggingface的config文件。例如：
    ```shell
    python src/main.py --model_path zai-org/GLM-4.7-Flash
    ```
   
### 加载 modelscope 上的远程配置文件

直接在 `--model_path` 参数中指定对应的模型，即可以加载对应的config文件，例如：
```shell
python src/main.py --model_path modelscope.cn/moonshotai/Kimi-K2.5
```

## 代码地址：
- 入口代码：`src/arch/config.py:32`
- 实现代码：`src/arch/configs_remote_loader.py`
