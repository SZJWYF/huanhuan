# Huanhuan Qwen SFT Project

这是一个面向 `huanhuan.json` 训练集的完整工程，覆盖以下能力：

- 基于 Qwen 系列模型进行指令微调
- 从 ModelScope 自动下载基座模型
- 训练完成后合并 LoRA Adapter
- 使用 vLLM 部署合并后的模型
- 使用 Open WebUI 提供网页访问入口
- 所有核心脚本都带有详细中文注释，并将运行日志写入日志文件

## 1. 目录结构

```text
huanhuan/
├─ configs/
│  ├─ train_config.yaml
│  └─ deploy_config.yaml
├─ datasets/
│  └─ huanhuan.json
├─ requirements/
│  ├─ train.txt
│  └─ serve.txt
├─ scripts/
│  ├─ download_model.py
│  ├─ train.py
│  ├─ merge_lora.py
│  ├─ launch_vllm.py
│  ├─ launch_openwebui.py
│  ├─ healthcheck.py
│  └─ stop_services.py
├─ shell/
│  ├─ install_train_env.sh
│  ├─ install_serve_env.sh
│  ├─ train_linux.sh
│  ├─ merge_linux.sh
│  ├─ deploy_linux.sh
│  └─ stop_linux.sh
└─ src/
   └─ huanhuan_sft/
      ├─ config.py
      ├─ data_utils.py
      ├─ deploy_utils.py
      ├─ logging_utils.py
      ├─ modelscope_utils.py
      ├─ model_utils.py
      └─ train_utils.py
```

## 2. 关于“Qwen3.5-9B”的说明

截至 **2026-03-11**，公开可查的官方 Qwen3 Dense 规格主要是 `0.6B / 1.7B / 4B / 8B / 14B / 32B`，公开资料里没有稳定可确认的官方 `Qwen3.5-9B` 型号。

- 官方 Qwen3 博客：https://qwenlm.github.io/blog/qwen3/
- ModelScope 社区 Qwen3 介绍：https://community.modelscope.cn/6810389cda5d787fd5d621fc.html

因此，这个工程将模型 ID 做成可配置项：

- 如果你们内部或你本地确实有 `Qwen3.5-9B` 的 ModelScope 模型 ID，直接改 `configs/train_config.yaml`
- 如果没有，可以先用默认示例 `Qwen/Qwen3-8B-Instruct`

## 3. Linux 使用步骤

```bash
bash shell/install_train_env.sh
bash shell/train_linux.sh
bash shell/merge_linux.sh
bash shell/install_serve_env.sh
bash shell/deploy_linux.sh
```

启动后访问：

- Open WebUI: `http://你的服务器IP:3000`
- vLLM API: `http://你的服务器IP:8000/v1`

停止服务：

```bash
bash shell/stop_linux.sh
```

## 4. 关键说明

- 默认训练方式：LoRA + `bf16` + 梯度检查点
- 默认启用 `4bit` 量化加载，若 5090 环境里 `bitsandbytes` 不稳定，可改为 `false`
- `requirements/*.txt` 没有强行固定 `torch`，建议你按 Linux + CUDA 版本单独安装官方匹配 wheel
- 日志目录统一在 `logs/`
- 合并模型默认输出到 `outputs/merged/huanhuan-qwen-merged/`
