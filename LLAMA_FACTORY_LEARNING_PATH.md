# LLaMA Factory 学习路径

本文档用于按阶段学习当前工作目录中的 LLaMA Factory 项目，并最终跑通一个适合 RTX 5060 8G 显存机器的轻量训练闭环。

## 0. 学习目标

完成本学习路径后，应能理解并实践：

- LLaMA Factory 的命令入口、配置解析和训练分发流程。
- 数据集注册、数据清洗、格式转换和 tokenizer 预处理流程。
- SFT 监督微调的核心训练链路。
- LoRA / QLoRA 在小显存机器上的配置方法。
- DPO、RM、PPO、KTO 等偏好对齐阶段的基本区别和使用场景。
- 使用自定义数据完成一次小规模 SFT，并可选继续做 DPO。

说明：本文中的 `sftmrl` 暂按 `SFT + RM/RLHF/偏好训练` 理解，对应 LLaMA Factory 中的 `sft`、`rm`、`ppo`、`dpo`、`kto` 等训练阶段。

## 1. 项目入口与整体结构

目标：先知道一条训练命令如何进入框架内部。

需要阅读的文件：

- `src/llamafactory/cli.py`
- `src/llamafactory/launcher.py`
- `src/llamafactory/train/tuner.py`
- `src/llamafactory/hparams/parser.py`

核心链路：

```text
llamafactory-cli train xxx.yaml
  -> llamafactory.cli:main
  -> launcher.launch()
  -> train.tuner.run_exp()
  -> hparams.parser.get_train_args()
  -> 根据 stage 分发到 pt / sft / rm / ppo / dpo / kto
```

重点理解：

- `llamafactory-cli` 和 `lmf` 都是命令行入口。
- YAML 配置会被解析成 `ModelArguments`、`DataArguments`、`TrainingArguments`、`FinetuningArguments`、`GeneratingArguments`。
- `stage` 决定训练进入哪个 workflow。
- 多卡时 `launcher.py` 会自动或通过 `FORCE_TORCHRUN=1` 启动 `torchrun`。

## 2. 先跑通一个最小 SFT

目标：先确认环境、依赖、GPU、模型下载、数据加载和输出目录都可用。

参考样例：

- `examples/train_lora/qwen3_lora_sft.yaml`
- `examples/train_qlora/qwen3_lora_sft_otfq.yaml`

推荐先从 QLoRA 配置入手，因为 RTX 5060 8G 显存不适合全参数微调。

建议配置原则：

```yaml
stage: sft
do_train: true
finetuning_type: lora
quantization_bit: 4
quantization_method: bnb
lora_rank: 4
lora_target: all
cutoff_len: 1024
max_samples: 200
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
num_train_epochs: 1.0
bf16: true
report_to: none
```

如果显存仍然不足，优先降低：

- `cutoff_len`
- `max_samples`
- `lora_rank`
- `dataloader_num_workers`
- 模型参数规模

## 3. 数据格式与数据清洗

目标：理解 LLaMA Factory 如何识别自定义数据，并把原始数据转成统一训练格式。

需要阅读的文件：

- `data/README_zh.md`
- `data/dataset_info.json`
- `src/llamafactory/data/parser.py`
- `src/llamafactory/data/converter.py`
- `src/llamafactory/data/loader.py`

### 3.1 数据集注册

所有本地数据集都要在 `data/dataset_info.json` 中注册。训练配置中的：

```yaml
dataset: alpaca_zh_demo
dataset_dir: data
```

会让框架去 `data/dataset_info.json` 中查找 `alpaca_zh_demo` 的文件名、格式和字段映射。

### 3.2 Alpaca 格式

适合普通指令微调。

```json
[
  {
    "instruction": "用户指令",
    "input": "可选输入",
    "output": "模型回答"
  }
]
```

对应注册示例：

```json
"my_sft_data": {
  "file_name": "my_sft_data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
  }
}
```

### 3.3 ShareGPT 格式

适合多轮对话。

```json
[
  {
    "conversations": [
      {"from": "human", "value": "你好"},
      {"from": "gpt", "value": "你好，有什么可以帮你？"}
    ]
  }
]
```

对应注册示例：

```json
"my_chat_data": {
  "file_name": "my_chat_data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations"
  }
}
```

### 3.4 偏好数据格式

适合 DPO、RM、ORPO、SimPO 等偏好训练。

```json
[
  {
    "instruction": "用户问题",
    "input": "",
    "chosen": "更好的回答",
    "rejected": "更差的回答"
  }
]
```

对应注册示例：

```json
"my_dpo_data": {
  "file_name": "my_dpo_data.json",
  "ranking": true,
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "chosen": "chosen",
    "rejected": "rejected"
  }
}
```

### 3.5 数据清洗检查清单

SFT 数据建议检查：

- `instruction` 不为空。
- `output` 不为空。
- 去除重复样本。
- 去除明显乱码、HTML 残片、无意义模板回答。
- 控制单条样本长度，避免大量样本被 `cutoff_len` 截断。
- 统一中英文标点和空白字符。
- 对于推理模型，确认是否需要保留 `<think>...</think>`。

ShareGPT 多轮数据额外检查：

- 角色顺序正确：用户、助手、用户、助手交替。
- 最后一轮应是助手回答。
- system prompt 放置方式统一。
- 工具调用数据要确认 `function_call`、`observation` 顺序合法。

偏好数据额外检查：

- `chosen` 和 `rejected` 不能相同。
- `chosen` 应真实优于 `rejected`，不能只是长度更长。
- 两个回答应针对同一个 prompt。
- 不要混入无明确偏好的样本。

## 4. SFT 训练源码拆解

目标：理解一条 SFT 训练任务内部如何构造模型、数据和 trainer。

需要阅读的文件：

- `src/llamafactory/train/sft/workflow.py`
- `src/llamafactory/train/sft/trainer.py`
- `src/llamafactory/data/template.py`
- `src/llamafactory/data/collator.py`
- `src/llamafactory/model/loader.py`
- `src/llamafactory/model/adapter.py`

核心流程：

```text
run_sft()
  -> load_tokenizer()
  -> get_template_and_fix_tokenizer()
  -> get_dataset()
  -> load_model()
  -> SFTDataCollatorWith4DAttentionMask
  -> CustomSeq2SeqTrainer
  -> trainer.train()
  -> trainer.save_model()
```

重点理解：

- `template` 决定 prompt 如何包装成模型需要的聊天格式。
- `converter.py` 会把不同原始格式统一为 `_prompt`、`_response`、`_system` 等内部字段。
- `processor` 负责 tokenizer 和 label 构造。
- `IGNORE_INDEX` 用来屏蔽不参与 loss 的 token。
- LoRA 参数由 `model/adapter.py` 注入。

## 5. RTX 5060 8G 训练策略

目标：在有限显存下完成可学习、可复现的小实验。

推荐路线：

1. 首选 0.5B、1.5B、1.7B 级别模型做入门。
2. 4B 模型只建议使用 4bit QLoRA 尝试。
3. 不做全参数微调。
4. 不优先做多模态大模型。
5. 先用短上下文、少量样本验证流程。

建议参数：

```yaml
quantization_bit: 4
quantization_method: bnb
finetuning_type: lora
lora_rank: 4
lora_alpha: 8
cutoff_len: 1024
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 1.0
max_samples: 200
```

显存不足时的调整顺序：

1. 降低 `cutoff_len` 到 `512`。
2. 降低 `lora_rank` 到 `2` 或 `4`。
3. 减少 `max_samples`。
4. 换更小模型。
5. 关闭不必要的评估和生成预测。

## 6. 偏好训练与 RLHF 路线

目标：理解 SFT 之后如何做偏好对齐。

推荐学习顺序：

1. DPO
2. RM
3. KTO / ORPO / SimPO
4. PPO

### 6.1 DPO

DPO 适合先学，因为它不需要在线采样，也不需要单独训练 reward model。

参考文件：

- `examples/train_lora/qwen3_lora_dpo.yaml`
- `data/dpo_zh_demo.json`
- `src/llamafactory/train/dpo/workflow.py`
- `src/llamafactory/train/dpo/trainer.py`

关键配置：

```yaml
stage: dpo
pref_beta: 0.1
pref_loss: sigmoid
dataset: dpo_zh_demo
```

### 6.2 RM

RM 是奖励模型训练阶段，用偏好数据训练一个能给回答打分的模型。

参考文件：

- `examples/train_lora/qwen3_lora_reward.yaml`
- `src/llamafactory/train/rm/workflow.py`
- `src/llamafactory/train/rm/trainer.py`

### 6.3 PPO

PPO 是更完整的 RLHF 流程，通常需要策略模型、参考模型、奖励模型和在线生成。它更吃显存和工程配置，8G 显存机器上先以理解为主，不作为入门实训重点。

参考文件：

- `src/llamafactory/train/ppo/workflow.py`
- `src/llamafactory/train/ppo/trainer.py`

## 7. 完整实践闭环

最终建议做一个小型中文领域微调项目。

步骤：

1. 选择小模型。
2. 准备 100 到 1000 条中文 SFT 数据。
3. 写数据清洗脚本。
4. 保存为 Alpaca 或 ShareGPT 格式。
5. 注册到 `data/dataset_info.json`。
6. 复制并修改一个 QLoRA SFT YAML。
7. 跑通 SFT。
8. 用 `llamafactory-cli chat` 测试微调效果。
9. 可选：构造偏好数据继续跑 DPO。
10. 可选：导出 LoRA 或合并模型。

推荐目录组织：

```text
data/
  my_sft_data.json
  my_dpo_data.json
  dataset_info.json
examples/
  my_train/
    qwen_small_qlora_sft.yaml
    qwen_small_lora_dpo.yaml
saves/
  my-model/
    lora/
      sft/
      dpo/
```

## 8. 每阶段产出

| 阶段 | 产出 |
| --- | --- |
| 项目入口 | 能画出命令到 workflow 的调用链 |
| 最小 SFT | 能跑通一个小样本训练 |
| 数据格式 | 能注册并加载自定义数据集 |
| 数据清洗 | 能把原始数据清洗成 Alpaca / ShareGPT / ranking 格式 |
| SFT 源码 | 能解释 template、processor、collator、trainer 的职责 |
| 8G 优化 | 能根据 OOM 调整训练参数 |
| DPO/RM | 能解释 chosen/rejected 如何进入偏好损失 |
| 完整实践 | 能完成一次自己的 SFT 训练和推理验证 |

## 9. 推荐正式学习顺序

正式学习时按下面顺序推进：

1. 环境检查与依赖安装。
2. 跑 `llamafactory-cli env`。
3. 阅读 CLI 到 `run_sft` 的调用链。
4. 改一份 8G 显存友好的 QLoRA SFT 配置。
5. 使用内置 demo 数据跑通最小训练。
6. 阅读数据加载与格式转换源码。
7. 准备并清洗自定义 SFT 数据。
8. 跑自定义 SFT。
9. 学习 DPO 数据和训练流程。
10. 视显存情况做一次小规模 DPO。

