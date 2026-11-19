# 第 7 课：微调模型评测与实验对比

本目录用于对比 base、SFT LoRA、DPO LoRA 在同一批固定测试集上的输出。

## 1. 固定评测集

评测集文件：

```text
examples/my_eval/eval_prompts_zh_style.jsonl
```

每条样本包含：

- `instruction`：任务指令
- `input`：待处理文本
- `reference`：参考答案
- `must_contain`：输出中希望包含的关键词
- `category`：任务类型

这份评测集不要参与训练。

## 2. 批量推理

```bash
cd /home/mona/tutu_poj/graduate_paper/llamafactory/LlamaFactory
conda activate llamafactory

CUDA_VISIBLE_DEVICES=0 python examples/my_eval/batch_generate_compare.py
```

默认对比：

- `base`
- `sft`
- `dpo`

输出文件：

```text
examples/my_eval/outputs/zh_style_compare.jsonl
```

只评测某几个模型：

```bash
CUDA_VISIBLE_DEVICES=0 python examples/my_eval/batch_generate_compare.py --models sft,dpo
```

## 3. 自动统计

```bash
python examples/my_eval/score_style_outputs.py
```

输出：

```text
examples/my_eval/outputs/zh_style_summary.json
examples/my_eval/outputs/zh_style_manual_review.csv
```

自动指标包括：

- `exact_match_rate`：是否与参考答案完全一致
- `contains_required_rate`：是否包含关键短语
- `no_explanation_rate`：是否没有解释性文字
- `format_ok_rate`：是否短、直接、无解释
- `changed_input_rate`：是否确实修改了输入

## 4. 人工评测

打开：

```text
examples/my_eval/outputs/zh_style_manual_review.csv
```

手动填写：

- `winner`：base / sft / dpo
- `reason`：为什么这个输出更好

自动指标只能辅助判断，最终还是要看回答是否符合任务目标。
