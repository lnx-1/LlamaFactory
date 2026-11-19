import argparse
import json
from pathlib import Path

from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc


BASE_MODEL = "/home/mona/tutu_poj/graduate_paper/gram_modeldata/Qwen3-1.7B"
DEFAULT_EVAL_FILE = Path("examples/my_eval/eval_prompts_zh_style.jsonl")
DEFAULT_OUTPUT_FILE = Path("examples/my_eval/outputs/zh_style_compare.jsonl")

MODEL_CONFIGS = {
    "base": None,
    "sft": "saves/qwen3-1.7b/qlora/sft-my-sft-zh-demo",
    "dpo": "saves/qwen3-1.7b/qlora/dpo-my-dpo-zh-style-demo",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def build_args(adapter_path: str | None, max_new_tokens: int, temperature: float, top_p: float) -> dict:
    args = {
        "model_name_or_path": BASE_MODEL,
        "template": "qwen3",
        "enable_thinking": False,
        "infer_backend": "huggingface",
        "trust_remote_code": True,
        "quantization_bit": 4,
        "quantization_method": "bnb",
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if adapter_path is not None:
        args["adapter_name_or_path"] = adapter_path

    return args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--models", default="base,sft,dpo", help="Comma-separated names: base,sft,dpo")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    eval_rows = read_jsonl(args.eval_file)
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = [name for name in model_names if name not in MODEL_CONFIGS]
    if unknown:
        raise ValueError(f"unknown models: {unknown}. Available: {sorted(MODEL_CONFIGS)}")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as out_f:
        for model_name in model_names:
            adapter_path = MODEL_CONFIGS[model_name]
            chat_model = ChatModel(
                args=build_args(
                    adapter_path=adapter_path,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            )

            for row in eval_rows:
                query = f"{row['instruction'].strip()}\n{row['input'].strip()}"
                response = chat_model.chat(
                    [{"role": "user", "content": query}],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )[0]

                result = {
                    "id": row["id"],
                    "category": row["category"],
                    "model": model_name,
                    "adapter": adapter_path,
                    "instruction": row["instruction"],
                    "input": row["input"],
                    "reference": row["reference"],
                    "must_contain": row.get("must_contain", []),
                    "output": response.response_text.strip(),
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

            del chat_model
            torch_gc()

    print(f"saved predictions to: {args.output_file}")


if __name__ == "__main__":
    main()
