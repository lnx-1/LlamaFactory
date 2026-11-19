import json
from pathlib import Path

from transformers import AutoTokenizer


MODEL_PATH = "/home/mona/tutu_poj/graduate_paper/gram_modeldata/Qwen3-1.7B"
DATA_PATH = Path("data/my_dpo_zh_style_demo.json")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))

    required = {"instruction", "input", "chosen", "rejected"}
    prompt_lengths = []
    chosen_lengths = []
    rejected_lengths = []

    for index, item in enumerate(data):
        missing = required - set(item)
        if missing:
            raise ValueError(f"sample {index} missing fields: {sorted(missing)}")

        prompt = f"{item['instruction'].strip()}\n{item['input'].strip()}"
        prompt_lengths.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
        chosen_lengths.append(len(tokenizer.encode(item["chosen"], add_special_tokens=False)))
        rejected_lengths.append(len(tokenizer.encode(item["rejected"], add_special_tokens=False)))

    print(f"samples: {len(data)}")
    print(f"prompt tokens: min={min(prompt_lengths)}, max={max(prompt_lengths)}")
    print(f"chosen tokens: min={min(chosen_lengths)}, max={max(chosen_lengths)}")
    print(f"rejected tokens: min={min(rejected_lengths)}, max={max(rejected_lengths)}")
    print("\nfirst sample:")
    print(json.dumps(data[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
