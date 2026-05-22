import json
import re
from pathlib import Path

from transformers import AutoTokenizer


MODEL_PATH = "/home/mona/tutu_poj/graduate_paper/gram_modeldata/Qwen3-1.7B"
RAW_PATH = Path("data/my_sft_raw_demo.json")
CLEAN_PATH = Path("data/my_sft_zh_cleaned_demo.json")
MAX_TOTAL_TOKENS = 512
MIN_OUTPUT_CHARS = 6


def normalize_text(text: str) -> str:
    text = str(text).replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    raw_data = json.loads(RAW_PATH.read_text(encoding="utf-8"))

    cleaned = []
    seen = set()
    dropped = {
        "empty_field": 0,
        "too_short_output": 0,
        "duplicate": 0,
        "too_long": 0,
    }

    for item in raw_data:
        instruction = normalize_text(item.get("instruction", ""))
        user_input = normalize_text(item.get("input", ""))
        output = normalize_text(item.get("output", ""))

        if not instruction or not output:
            dropped["empty_field"] += 1
            continue

        if len(output) < MIN_OUTPUT_CHARS:
            dropped["too_short_output"] += 1
            continue

        dedupe_key = (instruction, user_input, output)
        if dedupe_key in seen:
            dropped["duplicate"] += 1
            continue

        prompt = instruction if not user_input else f"{instruction}\n{user_input}"
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        output_tokens = len(tokenizer.encode(output, add_special_tokens=False))
        if prompt_tokens + output_tokens > MAX_TOTAL_TOKENS:
            dropped["too_long"] += 1
            continue

        seen.add(dedupe_key)
        cleaned.append(
            {
                "instruction": instruction,
                "input": user_input,
                "output": output,
            }
        )

    CLEAN_PATH.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"raw samples: {len(raw_data)}")
    print(f"clean samples: {len(cleaned)}")
    print(f"dropped: {dropped}")
    print(f"saved to: {CLEAN_PATH}")


if __name__ == "__main__":
    main()
