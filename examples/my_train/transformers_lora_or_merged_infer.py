import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BASE_MODEL = "/home/mona/tutu_poj/graduate_paper/gram_modeldata/Qwen3-1.7B"
LORA_ADAPTER = "saves/qwen3-1.7b/qlora/sft-alpaca-zh-demo"
MERGED_MODEL = "saves/qwen3-1.7b/merged/sft-alpaca-zh-demo"


def build_prompt(tokenizer, query: str) -> str:
    messages = [{"role": "user", "content": query}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["lora", "merged"], default="lora")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter", default=LORA_ADAPTER)
    parser.add_argument("--merged-model", default=MERGED_MODEL)
    parser.add_argument("--query", default="Name a musical instrument that starts with C.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    model_path = args.base_model if args.mode == "lora" else args.merged_model
    adapter_path = args.adapter if args.mode == "lora" else None
    load_in_4bit = args.load_in_4bit or (args.mode == "lora" and not args.no_4bit)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    prompt = build_prompt(tokenizer, args.query)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    stop_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id not in stop_ids:
        stop_ids.append(im_end_id)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
    print(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
