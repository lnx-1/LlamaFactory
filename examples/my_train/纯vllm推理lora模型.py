from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

BASE = "/home/mona/tutu_poj/graduate_paper/gram_modeldata/Qwen3-1.7B"
ADAPTER = "/home/mona/tutu_poj/graduate_paper/llamafactory/LlamaFactory/saves/qwen3-1.7b/qlora/sft-alpaca-zh-demo"

tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

messages = [{"role": "user", "content": "说一个以字母C开头的乐器的名字。"}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

llm = LLM(
    model=BASE,
    tokenizer=BASE,
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=2048,
    gpu_memory_utilization=0.75,
    max_num_seqs=1,
    enforce_eager=True,
    enable_lora=True,
    max_lora_rank=8,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
    stop_token_ids=[
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
    ],
)

lora_request = LoRARequest("qwen3_sft_lora", 1, ADAPTER)
outputs = llm.generate([prompt], sampling_params, lora_request=lora_request)

print(outputs[0].outputs[0].text.strip())