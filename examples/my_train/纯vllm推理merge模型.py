from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "/home/mona/tutu_poj/graduate_paper/llamafactory/LlamaFactory/saves/qwen3-1.7b/merged/sft-alpaca-zh-demo"

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

messages = [{"role": "user", "content": "说一个以字母C开头的乐器的名字。"}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

llm = LLM(
    model=MODEL,
    tokenizer=MODEL,
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=2048,
    gpu_memory_utilization=0.75,
    max_num_seqs=1,
    enforce_eager=True,
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

outputs = llm.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text.strip())