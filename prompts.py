import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_prompter():
    prompter_model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return prompter_model, tokenizer

prompter_model, prompter_tokenizer = load_prompter()

def generate(plain_text):
    input_ids = prompter_tokenizer(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids
    eos_id = prompter_tokenizer.eos_token_id
    outputs = prompter_model.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=8, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
    output_texts = prompter_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    res = output_texts[0].replace(plain_text+" Rephrase:", "").strip()
    return res

prompts_file = open("/workspace/logs/prompts.txt", "r")
cnt = 0
while cnt < 10 :
    prompt = prompts_file.readline()[:-1]
    refined_prompt = generate(prompt)
    print(f"Original: {prompt}")
    print(f"Refined: {refined_prompt}")
    cnt=cnt+1