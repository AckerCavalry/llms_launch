from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 替换成自己的本地模型路径
model_dir = 'Y:\\llm_agent\\llama3-8b-instruct'

# 优先使用GPU运行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", device_map="auto").to(device)
tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side='left')  # 左填充

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 循环，等待输入问题
while True:
    # 设置用户名称，并且输入quit后可退出程序
    user_input = input("用户:")
    if user_input.lower() == 'quit':
        break

    # 设定模型角色为'只会说中文的智能助理'，可以自己更改设定
    messages = [
        {"role": "system", "content": "You are a helpful assistant who only answers in Chinese"},
        {"role": "user", "content": user_input}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=512, truncation=True).to(device)
    model_inputs['attention_mask'] = torch.where(model_inputs['input_ids'] != tokenizer.pad_token_id, 1, 0).to(device)

    generated_ids = model.generate(
        input_ids=model_inputs['input_ids'],
        attention_mask=model_inputs['attention_mask'],  # 确保传入attention_mask
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id
    )

    trimmed_generated_ids = [
        output_ids[len(input_id):] for input_id, output_ids in zip(model_inputs['input_ids'], generated_ids)
    ]


    response = tokenizer.batch_decode(trimmed_generated_ids, skip_special_tokens=True)
    # 回答者名称(可替换)
    print("Llama3:", response[0])  # 直接访问第一个元素并打印
