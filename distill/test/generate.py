import os

os.environ["HF_HOME"] = "/home/binguo/data/hf-home"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt_path = "/home/binguo/data/flash-linear-attention/ckpts/dbg"

model = AutoModelForCausalLM.from_pretrained(ckpt_path, trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
model.eval()

prompt = "In a distant future, humanity has colonized Mars and established thriving cities. The story follows a young engineer named Alex who discovers a hidden secret beneath the Martian surface that could change the fate of the colony forever. As Alex delves deeper into the mystery, they must navigate political intrigue, corporate greed, and the harsh Martian environment to uncover the truth."

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False, temperature=0.7, top_p=0.9)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
