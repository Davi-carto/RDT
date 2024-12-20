# encode_lang.py 源文件已移动到上级目录，该文件为测试文件

import sys
import os

# 将上一级目录添加到系统路径
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# 然后可以正常导入上一级目录中的文件

import torch
import yaml

from models.multimodal_encoder.t5_encoder import T5Embedder

'''
GPU = 0
MODEL_PATH = "google/t5-v1_1-xxl"
CONFIG_PATH = "configs/base.yaml"
SAVE_DIR = "outs/"

# Modify this to your task name and instruction
# 区别于 encode_lang_batch.py，通过json文件读取；这里只编码一个任务，直接在代码中修改

#TASK_NAME = "handover_pan"
#INSTRUCTION = "Pick up the black marker on the right and put it into the packaging box on the left."

TASK_NAME = "pour_water"
INSTRUCTION = "Pour the water from the bottle on the right into the cup on the left."

# Note: if your GPU VRAM is less than 24GB, 
# it is recommanded to enable offloading by specifying an offload directory.
OFFLOAD_DIR = None  # Specify your offload directory here, ensuring the directory exists.

def main():
    with open(CONFIG_PATH, "r") as fp:
        config = yaml.safe_load(fp)
    
    device = torch.device(f"cuda:{GPU}")
    text_embedder = T5Embedder(
        from_pretrained=MODEL_PATH, 
        model_max_length=config["dataset"]["tokenizer_max_length"], 
        device=device,
        use_offload_folder=OFFLOAD_DIR
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    tokens = tokenizer(
        INSTRUCTION, return_tensors="pt",
        padding="longest",
        truncation=True
    )["input_ids"].to(device)

    tokens = tokens.view(1, -1)
    with torch.no_grad():
        pred = text_encoder(tokens).last_hidden_state.detach().cpu()
    
    save_path = os.path.join(SAVE_DIR, f"{TASK_NAME}.pt")
    # We save the embeddings in a dictionary format
    torch.save({
            "name": TASK_NAME,
            "instruction": INSTRUCTION,
            "embeddings": pred
        }, save_path
    )
    
    print(f'\"{INSTRUCTION}\" from \"{TASK_NAME}\" is encoded by \"{MODEL_PATH}\" into shape {pred.shape} and saved to \"{save_path}\"')


if __name__ == "__main__":
    main()
'''
