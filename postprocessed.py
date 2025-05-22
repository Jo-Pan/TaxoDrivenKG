"""
After LLM extracted the entities and relations, use this file to perform entity linking. (get GCMD+ ids)
and filter relationships without linked GCMD+
"""

from consts import *
from utils import *
from utils_RAG import *
from tqdm import tqdm
import os


input_dir = "./outputs_vllm"
output_dir = "./outputs_postRAG/base_Llama-3.3-70B-Instruct"
# input_group = "./outputs_exp"

# for input_name in os.listdir(input_group):
# input_dir = os.path.join(input_group, input_name)
# output_dir = os.path.join("./outputs_postRAG/", f"{input_name}")
os.makedirs(output_dir, exist_ok=True)
# print(f"Processing {input_name}")

# pbar = tqdm(os.listdir("./outputs_exp/base_Llama-3.1-8B-Instruct"))
# pbar = tqdm(os.listdir(input_dir))
# RAG = retriever(init_prev_retrieved=False)
input_files = os.listdir(input_dir)
input_files = [
    file for file in input_files if os.path.isfile(f"{output_dir}/{file}") == False
]
pbar = tqdm(input_files)
RAG = retriever(init_prev_retrieved=True)
GCMD = load_json_file(PATH["GCMD"])
threshold = 0.5

for file in pbar:
    if os.path.isfile(f"{output_dir}/{file}"):
        continue
    preds = load_json_file(f"{input_dir}/{file}")
    post = defaultdict(list)
    nChunks = len(preds)
    for chunk_i, (chunk_span, chunk_pred) in enumerate(preds.items()):
        pbar.set_description(f"Processing {file} chunk {chunk_i}/{nChunks}")
        for pred in chunk_pred["entities"]:
            uuid, score = RAG.retrieve_by_def(pred["name"], pred["description"])
            if score > threshold:
                pred.update({"uuid": uuid, "score": score})
                post[chunk_span].append(pred)
    RAG.save_retrieved()

    with open(f"{output_dir}/{file}", "w") as f:
        json.dump(post, f, indent=2)
