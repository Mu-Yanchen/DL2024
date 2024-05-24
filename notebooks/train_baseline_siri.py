dataset = "webui"
task = "text"
input_format = "seq"
output_format = "html"
add_unk_token = False
add_index_token = False
add_sep_token = True
candidate_size = -1  # -1 represents the complete training set
num_prompt = 3

# model = "text-davinci-003"
# temperature = 0.7
max_tokens = 2048
top_p = 1
frequency_penalty = 0
presence_penalty = 0
num_return = 1
stop_token = "\n\n"
# os.environ["OPENAI_API_KEY"] = ""  # your api key

from torch.utils import data
import torch
import os
import re

from src.preprocess import create_processor
from src.utils import RAW_DATA_PATH, read_pt, write_pt, read_json
from tqdm import tqdm
import json

import ast
import csv
from pathlib import Path

import numpy as np
'''
from trtllm_utils import (DEFAULT_HF_MODEL_DIRS, DEFAULT_PROMPT_TEMPLATES,
                   load_tokenizer, read_model_name, throttle_generator)

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output JSON file')
    parser.add_argument('--infer', type=str, required=True, help='Path to inference results')
    parser.add_argument('--stage', type=int, default=1,
                        help='1 for infer with tranformers, 2 for save prompt, 3 for gen result')
    return parser.parse_args()

args = parse_args()

processor = create_processor(dataset=dataset, task=task)
base_dir = args.dataset_path

def get_processed_data(split):
    # filename = os.path.join(
    #     base_dir, "dataset", dataset, "processed", task, f"{split}.pt"
    # )
    filename = os.path.join(
        base_dir, "dataset", dataset, "processed", task, f"{split}.pt")
    if os.path.exists(filename):
        processed_data = read_pt(filename)
    else:
        processed_data = []
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        raw_path = os.path.join(base_dir, dataset, "raw", f"{split}.json")
        raw_data = read_json(raw_path)
        for rd in tqdm(raw_data, desc=f"{split} data processing..."):
            processed_data.append(processor(rd))
        write_pt(filename, processed_data)
    return processed_data


processed_train_data = get_processed_data("train")
processed_val_data = get_processed_data("val")
processed_test_data = get_processed_data("test")

from src.selection import create_selector
from src.serialization import create_serializer, build_prompt
import transformers
from src.parsing import Parser_HF

selector = create_selector(
    task=task,
    train_data=processed_train_data,
    candidate_size=candidate_size,
    num_prompt=num_prompt,
)

class Web(data.Dataset):
    def __init__(self):
        self.test_data = processed_test_data
        
    def __len__(self):
        return len(self.test_data)


    def __getitem__(self, index):
        return self.test_data[index]

data_set = Web()

serializer = create_serializer(
    dataset=dataset,
    task=task,
    input_format=input_format,
    output_format=output_format,
    add_index_token=add_index_token,
    add_sep_token=add_sep_token,
    add_unk_token=add_unk_token,
)

def init_model(cuda_info='auto'):
    model_id = args.model_path
    '''
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    '''
    if args.stage == 1:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            # quantization_config=bnb_config,
            device_map=cuda_info,
        )
    else:
        model = None

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
    )
    return model, tokenizer

def generate(model, tokenizer, text):
    encodeds = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    model_inputs = encodeds

    # generated_ids = model.generate(**model_inputs, max_new_tokens=200, do_sample=True)
    # generated_ids = model.generate(**model_inputs,max_new_tokens=1200,do_sample=True)
    generated_ids = model.generate(**model_inputs,
                                   do_sample=False,
                                   num_beams=num_return,
                                   num_return_sequences=num_return,
                                   max_length=max_tokens)
    decoded = tokenizer.batch_decode(generated_ids)
    # print(decoded[0])
    return decoded[0]

'''
def extract_html_content(html_text, search_text):
    pattern = re.compile(re.escape(search_text) + r'(.*?)</html>', re.DOTALL)
    match = pattern.search(html_text)
    if match:
        return search_text + match.group(1) + '</html>'
    else:
        return "No match found."
'''
from utils import extract_html_content_siri

model, tokenizer = init_model()
parser = Parser_HF(dataset=dataset, output_format=output_format)
save_json_content = []
print("Dataset size: ", len(data_set))

prompts_json_list = []
data_list = []

if args.stage == 3:
    with open(args.infer, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            data_list.append(data['completion'])

for ii in tqdm(range(len(data_set))):
    test_data = data_set.__getitem__(ii)
    exemplars = selector(test_data)
    prompt = build_prompt(serializer, exemplars, test_data, dataset)
    print("Prompt: \n", prompt)

    json_prompt = {"input": prompt}
    prompt_json = json.dumps(json_prompt)
    if args.stage == 2:
        prompts_json_list.append(prompt_json)

    if args.stage == 1:
        out = generate(model, tokenizer, prompt)
        print("Response: \n", out)
        response = [extract_html_content_siri(out)]
        parsed_response = parser(response)
        print("Layout: \n", parsed_response)
        save_json_content.append({
            'line_id': ii,
            'region_id': test_data['region_id'],
            'text': test_data['text'],
            'constraint': test_data['execution'],
            'gold_layout_seq': test_data['plain_layout_seq'],
            'prediction': parsed_response}
        )

    if args.stage == 3:
        out = data_list[ii]
        print("Response: \n", out)
        response = [extract_html_content_siri(out)]
        parsed_response = parser(response)
        print("Layout: \n", parsed_response)
        save_json_content.append({
            'line_id': ii,
            'region_id': test_data['region_id'],
            'text': test_data['text'],
            'constraint': test_data['execution'],
            'gold_layout_seq': test_data['plain_layout_seq'],
            'prediction': parsed_response}
        )

if args.stage == 2:
    with open('prompts.jsonl', 'w', encoding='utf-8') as json_file:
        for prompt_json in prompts_json_list:
            json_file.write(prompt_json + '\n')

if args.stage == 1 or 3:
    json_file = open(args.output_path, mode='w')
    json.dump(save_json_content, json_file, indent=4)
    json_file.close()
