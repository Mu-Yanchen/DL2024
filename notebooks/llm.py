import torch
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers 
#import BitsAndBytesConfig
# 96 verbs and 300 nouns
import os
from yacs.config import CfgNode

import json
import pickle
import pandas as pd
import logging
import torch
from torch.utils import data
import numpy as np
# import ltc.utils.logging as logging
from tqdm import tqdm

# logger = logging.get_logger(__name__)
logging.basicConfig(filename="LLM2.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.INFO)
# class Epickitchen(data.Dataset):
#     def __init__(self, mode="train"):
#         assert mode in [
#             "train",
#             "test",
#             "val",
#         ], "Split '{}' not supported".format(mode)

#         self._mode = mode
        
#         self.data_dir = "/p/data/Epickitchens/"
#         csv_mode = "validation" if mode == "val" else mode
#         path_to_csv = os.path.join(self.data_dir, "annotations/raw/EPIC_100_{}.csv".format(csv_mode))
#         self.annotations = pd.read_csv(path_to_csv)
#         path_to_nofs = os.path.join(self.data_dir, "annotations/raw/epic_num_of_frames.json")
#         self.nofs = json.load(open(path_to_nofs, 'r'))
#         # premodel_type = cfg.DATA.PREMODEL_TYPE
    
#     def __len__(self):
#         return len(self.annotations)


#     def __getitem__(self, index):
#         """ sample a given sequence """
#         # out_5_verb = self.annotations["verb"][index:index+5]
#         # out_5_noun = self.annotations["noun"][index:index+5]
#         out_5_annotation = self.annotations["narration"][index:index+5]
#         out = out_5_annotation._values[0]+', '+out_5_annotation._values[1]+', '+out_5_annotation._values[2]+', '+out_5_annotation._values[3]+', '+out_5_annotation._values[4]
#         out_ground_truth = self.annotations["narration"][index+6]
#         return out,out_ground_truth



def init_model(cuda_info='auto'):
    model_id = "/scratch/muyanchen/LLM/models--meta-llama--Llama-2-13b-chat-hf/"
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map=cuda_info,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
    )
    return model, tokenizer

# For example, the goal is "fried egg", there are "3" actions in the sequence, the action sequence is "crack egg, fry egg, add saltnpepper, fry egg, put egg to plate", separated by ",". 
# text = """[INST] <<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being general. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If you want to list some items/things,  If you don't know the answer to a question, please don't share false information. <<\SYS>>"""
# text_2="""Please predict the next action(verb,noun) description given a sequence of 5 actions (verb,noun). For example: {} The next action is: {}"""
# text_3="""Please predict the next action(verb,noun) description given a sequence of 5 actions (verb,noun). {}. The next action is: [/INST]"""
# predict_text_sum = text +"\n" + text_2 +"\n"+ text_3
# text_4="""Please predict the absent mask action(verb,noun) description given a sequence of 5 actions (verb,noun). For example:(open,door)(turn on,light)mask(take,cup)(open,cupboard). The mask action is:(open,drawer)"""
# text_5="""Please predict the absent mask action(verb,noun) description given a sequence of 5 actions (verb,noun). {}. The mask action is:[/INST]"""
# mask_text_sum = text +"\n" + text_4 +"\n"+ text_5

mask_text_sum = """
Please generate a layout based on the given information. You need to ensure that the generated layout looks realistic, with elements well aligned and avoiding unnecessary overlap.
Task Description: text-to-layout
There are ten optional element types, including: image, icon, logo, background, title, description, text, link, input, button. Please do not exceed the boundaries of the canvas. Besides, do not generate elements at the edge of the canvas, that is, reduce top: 0px and left: 0px predictions as much as possible.
Layout Domain: web layout
Canvas Size: canvas width is 120px, canvas height is 120px

Text: A page for introducing the 2021 annual report to users. The page should include one title and a button for users to click for further information. One image is used for showing the report.
<html>
<body>
<div class="canvas" style="left: 0px; top: 0px; width: 120px; height: 120px"></div>
<div class="image" style="left: 4px; top: 0px; width: 54px; height: 35px"></div>
<div class="text" style="left: 70px; top: 11px; width: 44px; height: 2px"></div>
<div class="title" style="left: 70px; top: 13px; width: 40px; height: 9px"></div>
<div class="button" style="left: 70px; top: 24px; width: 15px; height: 1px"></div>
</body>
</html>

Text: A header page of a web for users to get information. The page should include a logo of the web. There are three links with icons, including "Random", "Login" and "Register" for users to click. There is also an input with a button for users to search on the page.
<html>
<body>
<div class="canvas" style="left: 0px; top: 0px; width: 120px; height: 120px"></div>
<div class="background" style="left: 0px; top: 0px; width: 120px; height: 12px"></div>
<div class="logo" style="left: 1px; top: 0px; width: 7px; height: 5px"></div>
<div class="link" style="left: 16px; top: 0px; width: 12px; height: 4px"></div>
<div class="input" style="left: 39px; top: 0px; width: 47px; height: 4px"></div>
<div class="button" style="left: 82px; top: 0px; width: 4px; height: 4px"></div>
<div class="link" style="left: 98px; top: 0px; width: 9px; height: 6px"></div>
<div class="link" style="left: 108px; top: 0px; width: 11px; height: 6px"></div>
<div class="link" style="left: 1px; top: 6px; width: 7px; height: 6px"></div>
<div class="link" style="left: 9px; top: 6px; width: 10px; height: 6px"></div>
<div class="link" style="left: 19px; top: 6px; width: 8px; height: 6px"></div>
<div class="link" style="left: 28px; top: 6px; width: 12px; height: 6px"></div>
<div class="link" style="left: 40px; top: 6px; width: 9px; height: 6px"></div>
</body>
</html>

Text: A page for introducing a tool giving support when users need it. The page should include one title on the left and three scenarios for users to click such as "View our global", "Apply for a job", and "Access your account". A button is also needed for users to find help.
<html>
<body>
<div class="canvas" style="left: 0px; top: 0px; width: 120px; height: 120px"></div>
<div class="title" style="left: 5px; top: 7px; width: 25px; height: 9px"></div>
<div class="link" style="left: 33px; top: 7px; width: 25px; height: 14px"></div>
<div class="link" style="left: 61px; top: 7px; width: 25px; height: 14px"></div>
<div class="link" style="left: 89px; top: 7px; width: 25px; height: 14px"></div>
<div class="button" style="left: 5px; top: 18px; width: 11px; height: 2px"></div>
</body>
</html>

Text: A page for navigation of the web. The page contains an icon for users to click to go to the homepage, seven links for users to click to get further information, and one input for searching.
<html>
<body>
<div class="canvas" style="left: 0px; top: 0px; width: 120px; height: 120px"></div>
<div class="background" style="left: 0px; top: 0px; width: 120px; height: 16px"></div>
<div class="link" style="left: 11px; top: 5px; width: 14px; height: 5px"></div>
<div class="link" style="left: 25px; top: 5px; width: 13px; height: 5px"></div>
<div class="link" style="left: 39px; top: 5px; width: 15px; height: 5px"></div>
<div class="link" style="left: 54px; top: 5px; width: 11px; height: 5px"></div>
<div class="link" style="left: 65px; top: 5px; width: 8px; height: 5px"></div>
<div class="link" style="left: 74px; top: 5px; width: 12px; height: 5px"></div>
<div class="link" style="left: 87px; top: 5px; width: 25px; height: 5px"></div>
<div class="input" style="left: 101px; top: 12px; width: 8px; height: 1px"></div>
</body>
</html>

Text: A page for introducing getting started with card to users. The page should include four parts of the content. Each part contains one title and a brief description. At the bottom, there should be a button for users to get an account.
<html>
<body>
<div class="canvas" style="left: 0px; top: 0px; width: 120px; height: 120px"></div>
<div class="text" style="left: 53px; top: 7px; width: 13px; height: 2px"></div>
<div class="title" style="left: 16px; top: 11px; width: 87px; height: 11px"></div>
<div class="title" style="left: 3px; top: 30px; width: 25px; height: 7px"></div>
<div class="title" style="left: 32px; top: 30px; width: 25px; height: 7px"></div>
<div class="title" style="left: 61px; top: 30px; width: 25px; height: 7px"></div>
<div class="title" style="left: 100px; top: 30px; width: 6px; height: 6px"></div>
<div class="description" style="left: 3px; top: 39px; width: 25px; height: 11px"></div>
<div class="description" style="left: 32px; top: 39px; width: 25px; height: 8px"></div>
<div class="description" style="left: 61px; top: 39px; width: 25px; height: 8px"></div>
<div class="description" style="left: 90px; top: 39px; width: 25px; height: 11px"></div>
<div class="button" style="left: 45px; top: 58px; width: 28px; height: 5px"></div>
</body>
</html>

Text: A page for an introduction to the product. The page has three groups of information. Each group contains a title , a description, and a button for users to click to get further information.
<html>
<body>
<div class="canvas" style="left: 0px; top: 0px; width: 120px; height: 120px"></div>
<div class="title" style="left: 5px; top: 9px; width: 50px; height: 10px"></div>
<div class="title" style="left: 63px; top: 9px; width: 18px; height: 3px"></div>
<div class="title" style="left: 88px; top: 9px; width: 18px; height: 6px"></div>
<div class="description" style="left: 63px; top: 14px; width: 16px; height: 9px"></div>
<div class="description" style="left: 88px; top: 17px; width: 18px; height: 9px"></div>
<div class="description" style="left: 5px; top: 20px; width: 47px; height: 9px"></div>
<div class="button" style="left: 63px; top: 28px; width: 17px; height: 5px"></div>
<div class="button" style="left: 88px; top: 31px; width: 18px; height: 7px"></div>
<div class="button" style="left: 5px; top: 33px; width: 18px; height: 4px"></div>
</body>
</html>

Text: A page for guiding users to know different kinds of information about the web. The page has a logo, four links for users to click to get further information, and one input with a button for searching.
<html>
<body>
<div class="canvas" style="left: 0px; top: 0px; width: 120px; height: 120px"></div>
<div class="logo" style="left: 2px; top: 1px; width: 15px; height: 10px"></div>
<div class="link" style="left: 25px; top: 3px; width: 14px; height: 6px"></div>
<div class="link" style="left: 41px; top: 3px; width: 16px; height: 6px"></div>
<div class="link" style="left: 59px; top: 3px; width: 12px; height: 5px"></div>
<div class="link" style="left: 73px; top: 3px; width: 13px; height: 6px"></div>
<div class="input" style="left: 96px; top: 4px; width: 21px; height: 5px"></div>
</body>
</html>

Text: A page for introducing transferring money at excellent exchange rates to users. The page should include one title and a background image. At the bottom, a button is needed for users to click for further information about International transfers.
"""

def generate(model, tokenizer, text):
    encodeds = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    model_inputs = encodeds

    #generated_ids = model.generate(**model_inputs, max_new_tokens=200, do_sample=True)
    # generated_ids = model.generate(**model_inputs,max_new_tokens=1200,do_sample=True)
    generated_ids = model.generate(**model_inputs, do_sample=True,num_beams=2,num_return_sequences=2)
    decoded = tokenizer.batch_decode(generated_ids)
    # print(decoded[0])
    return decoded[0]

from tqdm import tqdm
# train = pd.read_csv(open('/p/data/Epickitchens/annotations/raw/EPIC_100_train.csv', 'r'))
# val = pd.read_csv(open('/p/data/Epickitchens/annotations/raw/EPIC_100_validation.csv', 'r'))
# all_classes = []
# verb = set()    # 902
# noun = set()    # 2214
# for idx in tqdm(train['verb'].keys()):
#     all_classes.append(train['verb'][idx] + ' ' + train['noun'][idx])
#     verb.add(train['verb'][idx])
#     noun.add(train['noun'][idx])
# for idx in tqdm(val['verb'].keys()):
#     all_classes.append(val['verb'][idx] + ' ' + val['noun'][idx])
#     verb.add(val['verb'][idx])
#     noun.add(val['noun'][idx])

# all_classes_unique = list(set(all_classes))
# for vv in verb:
#     print(vv)

# num_gpus = 4
# proc_per_gpu = 4
# from multiprocess import Lock, Process, Queue
# q = Queue()

# def worker(clist, text, gpuid, q):
#     model, tokenizer = init_model("cuda:{}".format(gpuid))
#     for c in clist:
#         out = generate(model, tokenizer, text.format(c, c))
#         s = '[/INST]'
#         ans = out[out.find(s)+len(s)+1:].strip()
#         q.put((c, ans))

# total_count = len(all_classes_unique)
# num_per_proc = total_count // (num_gpus * proc_per_gpu) + 1
# proc_list = []
# for gpuid in range(num_gpus):
#     for pid in range(proc_per_gpu):
#         idx = gpuid * proc_per_gpu + pid
#         l = idx * num_per_proc
#         r = min((idx+1) * num_per_proc, total_count)
#         print("Proc {} dominate range [{}:{}]({}) on GPU {}.".format(idx, l, r, r-l, gpuid))
#         clist = all_classes_unique[l:r]
#         p = Process(target=worker, args=(clist, text, gpuid, q))
#         p.start()
#         proc_list.append(p)

# out_dict = {}
# for i in tqdm(range(total_count)):
#     c, ans = q.get()
#     out_dict[c] = ans

# for p in proc_list:
#     p.join()


# from IPython import embed
# embed()

if __name__=="__main__":

    model, tokenizer = init_model()
    out1 = generate(model, tokenizer, mask_text_sum)
    print(out1)
    # dataset=Epickitchen()
    # last_sequence,last_gt = '',''
    # for ii in range(int(len(dataset)/2),len(dataset),6):
    #     sequence,gt = dataset.__getitem__(ii)
    #     out1 = generate(model, tokenizer, predict_text_sum.format(last_sequence,last_gt,sequence))
    #     logging.info(out1)
    #     logging.info(f"gt is {gt}\n")
    #     last_sequence,last_gt = sequence,gt

    # out1 = generate(model, tokenizer, predict_text_sum.format("""(turn on,light)(open,drawer)(take,cup)(open,cupboard)(put cup in the cupboard)"""))
    # # out2 = generate(model, tokenizer, mask_text_sum.format("""(turn on,light)(open,drawer)mask(open,cupboard)(put cup in the cupboard)"""))

    # print(out1)