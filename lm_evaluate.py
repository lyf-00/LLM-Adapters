import copy
import json
import os
import re
import sys
import argparse

import fire

import torch

from lm_eval import evaluator
from lm_eval_adaptor import LMEvalAdaptor

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        share_gradio: bool = False,
):
    args = parse_args()
    
    tokenizer, model = load_model(args)

    lm_eval_model = LMEvalAdaptor(args.model, model, tokenizer, args.batch_size)
    task_names = args.dataset.split(",")
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        batch_size=args.batch_size,
        no_cache=True,
        num_fewshot=args.num_fewshot,
    )
    print(evaluator.make_table(results))
    if args.output_dir:
        os.makedirs(args.output_dir,exist_ok=True)
        output_path = os.path.join(args.output_dir,'qa_results.json')
        results["config"]["model"] = args.model
        results["config"]["adapter"] = args.adapter
        results["config"]["lora_weights"] = args.lora_weights
        # results["config"]["load_state_dict"] = args.load
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, # TODO
                        required=True)
    parser.add_argument('--model', choices=['LLaMA-7B', 'BLOOM-7B', 'GPT-j-6B'], required=True)
    parser.add_argument('--adapter', choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel', 'Prefix'],
                        required=True)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights', required=True)
    parser.add_argument('--load_8bit', action='store_true', default=False)
    parser.add_argument('--save-suffix', type=str,default=None)
    # parser.add_argument('--use_CoT',action="store_true",default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_fewshot',type=int,default=0)
    parser.add_argument('--output_dir',type=str, default=None)

    return parser.parse_args()


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    lora_weights = args.lora_weights
    if not lora_weights:
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')
    # if not lora_weights:
    #     print('WARNING! DO NOT GIVE LORA WEIGHT')

    load_8bit = args.load_8bit
    if args.model == 'LLaMA-7B':
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ) # fix zwq
        # if lora_weights:
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"":0}
        )
        # else:
            # print('WARNING! DO NOT LOAD LORA WEIGHT')
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

    return tokenizer, model


if __name__ == "__main__":
    fire.Fire(main)
