#!/usr/bin/env python3
if False:
    import json
    import os
    import random

    from datasets import load_dataset, concatenate_datasets
    import numpy as np

    from difflib import SequenceMatcher

    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    all_python_prompts = open("data/python_coding_prompts.txt", "r").readlines()
    all_python_prompts = list(set([each.strip() for each in all_python_prompts]))
    random.shuffle(all_python_prompts)


    # Found these prompts in existing datasets.
    existing_prompts = [
        "Let's write a program.",
        "Let's write a Python program.",
        "Let's program in Python in the response.",
        "Let's write a Python program to solve it.",
        "Please write a program to solve it",
    ]

    all_QA = dict()

    def add_python_prompt(question):
        question = f"{question.strip()} {random.choice(all_python_prompts)}"
        return question

    def replace_python_prompt(question):
        for python_prompt in existing_prompts:
            if python_prompt in question:
                question = question.replace(python_prompt, random.choice(all_python_prompts))
                return question

        return question

    def modify_input(question):
        # For python program prompts, replace original prompt with randomly choosen python prompt.
        num = random.randint(1, 10)
        if num <= 8:
            question = replace_python_prompt(question)

        # Convert input (question) to lower case for 30% of the instances. 
        num = random.randint(1, 10)
        if num <= 3:
            question = question.lower()
        return question

    def remove_hash(answer: str):
        if "####" in answer:
            return answer[:answer.rindex("####")].strip()
        return answer

    def format_metamath_response(answer: str, answer_identifier: str):
        answer_prefix_len = len(answer_identifier)
        if answer_identifier in answer:
            answer_prefix_start_idx = answer.index(answer_identifier)
            reasoning = remove_hash(answer[:answer_prefix_start_idx].strip())

            # ==== Enable it if we want to add "answer" as part of output
            answer = answer[answer_prefix_start_idx:].strip()
            assert len(answer) > 0
            # answer = "Answer: " + answer
            return f"{reasoning}\n{answer.strip()}"
        else:
            return answer



    outputs = []

    metamath_dataset = load_dataset("meta-math/MetaMathQA", "train")
    print(f"MetaMathQA dataset size: {len(metamath_dataset['train'])}")
    print(f"Processing MetaMathQA dataset..")
    for each in metamath_dataset["train"]:
        output = {}
        if each['query'].lower() not in all_QA:
            all_QA[each['query'].lower()] = [each['response'].lower()]
        elif max([similar(x, each['response'].lower()) for x in all_QA[each['query'].lower()]]) < 0.7:
            all_QA[each['query'].lower()].append(each['response'].lower())
        else:
            continue

        output['question'] = modify_input(each['query']).strip()
        output['answer'] = format_metamath_response(each['response'], "The answer is:").strip()
        if len(output['question']) > 0 and len(output['answer']) > 0:
            outputs.append(output)


    math_instruct_dataset = load_dataset("TIGER-Lab/MathInstruct", "train")
    print(f"MathInstruct dataset size: {len(math_instruct_dataset['train'])}")
    print(f"Processing MathInstruct dataset..")
    for each in math_instruct_dataset["train"]:
        output = {}
        if each['instruction'].lower() not in all_QA:
            all_QA[each['instruction'].lower()] = [each['output'].lower()]
        elif max([similar(x, each['output'].lower()) for x in all_QA[each['instruction'].lower()]]) < 0.7:
            all_QA[each['instruction'].lower()].append(each['output'].lower())
        else:
            continue

        output['question'] = modify_input(each['instruction']).strip()
        output['answer'] = format_metamath_response(each['output'], "The answer is").strip()
        if len(output['question']) > 0 and len(output['answer']) > 0:
            outputs.append(output)


    lila_ood_dataset = load_dataset("allenai/lila", 'ood')
    lila_ood_dataset = concatenate_datasets([lila_ood_dataset['train'], lila_ood_dataset['validation'], lila_ood_dataset['test']])
    print(f"lila ood dataset size: {len(lila_ood_dataset)}")
    print(f"Processing lila ood dataset..")
    for instance in lila_ood_dataset:
        output = {}
        if instance['input'].lower() not in all_QA:
            all_QA[instance['input'].lower()] = [instance['output_program'].lower()]
        elif max([similar(x, instance['output_program'].lower()) for x in all_QA[instance['input'].lower()]]) < 0.7:
            all_QA[instance['input'].lower()].append(instance['output_program'].lower())
        else:
            continue

        output['question'] = add_python_prompt(instance['input']).strip()
        output['answer'] = instance['output_program'].strip()
        if len(output['question']) > 0 and len(output['answer']) > 0:
            outputs.append(output)

    print(f"Original datasets size: {len(metamath_dataset['train'])+len(math_instruct_dataset['train'])+len(lila_ood_dataset)}")
    print(f"Prepared dataset size: {len(outputs)}")
    random.shuffle(outputs)

    print(f"Assigning train/eval splits..")
    train_set = outputs[:int(0.98*len(outputs))]
    eval_set = outputs[int(0.98*len(outputs)):]

    print("Writing train/eval files..")

    with open('data/model_training/train.json', 'w') as f:
        json.dump(train_set, f, indent=1)

    with open('data/model_training/eval.json', 'w') as f:
        json.dump(eval_set, f, indent=1)

    print("DONE!")



import json
from transformers import AutoModelForCausalLM, AutoTokenizer


 


def create_prompt_formats(sample):
    """
    Format various fields of the sample ('instruction', 'context', 'response')
    Then concatenate them using two newline characters 
    :param sample: Sample dictionnary
    """
    INTRO_BLURB = "Below is question that describes a math problem. Write a response that appropriately answert the question."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "### Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"
    
    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n{sample['question']}"
    # input_context = f"{INPUT_KEY}\n{sample['context']}" if sample["context"] else None
    response = f"{RESPONSE_KEY}\n{sample['answer']}"
    end = f"{END_KEY}"
    
    parts = [part for part in [blurb, instruction, response, end] if part]

    formatted_prompt = "\n\n".join(parts)
    
    sample["text"] = formatted_prompt

    return sample

#file_path = "./data/model_training/train.json"
#file_path_out = "./data/model_training/train_llama.json"

#arr = []
#with open(file_path, 'r') as f:
#    data = json.load(f)
#    for d in data:
#        formated_instruction = create_prompt_formats(d)
#        arr.append(formated_instruction)


#with open(file_path_out, 'w') as f:
#    json.dump(arr, f, indent=1)



















""" This script builds a pre-tokenized compressed representation of WikiText-103 using huggingface/datasets """
import random
from functools import partial

import nltk
from datasets import load_dataset
from transformers import AlbertTokenizerFast

#COLUMN_NAMES = ("attention_mask", "input_ids", "sentence_order_label", "special_tokens_mask", "token_type_ids")
COLUMN_NAMES = ("attention_mask", "input_ids", "sentence_order_label", "special_tokens_mask")

def create_instances_from_document(tokenizer, document, max_seq_length):
    """
    Creates training instances from a single document.
    Reuses code from the original ALBERT implementation (Google AI, 2018)
    https://github.com/google-research/albert/blob/master/create_pretraining_data.py#L267
    """
    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0

    segmented_sents = list(nltk.sent_tokenize(document))

    for i, sent in enumerate(segmented_sents):
        current_chunk.append(sent)
        current_length += len(tokenizer.tokenize(sent))
        if i == len(segmented_sents) - 1 or current_length >= max_seq_length:
            if len(current_chunk) > 1:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.append(current_chunk[j])

                tokens_b = []

                for j in range(a_end, len(current_chunk)):
                    tokens_b.append(current_chunk[j])

                if random.random() < 0.5:
                    # Random next
                    is_random_next = True
                    # Note(mingdachen): in this case, we just swap tokens_a and tokens_b
                    tokens_a, tokens_b = tokens_b, tokens_a
                else:
                    # Actual next
                    is_random_next = False

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                instance = tokenizer(
                    " ".join(tokens_a),
                    " ".join(tokens_b),
                    truncation="longest_first",
                    max_length=max_seq_length,
                    # We use this option because DataCollatorForLanguageModeling
                    # is more efficient when it receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )
                assert len(instance["input_ids"]) <= max_seq_length
                instance["sentence_order_label"] = 1 if is_random_next else 0
                instances.append(instance)

            current_chunk = []
            current_length = 0

    return instances


def tokenize_function(tokenizer, examples):
    # Remove empty texts
    texts = (text for text in examples["text"] if len(text) > 0 and not text.isspace())

    new_examples = {col: [] for col in COLUMN_NAMES}

    for text in texts:
        instances = create_instances_from_document(tokenizer, text, max_seq_length=32768)
        #print(instances)
        for instance in instances:
            for key, value in instance.items():
                new_examples[key].append(value)

    return new_examples


if __name__ == "__main__":
    random.seed(0)
    nltk.download("punkt")

    # Old Albert Code
    # tokenizer = AlbertTokenizerFast.from_pretrained("albert-large-v2")
    # wikitext = load_dataset("wikitext", "wikitext-103-v1", cache_dir="./data/cache")
    # End Old Albert Code ##############################################################

    model_name = "HuggingFaceH4/zephyr-7b-beta"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=False)
    tokenizer.pad_token = tokenizer.eos_token  

    wikitext = load_dataset('json', data_files="/home/shaunst/repo/hivemind/examples/mistral/data/model_training/train_llama.json")


    tokenized_datasets = wikitext.map(
        partial(tokenize_function, tokenizer),
        batched=True,
        num_proc=8,
        remove_columns=["text"],
    )

    tokenized_datasets.save_to_disk("./data/albert_tokenized_wikitext")
    tokenizer.save_pretrained("./data/tokenizer")
