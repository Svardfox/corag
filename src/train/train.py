import os
import sys
import json
import logging
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datasets import load_dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.config import Arguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChainOfRagCollator:
    tokenizer: transformers.PreTrainedTokenizer
    max_len: int = 2048
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        newline_id = self.tokenizer.encode("\n", add_special_tokens=False)[-1]
            
        if im_start_id is None or im_start_id == self.tokenizer.unk_token_id:
            im_start_id = 151644
            im_end_id = 151645

        for feature in features:
            messages = feature["messages"]
            
            input_ids = []
            labels = []
            
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                # Check for observation mapping
                if role == "observation":
                    # Map observation to system or specific observation role if model supports
                    # We treat it as non-trainable context
                    role_str = "system" # or "observation"
                else:
                    role_str = role
                
                # 1. Header 拼接: <|im_start|>role\n
                role_ids = self.tokenizer.encode(role_str, add_special_tokens=False)
                header_ids = [im_start_id] + role_ids + [newline_id]
                
                # 2. Content 拼接
                content_ids = self.tokenizer.encode(content, add_special_tokens=False)
                
                # 3. Footer 拼接: <|im_end|>\n
                footer_ids = [im_end_id] + [newline_id]
                
                # Full specific part
                part_ids = header_ids + content_ids + footer_ids
                input_ids.extend(part_ids)
                
                # Label Masking
                if role == "assistant":
                    # Mask the header, train on content + footer (EOS)
                    part_labels = [-100] * len(header_ids) + content_ids + footer_ids
                else:
                    # Mask User / System / Observation
                    part_labels = [-100] * len(part_ids)
                
                labels.extend(part_labels)
            
            # Truncate / Pad
            if len(input_ids) > self.max_len:
                # Truncate to max_len
                input_ids = input_ids[:self.max_len]
                labels = labels[:self.max_len]
            
            attention_mask = [1] * len(input_ids)
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)
            
        # Padding
        padded = self.tokenizer.pad(
            {"input_ids": input_ids_list, "attention_mask": attention_mask_list},
            padding=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        # Pad labels manually with -100
        max_batch_len = padded["input_ids"].shape[1]
        padded_labels = []
        for l in labels_list:
            padded_l = l + [-100] * (max_batch_len - len(l))
            padded_labels.append(padded_l)
            
        padded["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        
        return padded

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    train_file: Optional[str] = field(
        default="data/train_with_graph_retrieval.jsonl",
        metadata={"help": "The input training data file (a jsonl file)."}
    )

def train():
    parser = transformers.HfArgumentParser((ModelArguments, Arguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    
    # Load Model
    print(f"Loading model: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        pad_token='<|endoftext|>'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Data
    data_files = {}
    if model_args.train_file:
        data_files["train"] = model_args.train_file
        
    dataset = load_dataset("json", data_files=data_files)
    
    # Collator
    collator = ChainOfRagCollator(tokenizer=tokenizer, max_len=training_args.max_len)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=collator,
    )
    
    if training_args.do_train:
        trainer.train()
        trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
