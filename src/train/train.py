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
    sub_query_only: bool = False
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        
        for feature in features:
            messages = feature["messages"]
            
            input_ids = []
            labels = []
            
            # Manual ChatML formatting
            # TODO: Use the model's official chat template in production
            # Format: <|im_start|>role\ncontent<|im_end|>\n
            
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
                
                # Header
                header = f"<|im_start|>{role_str}\n"
                header_ids = self.tokenizer.encode(header, add_special_tokens=False)
                
                # Content
                content_ids = self.tokenizer.encode(content, add_special_tokens=False)
                
                # Footer
                footer = "<|im_end|>\n"
                footer_ids = self.tokenizer.encode(footer, add_special_tokens=False)
                
                # Full specific part
                part_ids = header_ids + content_ids + footer_ids
                input_ids.extend(part_ids)
                
                # Label Masking
                if role == "assistant":
                    # Determine if this message should be trained at all
                    is_sub_query = content.startswith("SubQuery:")
                    if self.sub_query_only and not is_sub_query:
                        # Mask everything if in sub_query_only mode and this isn't a sub-query
                        part_labels = [-100] * len(part_ids)
                    else:
                        # Standard training or sub_query_only mode with a sub-query
                        # Mask the header
                        part_labels = [-100] * len(header_ids)
                        
                        # Shield loss for prefixes: "SubQuery:", "SubAnswer:", "Final Answer:"
                        prefix = None
                        for p in ["SubQuery: ", "SubAnswer: ", "Final Answer: "]:
                            if content.startswith(p):
                                prefix = p
                                break
                        
                        if prefix:
                            # Encode prefix to get its token length
                            prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
                            # Mask prefix and keep the rest of content + footer
                            part_labels += [-100] * len(prefix_ids)
                            part_labels += content_ids[len(prefix_ids):] + footer_ids
                        else:
                            # No prefix found, train on full content
                            part_labels += content_ids + footer_ids
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
    sub_query_only: bool = field(
        default=False,
        metadata={"help": "If true, only train on sub-queries and ignore other assistant messages."}
    )

def train():
    parser = transformers.HfArgumentParser((ModelArguments, Arguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    
    # Load Model
    print(f"Loading model: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
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
    collator = ChainOfRagCollator(
        tokenizer=tokenizer, 
        max_len=training_args.max_len,
        sub_query_only=model_args.sub_query_only
    )
    
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
