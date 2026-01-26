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
            
            if self.tokenizer.bos_token_id is not None:
                input_ids.append(self.tokenizer.bos_token_id)
                labels.append(-100)
            
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "observation":
                    role_str = "user"
                    if not content.startswith("Retrieved Context:"):
                        content = f"Retrieved Context:\n{content}"
                else:
                    role_str = role
                
                # 1. Header 拼接: <|im_start|>role\n
                role_ids = self.tokenizer.encode(role_str, add_special_tokens=False)
                header_ids = [im_start_id] + role_ids + [newline_id]
                
                # 2. Content 拼接
                content_ids = self.tokenizer.encode(content, add_special_tokens=False)
                
                # 3. Footer 拼接: <|im_end|>\n
                footer_ids = [im_end_id] + [newline_id]
                
                # 完整片段 ID 序列
                part_ids = header_ids + content_ids + footer_ids
                input_ids.extend(part_ids)
                
                # 训练逻辑修复：移除有害的前缀 Mask，确保模型学习如何“开口”输出 SubQuery 等
                if role == "assistant":
                    is_sub_query = content.startswith("SubQuery:")
                    if self.sub_query_only and not is_sub_query:
                        # 如果开启了只练子查询模式，则屏蔽非子查询的 Assistant 消息
                        part_labels = [-100] * len(part_ids)
                    else:
                        # 只 Mask 掉 Header 部分 (<|im_start|>assistant\n)
                        # 保留 Content (含 SubQuery:/Final Answer: 前缀) 和 Footer 的 Loss
                        part_labels = [-100] * len(header_ids) + content_ids + footer_ids
                else:
                    # 对于 User/System/Observation 片段，Loss 全部屏蔽 (-100)
                    part_labels = [-100] * len(part_ids)
                
                labels.extend(part_labels)
            
            # 截断处理
            if len(input_ids) > self.max_len:
                input_ids = input_ids[:self.max_len]
                labels = labels[:self.max_len]
            
            attention_mask = [1] * len(input_ids)
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)
            
        # 批量 Padding
        padded = self.tokenizer.pad(
            {"input_ids": input_ids_list, "attention_mask": attention_mask_list},
            padding=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        # 对 Labels 进行手动 Padding (-100)
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
