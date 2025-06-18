import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import numpy as np
from tqdm import tqdm
import torch
from tabulate import tabulate
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from huggingface_hub.hf_api import HfFolder
HfFolder.save_token("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")

from util import Data_to_hf



# ================================ FINETUNING CLASS ================================ # 

class Finetune_w_checkpoint():
    
    def __init__(self, **kwargs):
        
        # Extract parameters from kwargs
        ft_data_dir = kwargs.get("ft_data_dir")
        model_name = kwargs.get("model_name")
        num_of_samples = kwargs.get("num_of_samples", 1000)
        self.ft_model_name = kwargs.get("ft_model_name")

        # Parameter Check
        if not all([ft_data_dir, model_name, self.ft_model_name]):
            raise ValueError("Missing required parameters: 'ft_data_dir', 'model_name', or 'ft_model_name'.")

        # Initialize Model and Tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
        self.model.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

        # Initialize Object of Data_to_hf class
        # Run the object to get hf_dataset for finetuning
        data_to_hf = Data_to_hf(directory_path=ft_data_dir, num_of_samples=num_of_samples)
        self.dataset = data_to_hf.run()
        
        # Get the Maximum Sequence Length from dataset
        self.max_seq_length = self.get_max_seq_length_from_chat_dataset(self.dataset, self.tokenizer)

        # Compute the interval of steps to be checkpointed
        total_steps = self.compute_total_steps(len(self.dataset))
        save_steps = (total_steps // 4)+1

        # Initialize LoRA and Tuning configuration
        self.peft_config, self.args = self.lora_config_and_args(
            ft_model_name=self.ft_model_name,
            max_seq_length=self.max_seq_length,
            save_steps=save_steps
        )

    # -------------------- HELPER FUNCTIONS -------------------- #
    
    def compute_total_steps(self, num_samples, batch_size=1, gradient_accumulation_steps=8, epochs=1):
        steps_per_epoch = (num_samples // batch_size) // gradient_accumulation_steps
        return steps_per_epoch * epochs

    def printTrainer(self, trainer, tokenizer):
        train_dataloader = trainer.get_train_dataloader()
        for batch_data in train_dataloader:
            input_ids = batch_data['input_ids'][0]
            attention_mask = batch_data['attention_mask'][0]
            label_ids = batch_data['labels'][0]

            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            decoded = tokenizer.decode(input_ids, skip_special_tokens=False)

            print("Tokens:")
            for i, token in enumerate(tokens):
                attn = attention_mask[i].item()
                label_id = label_ids[i].item()
                label_token = tokenizer.convert_ids_to_tokens([label_id])[0] if label_id != -100 else 'IGN'
                print(f"{i:2d}: {token:12s} | Label_id: {label_id} | Attention: {attn} | Label: {label_token}")

            print("\nDecoded sentence:")
            print(decoded)
            break

    def get_max_seq_length_from_chat_dataset(self, dataset, tokenizer, instruction_template="<|im_start|>user\n", response_template="<|im_start|>assistant\n", percentile=97, verbose=True):    
        input_lengths = []
        for example in tqdm(dataset, desc="Tokenizing chat examples"):
            messages = example["messages"]
            full_text = ""
            for message in messages:
                role = message["role"]
                content = message["content"]
                if role == "system":
                    full_text += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    full_text += f"{instruction_template}{content}<|im_end|>\n"
                elif role == "assistant":
                    full_text += f"{response_template}{content}<|im_end|>\n"
                else:
                    raise ValueError(f"Unsupported role: {role}")

            tokens = tokenizer(full_text, add_special_tokens=False, truncation=False)["input_ids"]
            input_lengths.append(len(tokens))

        if verbose:
            stats = [
                ["Min", np.min(input_lengths)],
                ["Max", np.max(input_lengths)],
                ["Mean", f"{np.mean(input_lengths):.2f}"],
                ["Median", np.median(input_lengths)],
                [f"{percentile}th Percentile", np.percentile(input_lengths, percentile)]
            ]
            print("\n📊 Chat Token Stats:\n")
            print(tabulate(stats, headers=["Metric", "Value"], tablefmt="github"))

        return int(np.percentile(input_lengths, percentile))

    # -------------------- CONFIG FUNCTION -------------------- #

    def lora_config_and_args(self, ft_model_name, max_seq_length, rank_dimension=4, lora_alpha=8, lora_dropout=0.05, save_steps=100):
        peft_config = LoraConfig(
            r=rank_dimension,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

        args = SFTConfig(
            max_seq_length=max_seq_length,
            output_dir=f"Models/{ft_model_name}",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            learning_rate=2e-4,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            logging_steps=10,
            save_strategy="steps",              # Save based on step count
            save_steps=save_steps,              # Save at halfway point
            bf16=True
        )

        return peft_config, args

    # -------------------- MAIN FUNCTION -------------------- #

    def run(self, ft_w_lora:bool):
        collator = DataCollatorForCompletionOnlyLM(
            instruction_template="<|im_start|>user\n",
            response_template="<|im_start|>assistant\n",
            tokenizer=self.tokenizer,
            mlm=False
        )

        if ft_w_lora == True:
            trainer = SFTTrainer(
                model=self.model,
                args=self.args,
                train_dataset=self.dataset,
                peft_config=self.peft_config,
                processing_class=self.tokenizer,
                data_collator=collator,
            )
        else:
            trainer = SFTTrainer(
                model=self.model,
                args=self.args,
                train_dataset=self.dataset,
                processing_class=self.tokenizer,
                data_collator=collator,
            )

        self.printTrainer(trainer, self.tokenizer)
        
        trainer.train()

        # trainer.push_to_hub(f"ibrahimbukhariLingua/{self.ft_model_name}")

        # return f"ibrahimbukhariLingua/{self.ft_model_name}"




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model with checkpoint support.")

    parser.add_argument('--ft_data_dir', type=str, required=True, help='Path to the fine-tuning data directory.')
    parser.add_argument('--model_name', type=str, required=True, help='Base model name to fine-tune.')
    parser.add_argument('--ft_model_name', type=str, required=True, help='Name to assign to the fine-tuned model.')
    parser.add_argument('--num_of_samples', type=int, required=True, help='Number of samples to use for fine-tuning.')
    parser.add_argument('--ft_w_lora', action='store_true', help='Whether to fine-tune with LoRA.')

    args = parser.parse_args()

    kwargs = {
        'ft_data_dir': args.ft_data_dir,
        'model_name': args.model_name,
        'ft_model_name': args.ft_model_name,
        'num_of_samples': args.num_of_samples,
    }

    ft_trainer = Finetune_w_checkpoint(**kwargs)
    ft_trainer.run(ft_w_lora=args.ft_w_lora)
    
    
    """
    
    python3 your_script.py \
    --ft_data_dir data/training/synth_wiki_finance_v2.2 \
    --model_name qwen/qwen2.5-0.5B-instruct \
    --ft_model_name testing_new_ft_code \
    --num_of_samples 1000 \
    --ft_w_lora

    """

