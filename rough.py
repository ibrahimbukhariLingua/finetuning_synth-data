from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from torch.nn.utils.rnn import pad_sequence
import datasets

from pathlib import Path
import random


def normalize_message(msg):
    return {
        "role": msg["role"],
        "content": msg["content"].strip(),
        "tools": msg.get("tools", []),
        "reasoning_content": msg.get("reasoning_content", ""),
    }


def normalize_cpt_item(item):
    return {"text": item["text"].strip()}


def normalize_sft_item(x):
    return {"messages": [normalize_message(msg) for msg in x["messages"]]}


def load_cpt_dataset(root_dir, num_proc=1):
    ds_max_sizes = {
        "agefi_articles": 35_000,
        "bankofengland": 6_000,
        "citeco": 500,
        "eirl": 111,
        "iastate": 1300,
        "investopedia": 50_000,
        "lafinancepourtous": 6_000,
        "multi-translations": 200_000,
        "sinvestir": 200,
        "touteleurope": 8_000,
        "wikipedia": 150_000,
        "worldbank/blogs": 30_000,
        "worldbank/documents": 50_000,
        # "worldbank/openknowledge": 3,
        "fineweb_edu_fin": 100_000,
        "fineweb2_fin/fr": 200_000,
        "fineweb2_fin/de": 200_000,
    }

    ds_list = []
    for name, maxlen in ds_max_sizes.items():
        path = Path(root_dir, name)
        ds = datasets.load_dataset("json", data_dir=str(path), split="train")

        if len(ds) > maxlen:
            selected_ids = random.sample(range(len(ds)), k=maxlen)
            ds = ds.select(selected_ids)

        ds = ds.map(
            normalize_cpt_item, num_proc=num_proc, remove_columns=ds.column_names
        )
        ds_list.append(ds)

    return datasets.concatenate_datasets(ds_list)


def load_parallel_dataset(root_dir, num_proc=1):
    ds_max_sizes = {
        "enar": 50_000,
        "ende": 50_000,
        "enes": 50_000,
        "enfr": 50_000,
        "enit": 50_000,
        "enja": 50_000,
        "ennl": 50_000,
        "enpt": 50_000,
        "frde": 50_000,
        "fres": 50_000,
        "frit": 50_000,
        "frnl": 50_000,
    }

    ds_list = []
    for name, maxlen in ds_max_sizes.items():
        path = Path(root_dir, name)
        ds = datasets.load_dataset("json", data_dir=str(path), split="train")

        if len(ds) > maxlen:
            selected_ids = random.sample(range(len(ds)), k=maxlen)
            ds = ds.select(selected_ids)

        ds = ds.map(
            normalize_sft_item, num_proc=num_proc, remove_columns=ds.column_names
        )
        ds_list.append(ds)

    return datasets.concatenate_datasets(ds_list)


def _load_sft_dataset_backup(path):
    from jsonl_datasets import JsonlDatasetReader

    reader = JsonlDatasetReader(path)

    def _normalize_msg(msg):
        return {
            "role": msg["role"],
            "content": msg["content"],
            "tool_calls": msg.get("tool_calls"),
        }

    def _normalize_sample(x):
        return {
            "messages": [_normalize_msg(msg) for msg in x["messages"]],
            "tools": x.get("tools", []),
        }

    tool_calls_features = {
        "type": datasets.Value("string"),
        "function": {
            "name": datasets.Value("string"),
            "arguments": datasets.Value("string"),
        },
    }

    messages_features = {
        "role": datasets.Value("string"),
        "content": datasets.Value("string"),
        "tool_calls": datasets.Sequence(tool_calls_features),
    }

    tools_features = {
        "name": datasets.Value("string"),
        "description": datasets.Value("string"),
        "parameters": {
            "type": datasets.Value("string"),
            "properties": datasets.Sequence(),
            "required": datasets.Sequence(datasets.Value("string")),
        },
    }

    features = datasets.Features(
        {
            "messages": datasets.Sequence(messages_features),
            "tools": datasets.Sequence(tools_features),
        }
    )
    ds = datasets.Dataset.from_generator(
        (_normalize_sample(x) for x in reader), features=features
    )


def load_sft_dataset(root_dir, num_proc=1):
    ds_max_sizes = {
        # Worse datasets
        "twitter-financial-news-sentiment": 10_000,
        "twitter-financial-news-topic": 10_000,
        "FiQA_Financial_Phrasebank_Combined": 10_000,
        "FinRiskAnalysis": 1_000,
        # Slightly better datasets
        "synthetic_pii_finance_multilingual": 20_000,
        "agefi_title_generation": 20_000,
        "agefi_tags": 20_000,
        "MetaMathQA": 20_000,
        "agefi_qa_rag": 20_000,
        "balance_sheet": 20_000,
        "Expert_comptable": 20_000,
        "finance": 20_000,
        "agefi_subheadline_generation": 20_000,
        # Better datasets
        "OpenCodeReasoning": 10_000,
        "synthetic-multi-pii-ner-v1": 5_000,
        "glaive-rag-v1": 50_000,
        "aya_dataset": 50_000,
        "glaive-code-assistant-v3": 50_000,
        # "glaive-function-calling-v2": 50_000,
        "financial-instruction-aq22": 50_000,
        "Synthia-v1.5-I": 50_000,
        "Synthia-Coder-v1.5-I": 50_000,
        "Synthia-v1.5-II": 50_000,
        "Llama-Nemotron-Post-Training-Dataset/chat": 30_000,
        "Llama-Nemotron-Post-Training-Dataset/code": 30_000,
        "Llama-Nemotron-Post-Training-Dataset/math": 30_000,
        "Llama-Nemotron-Post-Training-Dataset/science": 30_000,
        "Llama-Nemotron-Post-Training-Dataset/safety": 10_000,
        # Hard coded datasets
        "llm-ff-hard-coded": 1000,
    }

    ds_list = []
    for name, maxlen in ds_max_sizes.items():
        path = Path(root_dir, name)
        try:
            ds = datasets.load_dataset("json", data_dir=str(path), split="train")
        except Exception as e:
            raise e
            # TODO: take into account tool calling dataset. It needs to reformat the non-tool dataset as a tool calling one (but with empty tool related properties)
            ds = _load_sft_dataset_backup(path)

        if len(ds) > maxlen:
            selected_ids = random.sample(range(len(ds)), k=maxlen)
            ds = ds.select(selected_ids)

        ds = ds.map(
            normalize_sft_item, remove_columns=ds.column_names, num_proc=num_proc
        )
        ds_list.append(ds)

    return datasets.concatenate_datasets(ds_list)


def unify_cpt_and_sft_datasets(
    cpt_datasets,
    sft_datasets,
    cpt_text_key="text",
    sft_messages_key="messages",
    num_proc=1,
):
    """
    Combines CPT and SFT datasets into a unified format with columns:
        - 'text': for CPT samples
        - 'messages': for SFT samples
    Assumes each input dataset is a Hugging Face Dataset object.

    Args:
        cpt_datasets: list of CPT datasets
        sft_datasets: list of SFT datasets
        cpt_text_key: key in CPT dataset containing plain text
        sft_messages_key: key in SFT dataset containing OpenAI-style messages

    Returns:
        A single unified Dataset.
    """

    unified_cpt = []
    for ds in cpt_datasets:
        unified = ds.map(
            lambda x: {"text": x[cpt_text_key], "messages": None},
            remove_columns=ds.column_names,
            num_proc=num_proc,
        )
        unified_cpt.append(unified)

    unified_sft = []
    for ds in sft_datasets:
        unified = ds.map(
            lambda x: {"text": None, "messages": x[sft_messages_key]},
            remove_columns=ds.column_names,
            num_proc=num_proc,
        )
        unified_sft.append(unified)

    all_datasets = unified_cpt + unified_sft
    return datasets.concatenate_datasets(all_datasets)


def load_dataset(
    sft_root_dir,
    cpt_root_dir,
    parallel_root_dir,
    num_proc=1,
    sft_len_mult=1.0,
    cpt_len_mult=1.0,
    parallel_len_mult=1.0,
):
    print("Loading SFT datasets")
    sft_ds = load_sft_dataset(sft_root_dir, num_proc=num_proc)
    print("Loading CPT datasets")
    cpt_ds = load_cpt_dataset(cpt_root_dir, num_proc=num_proc)
    print("Loading parallel datasets")
    parallel_ds = load_parallel_dataset(parallel_root_dir, num_proc=num_proc)

    def _apply_len_mult(ds, mult):
        k = int(len(ds) * mult)
        available_ids = list(range(len(ds)))
        if k < len(ds):
            ids = random.sample(available_ids, k=k)
            ds = ds.select(ids)
        elif k > len(ds):
            ids = random.choices(available_ids, k=k)
            ds = ds.select(ids)
        return ds

    sft_ds = _apply_len_mult(sft_ds, sft_len_mult)
    cpt_ds = _apply_len_mult(cpt_ds, cpt_len_mult)
    parallel_ds = _apply_len_mult(parallel_ds, parallel_len_mult)

    print("Merging datasets")
    unified_dataset = unify_cpt_and_sft_datasets(
        [cpt_ds], [sft_ds, parallel_ds], num_proc=num_proc
    )
    print("Datasets fully loaded")
    return unified_dataset


### Collator


class DataCollatorForCPTAndSFTUnified:
    def __init__(
        self,
        tokenizer,
        instruction_prefix="<|im_start|>assistant\n",
        instruction_suffix="<|im_end|>",
        end_of_text_token="<|endoftext|>",
        max_length=4096,
    ):
        self.tokenizer = tokenizer
        self.instruction_prefix = instruction_prefix
        self.instruction_suffix = instruction_suffix
        self.end_of_text_token = end_of_text_token
        self.max_length = max_length

        self._prefix_ids = self.tokenizer.encode(
            self.instruction_prefix, add_special_tokens=False
        )
        self._suffix_ids = self.tokenizer.encode(
            self.instruction_suffix, add_special_tokens=False
        )

    def __call__(self, features):
        # final output variables
        input_ids = []
        labels = []
        attention_mask = []

        # Here we separate cpt and sft samples
        cpt_samples = []
        sft_samples = []
        for x in features:
            if x["text"] is not None:
                cpt_samples.append(x["text"] + "<|endoftext|>")
            if x["messages"] is not None:
                sft_samples.append(x["messages"])

        # Here we handle the simple text input
        if cpt_samples:
            encoded_cpt = self.tokenizer(
                cpt_samples,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )

            for ids in encoded_cpt["input_ids"]:
                input_ids.append(torch.tensor(ids))
                labels.append(torch.tensor(ids))
                attention_mask.append(torch.tensor([1] * len(ids)))

        # Here we handle SFT messages
        if sft_samples:
            for msg in sft_samples:
                formatted = self.tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=False
                )

                enc = self.tokenizer(
                    formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    return_offsets_mapping=False,
                    add_special_tokens=False,
                )

                input_id = enc.input_ids[0]
                label = torch.full_like(input_id, fill_value=-100)

                # Locate assistant spans
                idx = 0
                while idx < len(input_id) - len(self._prefix_ids):
                    # Look for instruction prefix
                    if (
                        list(input_id[idx : idx + len(self._prefix_ids)].tolist())
                        == self._prefix_ids
                    ):
                        start = idx + len(self._prefix_ids)
                        # Search for suffix
                        end = start
                        while end < len(input_id):
                            if (
                                list(
                                    input_id[end : end + len(self._suffix_ids)].tolist()
                                )
                                == self._suffix_ids
                            ):
                                end += len(self._suffix_ids)
                                break
                            else:
                                end += 1

                        label[start:end] = input_id[start:end]
                        idx = end
                    else:
                        idx += 1

                input_ids.append(input_id)
                labels.append(label)
                attention_mask.append(torch.full_like(input_id, fill_value=1))

        # Here we pad the merged inputs
        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            padding_side="left",
        )
        labels = pad_sequence(
            labels, batch_first=True, padding_value=-100, padding_side="left"
        )
        attention_mask = pad_sequence(
            attention_mask, batch_first=True, padding_value=0, padding_side="left"
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


###


def main(args):
    from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
    from accelerate import PartialState

    base_model = args.base_model
    # peft_config = LoraConfig(
    #     r=4,  # Rank dimension - typically between 4-32
    #     lora_alpha=8,  # LoRA scaling factor - typically 2x rank
    #     bias="none",  # Bias type for LoRA. the corresponding biases will be updated during training.
    #     target_modules="all-linear",  # Which modules to apply LoRA to
    #     task_type="CAUSAL_LM",  # Task type for model architecture
    # )

    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map={"": device_string},
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")

    train_dataset = load_dataset(
        args.sft_root, args.cpt_root, args.parallel_root, num_proc=args.num_proc
    )
    train_dataset = train_dataset.shuffle().select(
        range(min(len(train_dataset), 500_000))
    )

    instruction_prefix = "<|im_start|>assistant\n"
    instruction_suffix = "<|im_end|>"
    max_length = 4096

    collator = DataCollatorForCPTAndSFTUnified(
        tokenizer,
        instruction_prefix=instruction_prefix,
        instruction_suffix=instruction_suffix,
        end_of_text_token="<|endoftext|>",
        max_length=max_length,
    )

    # batch_size = 8
    # save_steps = 3000 // batch_size

    num_train_epochs = 1
    save_steps = 10_000

    training_args = SFTConfig(
        max_seq_length=max_length,
        output_dir=args.output_dir,
        bf16=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=num_train_epochs,
        learning_rate=1e-6,
        warmup_ratio=0.02,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        save_strategy="steps",
        save_steps=save_steps,
        report_to="tensorboard",
        logging_strategy="steps",
        logging_steps=10,
        activation_offloading=False,
        use_liger_kernel=True,
        dataloader_num_workers=args.num_proc,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--sft-root", required=True, type=str)
    parser.add_argument("--parallel-root", required=True, type=str)
    parser.add_argument("--cpt-root", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--num-proc", default=1, type=int)

    args = parser.pa