import transformers
from dataclasses import dataclass
from datasets import load_dataset
import warnings

def preprocess_function(examples, tokenizer, seq_len):
    model_inputs = {"input_ids": [[]]}
    acc_len = 0
    for message in examples["text"]:
        message_ids = tokenizer.encode(message, add_special_tokens=False)
        input_ids_list = []
        for i in range(0, len(message_ids), seq_len - 1):
            input_ids_list.append(
                message_ids[i : i + seq_len - 1] + [tokenizer.eos_token_id]
            )
        for input_ids in input_ids_list:
            if acc_len + len(input_ids) > seq_len:
                model_inputs["input_ids"].append([input_ids])
                acc_len = len(input_ids)
            else:
                model_inputs["input_ids"][-1].append(input_ids)
                acc_len += len(input_ids)
    return model_inputs

@dataclass
class DataCollatorWithFlattening(transformers.DefaultDataCollator):
    """
    Data collator used for padding free approach. Does the following:

    - concatate the entire mini batch into single long sequence [1, total_tokens]
    - uses `separator_id` to separate sequences within the concatenated `labels`, default value is -100
    - no padding will be added, returns `input_ids`, `labels` and `position_ids`
    """

    def __init__(
        self,
        *args,
        return_position_ids=True,
        separator_id=-100,
        max_len=8192,
        pad_token_id=128001,
        label_ignore_id=-100,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.return_position_ids = return_position_ids
        self.separator_id = separator_id
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.label_ignore_id = label_ignore_id
        warnings.warn(
            "Using `DataCollatorWithFlattening` will flatten the entire mini batch into single long sequence."
            "Make sure your attention computation is able to handle it!"
        )

    def __call__(self, features, return_tensors=None, separator_id=None):
        def padding_ret(ret):
            padding_len = self.max_len - len(ret["input_ids"])
            if self.return_position_ids:
                padded_position_ids = list(range(padding_len))
                ret["position_ids"] += padded_position_ids
            ret["input_ids"] += [self.pad_token_id] * padding_len
            ret["labels"] += [self.label_ignore_id] * padding_len
            ret["input_ids"] = ret["input_ids"][: self.max_len]
            ret["labels"] = ret["labels"][: self.max_len]
            return ret

        if return_tensors is None:
            return_tensors = self.return_tensors
        if separator_id is None:
            separator_id = self.separator_id

        rets = []
        for idx in range(0, len(features)):
            ret = {"input_ids": [], "labels": []}
            if self.return_position_ids:
                ret.update({"position_ids": []})
            for f_input_ids in features[idx]["input_ids"]:
                ret["input_ids"] += f_input_ids
                ret["labels"] += [separator_id] + f_input_ids[1:]
                if self.return_position_ids:
                    ret["position_ids"] += list(range(len(f_input_ids)))
            rets.append(padding_ret(ret))

        return transformers.default_data_collator(rets, return_tensors)


def prapare_dataset(tokenizer, data_args, split="train"):
    dataset = load_dataset(data_args.dataset_name, split=split)
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=1024,
        remove_columns=dataset.column_names,
        num_proc=128,
        fn_kwargs={"tokenizer": tokenizer, "seq_len": data_args.seq_len},
    )
    return processed_dataset
