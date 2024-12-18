# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from .processor_utils import get_paligemma_token_type_ids, get_pixel_values, infer_seqlen


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template import Template


logger = get_logger(__name__)


def _encode_pairwise_example_obj(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
    object: Optional[str],
) -> Tuple[List[int], List[int], List[int], List[int]]:
    instruction_prompt = ""
    if object == "no_object":
        instruction_prompt = """You are an expert in question answering, and you need to respond to me by integrating the dimensions of helpfulness, correctness, instruction following"""
    else:
        instruction_prompt="""You are an expert in question answering, and you need to focus on the {object} dimension to respond to me. The evaluation values for this dimension range from 1 to 5, representing 
        1. **Irrelevant**: No alignment.
        2. **Partial Focus**: Addresses one aspect poorly.
        3. **Partial Compliance**:
        - (1) Meets goals or restrictions, neglecting others.
        - (2) Acknowledges both but slight deviations.
        4. **Almost There**: Near alignment, minor deviations.
        5. **Comprehensive Compliance**: Fully aligns, meets all requirements. 
        Please provide me with a response based on the evaluation value of {d}."""
    
 
    # if processor is not None and not hasattr(processor, "image_seq_length"):  # llava-like models
    #     prompt[0]["content"] = template.image_token + prompt[0]["content"]

    if object == "no_object":
        prompt[0]['content'] = instruction_prompt + prompt[0]['content']
        chosen_messages = prompt + [response[0]]
        rejected_messages = prompt + [response[1]]
    else:
        prompt[0]['content'] = instruction_prompt.format(object=object,d=response[0][object]) + prompt[0]['content']
        chosen_messages = prompt + [response[0]]
        rejected_messages = prompt + [response[1]]
    prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, chosen_messages, system, tools)
    _, rejected_ids = template.encode_oneturn(tokenizer, rejected_messages, system, tools)

    if template.efficient_eos:
        chosen_ids += [tokenizer.eos_token_id]
        rejected_ids += [tokenizer.eos_token_id]

    if processor is not None and hasattr(processor, "image_seq_length"):  # paligemma models
        image_token_id = tokenizer.convert_tokens_to_ids(template.image_token)
        prompt_ids = [image_token_id] * getattr(processor, "image_seq_length") + prompt_ids

    # consider the response is more important
    source_len, target_len = infer_seqlen(len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), cutoff_len)
    prompt_ids = prompt_ids[:source_len]
    chosen_ids = chosen_ids[:target_len]
    rejected_ids = rejected_ids[:target_len]

    chosen_input_ids = prompt_ids + chosen_ids
    chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
    rejected_input_ids = prompt_ids + rejected_ids
    rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids

    return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels


def preprocess_pairwise_dataset_obj(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    model_inputs = {
        "chosen_input_ids_no_object": [],
        "chosen_attention_mask_no_object": [],
        "chosen_labels_no_object": [],
        "rejected_input_ids_no_object": [],
        "rejected_attention_mask_no_object": [],
        "rejected_labels_no_object": [],
        # helpfulness
        "chosen_input_ids_helpfulness": [],
        "chosen_attention_mask_helpfulness": [],
        "chosen_labels_helpfulness": [],
        "rejected_input_ids_helpfulness": [],
        "rejected_attention_mask_helpfulness": [],
        "rejected_labels_helpfulness": [],
        # correctness
        "chosen_input_ids_correctness": [],
        "chosen_attention_mask_correctness": [],
        "chosen_labels_correctness": [],
        "rejected_input_ids_correctness": [],
        "rejected_attention_mask_correctness": [],
        "rejected_labels_correctness": [],
        # instruction_following
        "chosen_input_ids_instruction_following": [],
        "chosen_attention_mask_instruction_following": [],
        "chosen_labels_instruction_following": [],
        "rejected_input_ids_instruction_following": [],
        "rejected_attention_mask_instruction_following": [],
        "rejected_labels_instruction_following": [],

    }
    if processor is not None:
        model_inputs["pixel_values"] = []
        if hasattr(processor, "image_seq_length"):  # paligemma models
            model_inputs["chosen_token_type_ids"] = []
            model_inputs["rejected_token_type_ids"] = []

    for i in range(len(examples["prompt"])):
        if len([examples["prompt"][i]]) % 2 != 1 or len(examples["response"][i]) < 2:
            logger.warning("Dropped invalid example: {}".format([examples["prompt"][i]] + examples["response"][i]))
            continue
        object_list = ['no_object','helpfulness','correctness','instruction_following']
        for object in object_list:
            
            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = _encode_pairwise_example_obj(
                prompt=[examples["prompt"][i]],
                response=examples["response"][i],
                system=examples["system"][i],
                tools=examples["tools"][i],
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                cutoff_len=data_args.cutoff_len,
                object=object
            )
            model_inputs["chosen_input_ids_"+object].append(chosen_input_ids)
            model_inputs["chosen_attention_mask_"+object].append([1] * len(chosen_input_ids))
            model_inputs["chosen_labels_"+object].append(chosen_labels)
            model_inputs["rejected_input_ids_"+object].append(rejected_input_ids)
            model_inputs["rejected_attention_mask_"+object].append([1] * len(rejected_input_ids))
            model_inputs["rejected_labels_"+object].append(rejected_labels)
            # if processor is not None:
            #     model_inputs["pixel_values"].append(get_pixel_values(examples["images"][i], processor))
            #     if hasattr(processor, "image_seq_length"):  # paligemma models
            #         model_inputs["chosen_token_type_ids"].append(
            #             get_paligemma_token_type_ids(len(chosen_input_ids), processor)
            #         )
            #         model_inputs["rejected_token_type_ids"].append(
            #             get_paligemma_token_type_ids(len(rejected_input_ids), processor)
            #         )

    return model_inputs


def print_pairwise_dataset_example_obj(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    
    object_list = ['no_object','helpfulness','correctness','instruction_following']
    for object in object_list:
        valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels_"+object]))
        valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels_"+object]))
        print("chosen_input_ids_"+object+":\n{}".format(example["chosen_input_ids_"+object]))
        print("chosen_inputs_"+object+":\n{}".format(tokenizer.decode(example["chosen_input_ids_"+object], skip_special_tokens=False)))
        print("chosen_label_ids_"+object+":\n{}".format(example["chosen_labels_"+object]))
        print("chosen_labels_"+object+":\n{}".format(tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)))
        print("rejected_input_ids_"+object+":\n{}".format(example["rejected_input_ids_"+object]))
        print("rejected_inputs_"+object+":\n{}".format(tokenizer.decode(example["rejected_input_ids_"+object], skip_special_tokens=False)))
        print("rejected_label_ids_"+object+":\n{}".format(example["rejected_labels_"+object]))
        print("rejected_labels_"+object+":\n{}".format(tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)))
