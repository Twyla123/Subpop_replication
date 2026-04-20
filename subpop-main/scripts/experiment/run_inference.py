import argparse
import ast
import pathlib
import warnings
from multiprocessing import Pool
from typing import List, Tuple, Optional, Dict
from functools import partial

import pandas as pd
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from subpop.utils.survey_utils import ordinal_emd, list_normalize

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
warnings.filterwarnings("ignore")


def prompt_chat_formatter(prompt: str) -> List[Dict[str, str]]:
    """
    Reformat the QA steering prompt into the chat conversation.
    QA prompt is in Question:-Ansewer: format. Convert to user-assistant.
    """
    prompt_split = prompt.split("Answer:")[:-1]
    prompt_split = [x.strip() for x in prompt_split]
    messages = []
    messages.append( # subpopulation steering question
        {"role": "user", "content": prompt_split[0].strip()}
    )
    messages.append( # subpopulation steering answer (ex. A. Very liberal)
        {"role": "assistant", "content": prompt_split[1].split("\n")[0].strip()}
    )
    messages.append( # survey question
        {
            "role": "user",
            "content": prompt_split[1].replace(messages[1]["content"], "").strip(),
        }
    )
    return messages


def get_llm_engine(args) -> Tuple:
    """
    Load the LLM engine on a local machine and define sampling parameters.
    """
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=1.0,
        logprobs=20,
    )
    llm = LLM(
        model=args.base_model_name_or_path,
        tensor_parallel_size=args.tp_size,
        max_logprobs=args.max_logprobs,
        enable_prefix_caching=args.enable_prefix_caching,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        enable_lora=True,
    )
    return sampling_params, llm


def inference_offline(args, data_list_test, sampling_params, llm, lora_idx):
    """
    Offline batched inference for input_prompts in the data_list_test.
    """
    tokenizer = llm.get_tokenizer()
    # prepare the alphabet (A, B, ...) tokens for logprob extraction
    alphabet_coded: List[Tuple[int, int]] = [
        tuple([
            tokenizer.encode(
                "Answer:" + chr(ord("A") + idx), add_special_tokens=False
            )[-1],
            tokenizer.encode(
                "Answer: " + chr(ord("A") + idx), add_special_tokens=False
            )[-1],
        ])
        for idx in range(26)
    ]

    # prepare the input prompt and run inference
    prompts = [data[1] for data in data_list_test]
    if args.is_chat:
        prompts = [prompt_chat_formatter(prompt) for prompt in prompts]
    if args.lora_path is not None:
        print(f"--> inference_offline: LoRA name = {args.lora_name[lora_idx]}")
        print(f"--> inference_offline: LoRA path = {args.lora_path[lora_idx]}")
        lora_request = LoRARequest(
            args.lora_name[lora_idx], lora_idx + 1,
            lora_path=args.lora_path[lora_idx],
        )
        if args.is_chat:
            outputs = llm.chat(prompts, sampling_params, lora_request=lora_request)
        else:
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    else:
        if args.is_chat:
            outputs = llm.chat(prompts, sampling_params)
        else:
            outputs = llm.generate(prompts, sampling_params)

    # extract logprobs and calculate the probability per option
    results = []
    for idx, output in enumerate(outputs):
        logprobs = output.outputs[0].logprobs[0]
        len_options = len(ast.literal_eval(data_list_test[idx][2]))
        prob_per_option = []
        for opt_idx in range(len_options):
            logprob_1 = logprobs.get(alphabet_coded[opt_idx][0], None)
            logprob_2 = logprobs.get(alphabet_coded[opt_idx][1], None)
            prob_1 = np.exp(logprob_1.logprob) if logprob_1 is not None else 0
            prob_2 = np.exp(logprob_2.logprob) if logprob_2 is not None else 0
            prob_per_option.append(prob_1 + prob_2)
        results.append(
            (
                idx,
                sum(prob_per_option),
                np.array(prob_per_option) / sum(prob_per_option),
            )
        )
    return results, (
        args.lora_name[lora_idx]
        if args.lora_name is not None
        else args.base_model_name_or_path
    )


def run_survey(args, sampling_params, llm, lora_idx) -> None:
    """
    Run inference for each input file and LoRA module.
    """

    # load the file.
    # llm_dist is output distribution from model
    # llm_prob_sum is the sum of probabilities assigned to option ('A', 'B', ...)
    # emd is the WD between precalculated human and model's distribution.
    test_df = pd.read_csv(args.input_paths[lora_idx])
    test_df["llm_dist"] = None
    test_df["llm_prob_sum"] = None
    test_df["emd"] = None
    data_list_test = [
        (idx, row["input_prompt"], row["ordinal"])
        for idx, row in test_df.iterrows()
    ]
    if args.debug:
        print(f"--> run_survey: data_list_test example = {data_list_test[0]}")
        import pdb; pdb.set_trace()
    
    # run the inference
    results, model_name = inference_offline(
        args, data_list_test, sampling_params, llm, lora_idx
    )
    if args.debug:
        print(f"--> run_survey: results example = {results[0]}")
        import pdb; pdb.set_trace()
    
    # save the inference result to dataframe
    for idx, prob_sum, probs in results:
        test_df.at[idx, "llm_dist"] = probs
        test_df.at[idx, "llm_prob_sum"] = prob_sum
        human_probs = list_normalize(
            ast.literal_eval(test_df.at[idx, "output_dist"])
        )
        ordinal_value = ast.literal_eval(test_df.at[idx, "ordinal"])
        test_df.at[idx, "emd"] = ordinal_emd(
            list_1=human_probs,
            list_2=probs,
            ordinal_value=ordinal_value,
        )
    if args.debug:
        print(f"--> run_survey: test_df example = {test_df.iloc[0]}")
        import pdb; pdb.set_trace()

    # save output
    model_name = model_name.split("/")[-1]
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    test_df.to_csv(
        args.output_dir
        + args.input_paths[lora_idx].split("/")[-1].split(".")[0]
        + f"_{model_name}.csv",
        index=False,
    )


def cli_args_parser():
    parser = argparse.ArgumentParser()

    # common arguments: input file path and output directory
    parser.add_argument("--input_paths", type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str)

    # offline inferene arguments
    # please refer to vllm (https://github.com/vllm-project/vllm) for engine arguments.
    parser.add_argument(
        "--base_model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf"
    )
    parser.add_argument("--is_chat", action="store_true")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_logprobs", type=int, default=20)
    parser.add_argument("--enable_prefix_caching", type=bool, default=True)
    parser.add_argument("--enforce_eager", type=bool, default=True)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--lora_path", type=str, nargs="+", default=None)
    parser.add_argument("--lora_name", type=str, nargs="+", default=None)

    # debug flag is provided for better understanding of intermediate artifacts.
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":

    """
    Run the inference on the test set and save outputs to the output directory.

    The following arguments are mainly required:
        --input_paths: list of the test files (can be more than 1 file)
        --output_dir: directory to save the output (all results will be saved here)
        --base model: the base model. Refer to huggingface
            additionally, if it is chat model, turn on the --is_chat flag.
        --lora_path: list of the LoRA path (optional)
        --lora_name: list of the LoRA name (optional)

    Given N input files and M LoRA paths, the script will run N*M inferences.
    Must specify the unique lora_name for each lora_path.
    For running inference with base model, leave lora_name and lora_path empty.
    In this case, the script will run N inferences.
    """

    args = cli_args_parser()

    # check argument consistency
    assert args.input_paths is not None, "Input paths should be provided."
    assert args.output_dir is not None, "Output directory should be provided."
    if args.lora_name is None and args.lora_path is not None:
        raise ValueError("LoRA name should be provided when LoRA path provided.")
    if args.lora_name is not None:
        assert len(args.lora_name) == len(
            args.lora_path
        ), "LoRA name and LoRA path should have the same length."

    # get the LLM engine
    sampling_params, llm = get_llm_engine(args)

    # lora_name is optional, if not provided, it will run the base model
    n_lora = 1 if args.lora_name is None else len(args.lora_name)
    total_runs = len(args.input_paths) * n_lora
    print(f"--> run_inference: total runs = {total_runs}")
    if args.lora_name is not None:
        args.lora_name = args.lora_name * len(args.input_paths)
        args.lora_path = args.lora_path * len(args.input_paths)
    args.input_paths = sorted(args.input_paths * n_lora)

    # run inference for each combination of input file and LoRA module
    for lora_idx in range(total_runs):
        run_survey(args, sampling_params, llm, lora_idx)
