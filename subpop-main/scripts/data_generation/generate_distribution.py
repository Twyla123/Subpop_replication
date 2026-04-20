import ast
import argparse
import itertools
import json
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any, Union, Optional

import pyreadstat
import pandas as pd
import numpy as np

from subpop.utils.survey_utils import list_normalize
from subpop.utils.surveydata_utils import (
    ActualSurveyData,
    REMOVED_WAVES,
    PROHIBITED_WAVES,
    PROHIBITED_QKEY_PREFIXES,
    GSS_ATTRIBUTE_TO_VARIABLE_MAP,
    GSS_VALUE_TO_LABEL_MAP,
)

"""
explanations of constants:
PROHIBITED_WAVES:
    waves included in OpinionQA dataset, not used for SubPOP-train
REMOVED_WAVES:
    waves missing at least one of demographic attribute information (checked by manual inspection)
PROHIBITED_QKEY_PREFIXES:
    qkeys that are parsed as ask-all questions, but are actually demographic / ideology label
    as we give demographic / ideology label as steering prompts, these questions are not used.            
GSS_ATTRIBUTE_TO_VARIABLE_MAP:
    mapping from attribute (ex. race or ethnicity) to variable name in GSS dataset (ex. raceacs)
GSS_VALUE_TO_LABEL_MAP:
    mapping from variable name (ex. less than high school) to label (ex. 0,1,2,3,4,...,11)
"""

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def process_qkey(
    qkey: str,
    refined_qkey_dict: Dict[str, Dict[str, str]],
    error_qkeys_list: List[str],
    attribute_group_pair: List[Union[Tuple[str, str], Tuple[List[str], List[str]]]],
    atp_waves_list: List[str],
    surveydata_dir: str,
):
    """
    Process a single qkey to generate a dataframe with necessary information
    List of necessary information:
        qkey (str) : question identifier
        attribute (str) : demographic / ideology attribute (ex. Age)
        group (str) : group (ex. 18-29)
        responses (str) : response distribution list, saved as string
        refusal_rate (float) : refusal rate
        ordinal (str) : ordinality information, saved as string (ex. [1.0, 2.0])
        question (str) : question text, refined by LLM
        options (str) : options list, saved as string (ex. ['Agree', 'Disagree', 'Refused])
    Args:
        qkey : question identifier to process
        refined_qkey_dict : dict. of original (from raw data) and refined question (by LLM)
        error_qkeys_list : list of qkeys with error (missing raw data, parsed incorrectly, etc.)
        attribute_group_pair : list of tuples with attribute and group information
        atp_waves_list : list of waves that the qkeys may belong to
        surveydata_dir : directory path to the survey data
    Returns:
        data_to_append (List[Dict[str, Any]]) : list of dictionaries with necessary info.
    """

    print(f"--> process_qkey: Working on {qkey}")
    surveydata = ActualSurveyData(
        wave_list=[int(qkey.split("W")[-1])],
        bank_qkeys=set(),
        query_qkeys=set(),
        refined_qbody_data=refined_qkey_dict,
        data_dir=surveydata_dir,
    )

    question = refined_qkey_dict[qkey]["refined_qbody"]
    options = list(surveydata.fetch_options(qkey).values())
    options = [option.strip() for option in options]
    question_data = []
    for attribute, group in attribute_group_pair:
        if len(attribute) == 1:
            attribute = attribute[0]
            group = group[0]
        try:
            responses = list(
                surveydata.fetch_response_distribution(
                    qkey, attribute, group, remove_refusal=False
                ).values()
            )
            if responses is None:
                continue
            refusal_rate = responses[-1]
            responses = list_normalize(responses[:-1])
            ordinal = [1.0] * len(responses)
            question_data.append(
                {
                    "qkey": qkey,
                    "attribute": str(attribute),
                    "group": str(group),
                    "responses": str(responses),
                    "refusal_rate": refusal_rate,
                    "ordinal": str(ordinal),
                    "question": question,
                    "options": str(options),
                }
            )
        except Exception as e:
            print(f"--> process_qkey: error on {qkey} with {attribute}, {group}: {e}")
            continue
    return question_data


def generate_combined_pairs(loaded_pair, n_combination):
    """Generate a list of combined (joint) demographics according to n_combination"""
    attribute_to_groups = {}
    for attr, group in loaded_pair:
        if attr not in attribute_to_groups:
            attribute_to_groups[attr] = []
        attribute_to_groups[attr].append(group)
    attribute_combinations = list(
        itertools.combinations(attribute_to_groups.keys(), n_combination)
    )
    combined_pairs = []
    for attributes in attribute_combinations:
        group_combinations = itertools.product(
            *[attribute_to_groups[attr] for attr in attributes]
        )
        for groups in group_combinations:
            combined_pairs.append((list(attributes), list(groups)))
    return combined_pairs


def generate_distribution_gss(
    qkey: str,
    option_list: List[str],
    attribute: str,
    group: str,
    surveydata: pd.DataFrame,
    meta: Any,
) -> Optional[Dict[str, float]]:
    """
    Get a response distribution for a given GSS question and subpopualtion.
    Args:
        qkey : question identifier string
        attribute : demographic / ideology attribute (ex. Age)
        group : group (ex. 18-29)
        surveydata : survey data to fetch response distribution
    Returns:
        list: response distribution list
    """

    # get the response and weight for the entire population
    try:
        surveydata_q = surveydata[qkey].values
        surveydata_min_label = np.nanmin(surveydata_q)
    except Exception as e:
        print(f"--> generate_distribution_gss : {qkey} data does not exist.")
        return None
    weight = surveydata["wtssnrps"].values

    # get the response and weight for the subpopulation
    attribute_coded = GSS_ATTRIBUTE_TO_VARIABLE_MAP[attribute]
    group_coded = GSS_VALUE_TO_LABEL_MAP[attribute][group]
    if attribute == "RACE":
        # due to how the race/ethnicity is coded in GSS, special handling required
        subpop_index = np.array([], dtype=int)
        for group_code in group_coded:
            subpop_index = np.append(
                subpop_index,
                np.where(surveydata[attribute_coded + str(group_code)] == 1)[0],
            )
        subpop_index = np.unique(subpop_index)
    else:
        subpop_index = np.where(np.isin(surveydata[attribute_coded], group_coded))[0]
    surveydata_q = surveydata_q[subpop_index]
    weight = weight[subpop_index]

    # aggregrate individual response to get a response distribution
    resp_dist = [0.0 for _ in range(len(option_list))]
    for resp_choice, resp_weight in zip(surveydata_q, weight):
        if isinstance(resp_choice, int) and not np.isnan(resp_weight):
            resp_dist[resp_choice - surveydata_min_label] += resp_weight
    sum_weights = sum(resp_dist)
    resp_dist = [resp / sum_weights for resp in resp_dist]
    return {option_list[i]: resp_dist[i] for i in range(len(option_list))}


def generate_distribution_subpop_eval(args):

    # load the refined question body and option list dictionary
    with open(args.refined_qkey_dict_path, "r") as f:
        refined_qkey_dict = json.load(f)

    # load the survey response data.
    # Note: this survey data has to be downloaded from the official website!
    surveydata, meta = pyreadstat.read_dta(
        REPO_ROOT / "data" / "subpop-eval" / "GSS2022.dta"
    )

    # generate a list of (attribute, group) tuples
    attribute_group_pair: List[Tuple[str, str]] = []
    demographics_data = pd.read_csv(os.path.join(args.demographics_data_path))
    for idx, row in demographics_data.iterrows():
        attribute = row["attribute"]
        group_list = ast.literal_eval(row["group"])
        for group in group_list:
            attribute_group_pair.append((attribute, group))
    attribute_group_pair = generate_combined_pairs(
        attribute_group_pair, args.n_combination
    )
    del demographics_data

    # for each question and subpopulation, generate a response distribution
    question_data = []
    for qkey in refined_qkey_dict.keys():
        question = refined_qkey_dict[qkey]["refined_qbody"]
        options = refined_qkey_dict[qkey]["option_list"]
        ordinal = refined_qkey_dict[qkey]["ordinal"]

        for attribute, group in attribute_group_pair:
            attribute = attribute[0] if len(attribute) == 1 else attribute
            group = group[0] if len(group) == 1 else group
            responses = generate_distribution_gss(
                qkey=qkey,
                option_list=options,
                attribute=attribute,
                group=group,
                surveydata=surveydata,
                meta=meta,
            )
            refusal_rate = responses.get("Refused", 0.0)
            responses = list_normalize(list(responses.values())[:-1])
            question_data.append(
                {
                    "qkey": qkey,
                    "attribute": str(attribute),
                    "group": str(group),
                    "responses": str(responses),
                    "refusal_rate": refusal_rate,
                    "ordinal": str(ordinal),
                    "question": question,
                    "options": str(options),
                }
            )

    surveydata_df = pd.DataFrame(question_data)
    surveydata_df.to_csv(args.output_path, index=False)


def generate_distribution_subpop_train(args):
    """Generate response distribution for each subpopulation and quesetion in SubPOP-train"""
    # load the list of qkeys with error during question text refining step
    if os.path.exists(args.error_qkeys_list_path):
        with open(args.error_qkeys_list_path, "r") as f:
            error_qkeys_list = json.load(f)
    else:
        error_qkeys_list = []

    # load the refined question body dictionary
    with open(args.refined_qkey_dict_path, "r") as f:
        refined_qkey_dict = json.load(f)
    refined_qkey_dict = {
        qkey: refined_qkey_dict[qkey]
        for qkey in refined_qkey_dict
        if (
            (qkey not in error_qkeys_list)
            and (qkey.split("_W")[0].lower() not in PROHIBITED_QKEY_PREFIXES)
            and (int(qkey.split("_W")[-1]) not in PROHIBITED_WAVES + REMOVED_WAVES)
        )
    }

    # generate a list of (attribute, group) tuples
    attribute_group_pair: List[Tuple[str, str]] = []
    demographics_data = pd.read_csv(os.path.join(args.demographics_data_path))
    for idx, row in demographics_data.iterrows():
        attribute = row["attribute"]
        group_list = ast.literal_eval(row["group"])
        for group in group_list:
            attribute_group_pair.append((attribute, group))
    attribute_group_pair = generate_combined_pairs(
        attribute_group_pair, args.n_combination
    )
    del demographics_data

    # process each qkey to generate a list of dictionaries with necessary information
    surveydata_list = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {
            executor.submit(
                process_qkey,
                qkey=qkey,
                refined_qkey_dict={qkey: refined_qkey_dict[qkey]},
                error_qkeys_list=error_qkeys_list,
                attribute_group_pair=attribute_group_pair,
                atp_waves_list=[int(qkey.split("W")[-1])],
                surveydata_dir=pathlib.Path(args.refined_qkey_dict_path).parent.parent,
            ): qkey for qkey in refined_qkey_dict.keys()
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    surveydata_list.extend(result)
            except Exception as e:
                print(f"--> main: error processing qkey {futures[future]}: {e}")

    surveydata_df = pd.DataFrame(surveydata_list)
    surveydata_df.to_csv(args.output_path, index=False)


def cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="subpop-train",
        help="dataset name",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of workers to use for multiprocessing",
    )
    parser.add_argument(
        "--n_combination",
        type=int,
        default=1,
        help="Number of group combinations to consider (default to 1)",
    )
    parser.add_argument(
        "--demographics_data_path",
        type=str,
        default=REPO_ROOT / "data" / "subpopulation_metadata" / "demographics_22.csv",
        help="Path to the demographics data",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = cli_args()
    dataset_name = args.dataset
    args.output_path = REPO_ROOT / "data" / dataset_name / "processed" / f"{dataset_name}.csv"
    args.error_qkeys_list_path = REPO_ROOT / "data" / dataset_name / "processed" / "error_qkeys_list.json"
    args.refined_qkey_dict_path = REPO_ROOT / "data" / dataset_name / "processed" / "refined_qkey_dict.json"

    if dataset_name == "subpop-train":
        generate_distribution_subpop_train(args)
    elif dataset_name == "subpop-eval":
        generate_distribution_subpop_eval(args)
    elif dataset_name == 'opinionqa':
        raise ValueError(
            f"--> main: dataset {args.dataset} is provided by OpinionQA."
            " Please refer to https://github.com/tatsu-lab/opinions_qa."
        )
    else:
        raise ValueError(f"--> main: invalid dataset name {args.dataset}")
