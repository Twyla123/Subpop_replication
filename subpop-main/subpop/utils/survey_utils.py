from typing import List, Any
from collections import Counter

import numpy as np

QA_FORMAT = "{question}\n\n{respondent}"

QUESTION_FORMAT = """{surveyor} {question_body}
{option_list}
{additional_instruction}"""


def ordinal_emd(
    list_1: List[float],
    list_2: List[float],
    ordinal_value: List[float],
) -> float:
    """
    Measure Wasserstein distance between two ordinal distributions.
    Args:
        list_1, list_2: two lists of floats representing the distributions
        ordinal_value: a list of floats representing the ordinal values
    Returns:
        float: Wasserstein distance between list_1 and list_2
    Example 1:
        list_1 = [0.1, 0.5, 0.4]
        list_2 = [0.2, 0.3, 0.5]
        ordinal_value = [1.0, 2.0, 1.5]
        WD = {|0.1-0.2| * (1.5-1.0) + |0.5-0.7| * (2.0-1.5)} / (2.0-1.0)
    Example 2:
        list_1 = [0.2, 0.5, 0.3]
        list_2 = [0.3, 0.4, 0.3]
        ordinal_value = [1.0, 2.0, -1.0]
        (-1.0 indicates no ordinality, e.g. 'not sure'. ocassionaly exist in SubPOP-eval)
        WD = {|0.2-0.3| * (2.0-1.0)} / (2.0-1.0)
    """
    assert len(list_1) == len(list_2), "-->ordinal_emd: two lists should have same legnth."

    # in case of no ordinality information, return nan
    if max(ordinal_value) == min(ordinal_value):
        return np.nan

    # sort by ordinality information
    ordinal_value, list_1, list_2 = zip(*sorted(zip(ordinal_value, list_1, list_2)))
    # find first non-negative ordinal_value index
    try:
        non_neg_idx = next((i for i, val in enumerate(ordinal_value) if val >= 0), 0)
        ordinal_value = ordinal_value[non_neg_idx:]
        list_1 = list_normalize(list_1[non_neg_idx:])
        list_1 = [1 / len(list_1)] * len(list_1) if sum(list_1) == 0 else list_1
        list_2 = list_normalize(list_2[non_neg_idx:])
        list_2 = [1 / len(list_2)] * len(list_2) if sum(list_2) == 0 else list_2
    except:
        return np.nan

    cum_dist_1 = np.cumsum(list_1)
    cum_dist_2 = np.cumsum(list_2)
    emd = 0
    for i in range(len(list_1) - 1):
        emd += abs(cum_dist_1[i] - cum_dist_2[i]) * (
            ordinal_value[i + 1] - ordinal_value[i]
        )
    return emd / (max(ordinal_value) - min(ordinal_value))


def generate_mcq(
    question_body: str,
    options: List[str],
    surveyor: str = "Question:",
    respondent: str = "Answer:",
    pre_label: str = "",
    post_label: str = ". ",
    add_answer_forcing: bool = False,
    additional_instruction: str = "",
) -> str:
    """
    Generate a multiple choice question format.
    Args:
        question_body: the question text
        options: a list of options
        surveyor: entity asking the question ('Question:', 'Surveyor:', etc.)
        respondent: entity answering the question ('Answer:', 'Respondent:', etc.)
        pre_label: the label before the option
        post_label: the label after the option
            - if pre_label = '(' and post_label = ')', the options will be formatted as (A).
        add_answer_forcing: whether to add an additional instruction to answer as a choice
            - Example: Answer as a choice between A,B,...
        additional_instruction: additional instruction to answer as a choice
    Returns:
        a QA-formatted string    
    """
    def generate_option(
        options: List[str],
        pre_label: str,
        post_label: str,
    ) -> str:
        return "\n".join(
            [
                f"{pre_label}{chr(ord('A') + i)}{post_label}{option.strip()}"
                for i, option in enumerate(options)
            ]
        ).strip()

    if add_answer_forcing:
        additional_instruction = (
            "Answer as a choice between "
            + ",".join(
                [
                    f"{pre_label}{chr(ord('A')+ i)}{post_label}".strip()
                    for i in range(len(options))
                ]
            ).strip()
            + ""
            if post_label.strip() == "."
            else "."
        )

    return QA_FORMAT.format(
        question=QUESTION_FORMAT.format(
            surveyor=surveyor.strip(),
            question_body=question_body.strip(),
            option_list=generate_option(
                options=options,
                pre_label=pre_label,
                post_label=post_label,
            ),
            additional_instruction=additional_instruction.strip(),
        ).strip(),
        respondent=respondent.strip(),
    )


def list_normalize(l: List[float]) -> List[float]:
    """normalize a list of floats to sum to 1.0"""
    if np.isclose(sum(l), 0):
        raise ValueError("--> list_normalize: sum of list is 0.")
    return [i / sum(l) for i in l]


def get_entropy(x: List[Any], norm: bool=True) -> float:
    """
    Calculate entropy of a list x.
    Args:
        x: a list of items, typically survey responses (ex. ['A','C','A','A','B'])
        norm: whether to normalize the entropy by the maximum entropy
    Returns:
        float: entropy of the list x
    """
    assert len(x) > 0, "-> get_entropy: list is empty."
    counts = Counter(tuple(item) for item in x)
    counts = np.array(list(counts.values()))
    counts = counts / np.sum(counts)
    entropy = -np.sum(counts * np.log2(counts + 1e-9))
    if not norm or counts.shape[0] == 1:
        return entropy
    return entropy / np.log2(counts.shape[0])