# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from functools import partial

# from subpop.train.datasets.grammar_dataset.grammar_dataset import get_dataset as get_grammar_dataset
# from subpop.train.datasets.alpaca_dataset import InstructionDataset as get_alpaca_dataset
# from subpop.train.datasets.samsum_dataset import get_preprocessed_samsum as get_samsum_dataset 
# from subpop.train.datasets.toxicchat_dataset import get_llamaguard_toxicchat_dataset as get_llamaguard_toxicchat_dataset
from subpop.train.datasets.custom_dataset import get_custom_dataset,get_data_collator, custom_collator_no_labels

DATASET_PREPROC = {
    # "alpaca_dataset": partial(get_alpaca_dataset),
    # "grammar_dataset": get_grammar_dataset,
    # "samsum_dataset": get_samsum_dataset,
    # "llamaguard_toxicchat_dataset": get_llamaguard_toxicchat_dataset,
    "custom_dataset": get_custom_dataset,
    "opnqa_steering_dataset": get_custom_dataset,
    "opnqa_single_demographic_dataset": get_custom_dataset,
}
DATALOADER_COLLATE_FUNC = {
    "custom_dataset": get_data_collator,
    "opnqa_steering_dataset": custom_collator_no_labels,
    "opnqa_single_demographic_dataset": custom_collator_no_labels,
}