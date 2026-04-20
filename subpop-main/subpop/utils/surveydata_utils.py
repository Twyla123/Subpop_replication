import os
import pathlib
import multiprocessing as mp
from collections import Counter
from functools import partial
from typing import List, Dict, Any, Tuple, Optional, Union, Set

import pandas as pd
import pyreadstat
import pickle
import numpy as np
import torch
import openai
import openai._exceptions as openai_exceptions
from tqdm import tqdm
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

from subpop.utils import survey_utils
from subpop.utils.backoff import retry_with_exponential_backoff

NEG_INF = -1e9
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

PROHIBITED_WAVES = [26, 29, 32, 34, 36, 41, 42, 43, 45, 49, 50, 54, 82, 92]
# PROHIBITED_WAVES: used in the opininoqa dataset.

REMOVED_WAVES = [47, 58, 59, 60, 63, 66, 61, 62, 64, 65, 67, 86, 89, 97, 100, 122, 132]
# PROHIBITED_WAVES: removed waves not included in SubPOP-Train
# wave 47 : non-standard categories for education
# wave 58 : income level not available
# wave 59 : categories for political affiliation not availble
# wave 60 : wave does not exist
# wave 63, 86 : political ideology not available
# wave 66 : religion not available
# wave 61, 62, 64, 65, 67, 100, 122 : racethnmod not available
# wave 132: data not publicly available
# wave 89, 97: fine-grained education level not availble

PROHIBITED_QKEY_PREFIXES = ['partyln']

GSS_ATTRIBUTE_TO_VARIABLE_MAP = {
    'CREGION': 'region',
    'EDUCATION': 'educ',
    'SEX': 'sexnow1',
    'POLIDEOLOGY': 'polviews',
    'INCOME': 'income16',
    'POLPARTY': 'partyid',
    'RACE': 'raceacs',
    'RELIG': 'relig',
}

GSS_VALUE_TO_LABEL_MAP = {
    'CREGION': {
        'Northeast': [1,2],
        'South': [5,6,7],
        'Midwest': [3,4],
        'West': [8,9],
    },
    'EDUCATION': {
        'College graduate/some postgrad': [16,17,18,19,20],
        'Less than high school': [0,1,2,3,4,5,6,7,8,9,10,11],
    },
    'SEX': {
        'Male': [1],
        'Female': [2],
        'Transgender': [3],
    },
    'POLIDEOLOGY': {
        'Liberal': [1,2,3],
        'Conservative': [5,6,7],
        'Moderate': [4],
    },
    'INCOME': {
        '$100,000 or more': [22,23,24,25,26],
        'Less than $30,000': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    },
    'POLPARTY': {
        'Democrat': [0,1],
        'Republican': [5,6],
    },
    'RACE': {
        'White': [1],
        'Black': [2],
        'Asian': [4,5,6,7,8,9,10],
        'Hispanic': [16],
    },
    'RELIG': {
        'Protestant': [1],
        'Jewish': [3],
        'Hindu': [7],
        'Atheist': [4],
        'Muslim': [9],
    },
}

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

AGE_COLUMN = ['F_AGECAT_FINAL', 'F_AGECAT']
SEX_COLUMN = ['F_SEX_FINAL', 'F_SEX', 'F_GENDER']
EDUCATION_COLUMN = ['F_EDUCCAT2']
CREGION_COLUMN = ['F_CREGION']
POLPARTY_COLUMN = ['F_PARTY_FINAL']
POLIDEOLOGY_COLUMN = ['F_IDEO']
INCOME_COLUMN = ['F_INCOME', 'F_INC_SDT1']
RELIG_COLUMN = ['F_RELIG']
RACE_COLUMN = ['F_RACETHNMOD']

"""
Conversion table for demographic and ideology fields.
keys: actual demographic field name in the .sav file of American Trends Panel
      (i.e. how the field is coded by the survey institution)
values: a dictionary of (trait explained in plain text) : (trait coded in the .sav file)
"""
CONVERSION_TABLE = {
    # age
    'f_agecat_final' : {
        '18-29' : '18-29', 
        '30-49' : '30-49', 
        '50-64' : '50-64', 
        '65+' : '65+'
    },
    'f_agecat' : {
        '18-29' : '18-29',
        '30-49' : '30-49',
        '50-64' : '50-64',
        '65+' : '65+'
    },
    # sex or gender
    'f_sex_final' : {
        'male' : 'male',
        'female' : 'female'
    },
    'f_sex' : {
        'male' : 'male',
        'female' : 'female'
    },
    'f_gender' : {
        'male' : 'a man',
        'female' : 'a woman'
    },
    # education
    'f_educcat2' : {
        'less than high school' : 'less than high school',
        'high school graduate' : 'high school graduate',
        'some college, no degree' : 'some college, no degree',
        "associate's degree" : "associate's degree",
        'college graduate/some postgrad' : 'college graduate/some post grad',
        'postgraduate' : 'postgraduate'
    },
    # region
    'f_cregion' : {
        'northeast' : 'northeast',
        'midwest' : 'midwest',
        'south' : 'south',
        'west' : 'west'
    },
    # political affiliation
    'f_party_final'  : {
        'republican' : 'republican', 
        'democrat' : 'democrat',
        'independent' : 'independent',
        'something else' : 'something else'
    },
    # political ideology
    'f_ideo' : {
        'very conservative' : 'very conservative',
        'conservative' : 'conservative',
        'moderate' : 'moderate',
        'liberal' : 'liberal',
        'very liberal' : 'very liberal'
    },
    # income
    'f_income': {
        'less than $30,000': [
            'less than $10,000',
            '$10,000 to less than $20,000',
            '$20,000 to less than $30,000'
        ],
        '$30,000-$50,000': [
            '$30,000 to less than $40,000',
            '$40,000 to less than $50,000'
        ],
        '$50,000-$75,000': ['$50,000 to less than $75,000'],
        '$75,000-$100,000': ['$75,000 to less than $100,000'],
        '$100,000 or more': [
            '$100,000 to less than $150,000',
            '$150,000 or more'
        ],
    },
    'f_inc_sdt1': {
        'less than $30,000': ['less than $30,000'],
        '$30,000-$50,000': [
            '$30,000 to less than $40,000',
            '$40,000 to less than $50,000'
        ],
        '$50,000-$75,000': [
            '$50,000 to less than $60,000',
            '$60,000 to less than $70,000',
            '$70,000 to less than $80,000'
        ],
        '$75,000-$100,000': [
            '$80,000 to less than $90,000',
            '$90,000 to less than $100,000'
        ],
        '$100,000 or more': ['$100,000 or more'],
    },
    # religion
    'f_relig' : {
        'protestant' : 'protestant', 
        'roman catholic' : 'roman catholic',
        'mormon' : 'mormon',
        'orthodox' : 'orthodox',
        'jewish' : 'jewish',
        'muslim' : 'muslim',
        'buddhist' : 'buddhist',
        'hindu' : 'hindu',
        'atheist' : 'atheist',
        'agnostic' : 'agnostic',
        'other' : 'other',
        'nothing in particular' : 'nothing in particular'
    },
    # race or ethnicity
    'f_racethnmod' : {
        'white' : 'white non-hispanic',
        'black' : 'black non-hispanic',
        'hispanic' : 'hispanic',
        'other' : 'other',
        'asian' : 'asian non-hispanic'
    }
}

class ActualSurveyData:
    """
    class ActualSurveyData handles the human response data
    providing a unified interface for retrieving questions and response distributions.
    """

    def __init__(
        self,
        wave_list: List[int],
        bank_qkeys: Set[str],
        query_qkeys: Set[str],
        data_dir: Union[str, pathlib.Path],
        relevance_data: Optional[Dict[Tuple[str, Union[str, List[str]]], float]] = None,
        refined_qbody_data: Optional[Dict[str, Dict[str, str]]] = None,
        n_sample_threshold: int = 100,
    ):
        self.wave_list = wave_list

        # load the response data and the survey metadata
        self.wave_data: Dict[int, pd.DataFrame] = {}
        self.wave_meta: Dict[int, Dict] = {}
        for i in wave_list:
            file_path = data_dir / f"ATP W{i}.sav"
            assert os.path.exists(file_path), f"Wave {i} data not found."
            self.wave_data[i], self.wave_meta[i] = pyreadstat.read_sav(file_path)

        # define qkeys for question bank and query
        self.bank_qkeys = bank_qkeys
        self.query_qkeys = query_qkeys
        for qkey in self.bank_qkeys | self.query_qkeys:
            assert self.get_wave_number(qkey) in wave_list, f"Invalid qkey: {qkey}"

        # load the precomputed relevance data if available
        self.relevance_data: Dict[Tuple[str, Union[str, List[str]]], float] = {}
        if relevance_data is not None:
            self.relevance_data = relevance_data
        self.refined_qbody_data: Dict[str, Dict[str, str]] = {}
        if refined_qbody_data is not None:
            self.refined_qbody_data = refined_qbody_data

        # miscellanous constants - worker count, sample threshold, lock
        self.N_PROCESS = mp.cpu_count()
        self.N_SAMPLE_THRESHOLD = n_sample_threshold
        self.lock = mp.Manager().Lock()
        self.openai_client = openai.OpenAI()

    @staticmethod
    def get_wave_number(qkey: str) -> int:
        return int(qkey.split("_W")[-1])

    @retry_with_exponential_backoff(
        max_retries=20,
        no_retry_on=(
            openai_exceptions.AuthenticationError,
            openai_exceptions.BadRequestError,
        ),
    )        
    def refine_question_body(self, qkey: str) -> str:
        """
        Refine the qkey with LLMs to make the question sensible in stand-alone.
        """
        mcq = self.get_mcq(qkey)
        mcq = mcq.split("Refused")[0]
        qbody = mcq.rsplit("\n", 1)[0].strip()
        instruction = (
            "Instruction: Refine the question with a minimal change to make the question sensible."
            + " Do not modify options, and do not modify a question if it makes sense. Always start your answer with \"Refined question:\"."
        )
        in_context_examples = [ # provide four in-context examples for few-shot prompting
            """Question: A cross // Do you have any of the following for spiritual purposes?
A. Yes, I have this for spiritual purposes
B. No, I do not have this for spiritual purposes

Refined question: Do you have a cross for spiritual purposes?""",
            """Question: As you may know, same-sex marriage is now legal in the U.S. Do you think this is [a good thing or a bad thing] for our society?
A. Very good thing
B. Somewhat good thing
C. Somewhat bad thing
D. Very bad thing

Refined question: As you may know, same-sex marriage is now legal in the U.S. Do you think this is a good thing or a bad thing for our society?""",
            """Question: On a different subject…How much, if at all, do white people benefit from advantages in society that black people do not have
A. A great deal
B. A fair amount
C. Not too much
D. Not at all

Refined question: How much, if at all, do white people benefit from advantages in society that black people do not have?""",
            """Question: Thinking about the past couple of weeks, would you say the news for Donald Trump has been...
A. Very good
B. Mostly good
C. Neither good nor bad
D. Mostly bad
E. Very bad

Refined question: Thinking about the past couple of weeks, would you say the news for Donald Trump has been...""",
        ]
        prompt = instruction + "\n\n" + "\n\n".join(in_context_examples) + "\n\n" + qbody
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a public opinion survey question designer."
                },
                { "role": "user", "content": prompt}
            ],
            temperature=0.0,
        ).choices[0].message.content
        return response.replace("Refined question:", "").strip()

    def fetch_options(self, qkey: str) -> Dict[float, str]:
        """
        Given a qkey, fetch options and return in a dictionary
        key is human identifier (float), value is option (str)
        Example: {1.0: 'Major', ..., 5.0: 'Minor', 99.0: 'Refused'}
        """
        return self.wave_meta[self.get_wave_number(qkey)].variable_value_labels[qkey]

    def fetch_question_body(self, qkey: str) -> str:
        """
        Given a qkey, fetch a question body.
        """
        if self.refined_qbody_data.get(qkey):
            return self.refined_qbody_data[qkey]["refined_qbody"].strip()
        qbody = self.wave_meta[self.get_wave_number(qkey)].column_names_to_labels[qkey]
        return qbody.partition(".")[2].strip()

    def get_mcq(self, qkey: str, chat_template: bool=False) -> Union[str, List[Dict]]:
        """
        Given a qkey, generate multiple choice question format.
        Args:
            qkey: qusetion identifier string
            chat_template: whether to format MCQ compatible with a chat model input.
        Returns:
            str: MCQ prompt, when chat_template is false.
            List[Dict]: MCQ prompt, when chat_template is true.
        """
        return_mcq = survey_utils.generate_mcq(
            question_body=self.fetch_question_body(qkey),
            options=list(self.fetch_options(qkey).values()),
            add_answer_forcing=True,
        )
        if chat_template:
            return [{'role': "user", "content": return_mcq.split('Answer:')[0].strip()}]
        return return_mcq

    def fetch_n_respondent(self, qkey: str, remove_refusal: bool = True) -> int:
        """
        Given a qkey, get the number of valid respondents (not a no-response)
        By default, does not count refusals as valid responses.
        Args:
            qkey: question key
            remove_refusal: whether to remove refusal from a valid response
        Returns:
            the number of valid respondents
        """
        responses = self.wave_data[self.get_wave_number(qkey)][qkey]
        # the last option for ATP is the refusal option
        refusal = list(self.fetch_options(qkey).keys())[-1]
        if remove_refusal:
            return np.sum(~np.isnan(responses) & (responses != refusal))
        return np.sum(~np.isnan(responses))

    def convert_field(
        self, wave_number: int, attribute: str, trait: str
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Convert the attribute (ex. AGE) to the field name in dataframe (ex. AGECAT)
        and also the trait (ex. 18-29) to the coded value (ex. 1.0).
        Args:
            wave_number: wave number (ex. 92)
            attribute: demographic information (ex. 'AGE')
            trait: subgroup information (ex. '18-29')
        Returns:
            field name and the coded value for the given attribute and trait in text.
        List of valid attribute and trait inputs
        (Based on available demographic / ideology categories of American Trends Panel)
            attribute - age:
                trait - 18-29, 30-49, 50-64, 65+
            attribute - sex:
                trait - male, female
            attribute - education:
                trait - less than high school,
                        high school graduate,
                        some college, no degree,
                        associate's degree,
                        college graduate/some postgrad,
                        postgraduate
            attribute - region:
                trait - northeast, midwest, south, west
            attribute - polparty:
                trait - republican, democrat, independent, something else
            attribute - polideo:
                trait - very conservative, conservative, moderate, liberal, very liberal
            attribute - income:
                trait - less than $30,000,
                        $30,000-$50,000,
                        $50,000-$75,000,
                        $75,000-$100,000,
                        $100,000 or more
            attribute - religion:
                trait - protestant,
                        roman catholic,
                        mormon,
                        orthodox,
                        jewish,
                        muslim,
                        buddhist,
                        hindu,
                        atheist,
                        agnostic,
                        other,
                        nothing in particular
            attribute - race:
                trait - white non-hispanic, black non-hispanic, hispanic,
                        other, asian non-hispanic
        """
        # overall (no specific subpopulation) case
        if attribute.lower() == "overall" and trait.lower() == "overall":
            return None, None
        column_list = globals().get(f"{attribute.upper()}_COLUMN", None)
        assert column_list is not None, f"--> convert_field: invalid attribute = {attribute}"

        # loop through the column list to find the attribute coded in the wave
        # how the attribute is coded by Pew Research in the wave data is wave-specific.
        # for example, gender is coded in either 'F_SEX' or 'F_GENDER' depending on the wave.
        attribute_coded = next((
                col for col in column_list 
                if col in self.wave_data[wave_number].columns
            ), None)
        assert attribute_coded, f"--> convert_field: column for {attribute} not found."

        # convert the trait in plain text (ex. '18-29') to the coded value (ex. 1.0)
        trait = CONVERSION_TABLE[attribute_coded.lower()].get(trait.lower(), None)        
        options = self.wave_meta[wave_number].variable_value_labels[attribute_coded]
        if isinstance(trait, str):
            trait_coded = next((
                key for key, value in options.items()
                if value.lower().startswith(trait.lower())
                ), None)
        elif isinstance(trait, list):
            trait_coded = [
                key for key, value in options.items()
                if value.lower() in [t.lower() for t in trait]
            ]
        assert trait_coded, f"--> convert_field: trait {trait} not found."

        return attribute_coded, trait_coded

    def fetch_wave_respondent(
        self,
        wave_number: int,
        attribute: Union[str, List[str]] = "Overall",
        trait: Union[str, List[str]] = "Overall",
    ) -> np.ndarray:
        """
        Get a list of respondents (human identifider, QKEY) satisfying the attribute and trait
        Args:
            wave_number: wave number (ex. 92)
            attribute: demographic information (ex. 'AGE', ['AGE', 'CREGION'])
            trait: subgroup information (ex. '18-29', ['18-29', 'Midwest'])
        Returns:
            numpy array of human identifiers (QKEY) from wave satisfying the attribute and trait
        """

        if isinstance(attribute, str):
            assert isinstance(trait, str), "Attribute and trait must be both str or list."
            attribute = [attribute]
            trait = [trait]
        elif isinstance(attribute, list):
            assert isinstance(trait, list), "Attribute and trait must be both str or list."
            assert len(attribute) == len(trait), "Attribute and trait must have the same length."
        else:
            raise ValueError("Attribute and trait must be str or list.")

        # convert the natural language trait to the coded value, as provided in the raw data.
        attribute_trait_coded_list = [
            self.convert_field(wave_number, a, t)
            for a, t in zip(attribute, trait)
        ]

        # get a subset of wave respondent by filtering the attribute and trait
        respondent_df = self.wave_data[wave_number]
        if attribute_trait_coded_list is not None:
            for attribute_coded, trait_coded in attribute_trait_coded_list:
                if attribute_coded is not None:
                    if isinstance(trait_coded, float):
                        respondent_df = respondent_df[
                            respondent_df[attribute_coded] == trait_coded
                        ]
                    elif isinstance(trait_coded, list):
                        respondent_df = respondent_df[
                            respondent_df[attribute_coded].isin(trait_coded)
                        ]

        # return the list of human identifiers satisfying attribute and trait
        return respondent_df["QKEY"].values

    def fetch_wave_weights(self, wave_number: int) -> Dict[float, float]:
        """
        Get a list of weights per each respondent in the wave.
        Args: wave_number: wave number (ex. 92)
        Returns: dictionary of human identifiers (QKEY) and their weights
        """
        columns = self.wave_data[wave_number].columns
        # get the weight information: it is encoded by 'Weight_W{wave_number}' format.
        weight_column = [
            col for col in columns
            if "weight_" in col.lower() and f"_w{wave_number}" in col.lower()
        ][0]
        return dict(zip(
                self.wave_data[wave_number]["QKEY"].values,
                self.wave_data[wave_number][weight_column].values,
        ))

    def fetch_response_data(
        self,
        qkey: str,
        attribute: Union[str, List[str]] = "Overall",
        trait: Union[str, List[str]] = "Overall",
        remove_refusal: bool = True,
    ) -> Dict[float, float]:
        """
        Given a question (qkey) and subpopulation information, return individual response list.
        Individuals in the response list are filtered by the subpopulation information.
        Args:
            Qkey and demographics of interest (attribute, trait)
            Whether to include refusals in the response list (remove_refusal)
        Returns:
            response_data: dictionary, key = human identifier number, value = response in float
            For example, {10123195.0: 1.0, 12129712.0: 3.0, ...}
        """
        if isinstance(attribute, str):
            assert isinstance(trait, str), "--> fetch_response_data: arg be both str or list."
        elif isinstance(attribute, list):
            assert isinstance(trait, list), "--> fetch_response_data: arg be both str or list."
            assert len(attribute) == len(trait), "--> fetch_response_data: arg be same length."

        wave_number = self.get_wave_number(qkey)
        response_data = list(zip(
                self.wave_data[wave_number]["QKEY"].values, # human identifier
                self.wave_data[wave_number][qkey].values, # response in float
        ))
        # remove nan(no-response) (always) and refusal response (optional, flag remove_refusal))
        response_data = [
            x for x in response_data if 
            isinstance(x[1], (int, float)) and not np.isnan(x[1])
        ]
        if remove_refusal:
            refusal = list(self.fetch_options(qkey).keys())[-1]
            response_data = [x for x in response_data if x[1] != refusal]

        # filter by the attribute and trait (subgroup) information
        subgroup_respondents = self.fetch_wave_respondent(
            wave_number, attribute, trait
        )
        response_data = [x for x in response_data if x[0] in subgroup_respondents]
        return dict(response_data)

    def fetch_joint_response_data(
        self,
        qkeys: List[str],
        attribute: Union[str, List[str]] = "Overall",
        trait: Union[str, List[str]] = "Overall",
        remove_refusal: bool = True,
    ) -> Dict[float, List[float]]:
        """
        Joint version of fetch_response_data.
        Given a list of questions (qkeys) and a subpopulation information,
        return a dictionary of human_id and their joint responses to the questions.
        Args:
            qkeys: list of question keys
            attribute: demographic information (ex. 'AGE', ['AGE', 'CREGION'])
            trait: subgroup information (ex. '18-29', ['18-29', 'Midwest'])
            remove_refusal: whether to remove refusal responses
        Returns:
            joint_response_data: key = human identifier number, value = list of responses
            For example, {10123195.0: [1.0, 3.0, ...], 12129712.0: [3.0, 2.0, ...], ...}
        """
        if not isinstance(qkeys, list) or len(qkeys) == 0:
            raise ValueError(f"--> fetch_joint_response_data: invalid qkeys list = {qkeys}")

        response_datas = [
            self.fetch_response_data(qkey, attribute, trait, remove_refusal)
            for qkey in qkeys
        ]
        human_intersection: Set[float] = set(response_datas[0])
        for response_data in response_datas[1:]:
            human_intersection &= set(response_data)
        return {
            human_id: [response_data[human_id] for response_data in response_datas]
            for human_id in human_intersection
        }

    def fetch_response_distribution(
        self,
        qkey: str,
        attribute: Union[str, List[str]] = "Overall",
        trait: Union[str, List[str]] = "Overall",
        remove_refusal: bool = True,
        use_weight: bool = True,
    ) -> Optional[Dict[float, float]]:
        """
        Get response distribution for a question (qkey) for subpopulation (attribute, trait)
        Args:
            qkey: question key
            attribute: demographic information (ex. 'AGE', ['AGE', 'CREGION'])
            trait: subgroup information (ex. '18-29', ['18-29', 'Midwest'])
            remove_refusal: whether to remove refusal responses
            use_weight: whether to use the weight information
        Returns:
            response distribution: dictionary
                key = response option, value = probability
                For example, {1.0: 0.2, 2.0: 0.3, 3.0: 0.5}
            None if no valid response
                happens when a question is not asked to a particular subpopulation
                or no population satisfies the subpopulation information (ex. Asian & Jewish)
        """
        response_data = self.fetch_response_data(
            qkey=qkey,
            attribute=attribute,
            trait=trait,
            remove_refusal=remove_refusal
        )
        weight_data = self.fetch_wave_weights(self.get_wave_number(qkey))

        options = list(self.fetch_options(qkey))
        options = options[:-1] if remove_refusal else options
        resp_dist: Dict[float, float] = {option: 0.0 for option in options}
        for human_id, response in response_data.items():
            resp_dist[response] = resp_dist[response] + (
                (
                    weight_data[human_id] 
                    if not np.isnan(weight_data[human_id]) else 0.0
                ) if use_weight else 1.0
            )
        if len(resp_dist) == 0 or sum(resp_dist.values()) == 0:
            return None
        resp_dist = {k: v / sum(resp_dist.values()) for k, v in resp_dist.items()}
        return dict(sorted(resp_dist.items()))

    def get_entropy(
        self,
        qkey: Union[str, List[str]],
        remove_refusal: bool = True
    ) -> Tuple[str, float]:
        """
        Get an entropy of either
            a response distribution given a question
            a joint response distribution given a list of questions
        Args:
            qkey: question indentifier string
            remove_refusal: whether to include refusal as a option.
        Returns:
            string, float: qkey and an entropy of response distribution.
        """
        if isinstance(qkey, str):
            qkey = [qkey]
        resp_dist = self.fetch_joint_response_data(qkey, remove_refusal=remove_refusal)
        return qkey, survey_utils.get_entropy(list(resp_dist.values()))

    def get_relevance(
        self,
        qkey1: str,
        qkey2: Union[str, List[str]],
        attribute: Union[str, List[str]] = "Overall",
        trait: Union[str, List[str]] = "Overall",
    ) -> float:
        """
        Given a question (qkey1) and a question or a list of questions (qkey2),
        calculate the relevance between the two questions conditioned in subpopulation.
        How to calculate relevance depends on the relevance metric to use.
        For example, we can use the set-wise mutual information (MI) as a metric,
        i.e. MI between qkey1 and qkey2 conditioned on the subpopulation.
        Returns:
            relevance value (float)
        """
        raise NotImplementedError("Subclass must implement this method.")

    def select_top_k(
        self,
        target_qkey: str,
        top_k: int = -1,
        objective: str = "pairwise",  # pairwise, set
        attribute: str = "Overall",
        trait: str = "Overall",
    ) -> List[Tuple[str, float]]:
        """
        Select top k questions based on the relevance data.
        Return qkeys of the top k questions in the smallest to largest order.
        Args:
            target_qkey: target question's qkey
            top_k: number of questions to retrieve
            objective: maximize the sum or pairwise relevance (pairwise)
                       or maximize the set relevance (set)
        Return:
            releveant_qkeys_and_values: list of top_k tuples (qkey, relevance_value)
            in the order of least relevant to most relevant.
        """
        assert target_qkey in self.query_qkeys, f"Qkey {target_qkey} not in query set."
        if top_k == 0:
            return []

        if objective == "pairwise":
            relevant_items = []
            for qkey in self.bank_qkeys:
                if qkey == target_qkey:
                    continue
                relevant_items.append(
                    (
                        qkey,
                        self.get_relevance(target_qkey, qkey, attribute, trait),
                    )
                )
            sorted_relevant_items = list(
                sorted(relevant_items, key=lambda item: item[1], reverse=True)
            )
            return [x for x in sorted_relevant_items if x[1] >= 0][:top_k][::-1]

        if objective == "set":
            relevant_qkey_list: List[str] = []
            relevance_value_list: List[float] = []
            current_relevance: float = 0.0
            while len(relevant_qkey_list) < top_k:
                relevance_gain: List[Tuple[str, float]] = []
                for qkey in self.bank_qkeys:
                    if qkey in relevant_qkey_list or qkey == target_qkey:
                        continue
                    relevance_gain.append(
                        (
                            qkey,
                            self.get_relevance(
                                target_qkey,
                                relevant_qkey_list + [qkey],
                            )
                            - current_relevance,
                        )
                    )
                best_qkey, best_gain = max(relevance_gain, key=lambda x: x[1])
                assert best_gain > 0, "No more gain in relevance."
                relevant_qkey_list.append(best_qkey)
                relevance_value_list.append(best_gain)
                current_relevance += best_gain
            return list(zip(relevant_qkey_list, relevance_value_list))[::-1]

    def get_response(self, human_id: float, qkey: str) -> Tuple[int, str]:
        """
        Get a response of a human (human_id) for a question (qkey)
        Returns:
            index of the response (int), response string (str)
            special cases:
                index = -1 : refusal
                index = -2 : participated for wave, nan response
                index = -3 : not participated for wave
        """

        def get_key_index(dictionary, key):
            keys = list(dictionary.keys())
            if key in keys:
                if keys.index(key) == len(keys) - 1:
                    return -1  # refusal
                return keys.index(key)
            return -2  # nan response

        wave_idx = self.get_wave_number(qkey)
        response = self.wave_data[wave_idx][qkey][
            self.wave_data[wave_idx]["QKEY"] == human_id
        ].values
        if len(response) == 0:
            return -3, "Not participated for wave"
        response = response[0]
        options = self.fetch_options(qkey)
        return get_key_index(options, response), options.get(response, "Nan response")


class ActualSurveyDatawithMI(ActualSurveyData):
    def __init__(
        self,
        wave_list: List[int],
        bank_qkeys: Set[str],
        query_qkeys: Set[str],
        data_dir: Union[str, pathlib.Path] = _REPO_ROOT / "data" / "ATP_raw_data",
        relevance_data: Optional[Dict[Tuple[str, str], float]] = None,
        n_sample_threshold: int = 100,
    ):
        super().__init__(
            wave_list=wave_list,
            bank_qkeys=bank_qkeys,
            query_qkeys=query_qkeys,
            data_dir=data_dir,
            relevance_data=relevance_data,
            n_sample_threshold=n_sample_threshold,
        )

    def get_MI(
        self,
        qkey1: str,
        qkey2: Union[str, List[str]],
        attribute: str = "Overall",
        trait: str = "Overall",
    ) -> float:
        """
        Get mutual information between two questions
        """
        if isinstance(qkey2, str):
            if (qkey1, qkey2) in self.relevance_data:
                return self.relevance_data[(qkey1, qkey2)]
            qkey2 = [qkey2]
        assert [qkey1] != qkey2, "Two questions must be different."

        joint_response = self.fetch_joint_response_data([qkey1] + qkey2, attribute, trait)
        if len(joint_response) < self.N_SAMPLE_THRESHOLD:
            return NEG_INF  # not enough samples to estimate mutual information

        tabular_response = np.array(list(joint_response.values()))
        p_target = tabular_response[:, 0]
        p_joint = np.apply_along_axis(
            lambda x: hash(tuple(x)), axis=1, arr=tabular_response[:, 1:]
        )
        return_value = max(adjusted_mutual_info_score(p_target, p_joint) * np.log2(np.e), 0)
        if len(qkey2) == 1:
            self.relevance_data[(qkey1, qkey2[0])] = return_value
        return return_value

    def get_relevance(
        self,
        qkey1: str,
        qkey2: Union[str, List[str]],
        attribute: str = "Overall",
        trait: str = "Overall",
    ) -> float:
        return self.get_MI(qkey1, qkey2, attribute, trait)
