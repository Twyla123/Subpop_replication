import json
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

from subpop.utils.surveydata_utils import ActualSurveyData


def process_qkey(qkey: str, wave_number: int) -> Tuple[str, Dict, str]:
    """
    Args:
        qkey: question identifier string
        wave_number: wave number the question belongs to
    Returns:
        tuple: (qkey, refined_data, error flag)
    """
    surveydata = ActualSurveyData(
        wave_list=[wave_number],
        bank_qkeys=set(),
        query_qkeys=set(),
        data_dir=ROOT_DIR / "data" / "subpop-train",
    )
    try:
        print(f"--> process_qkey: working on qkey {qkey}.")
        original_qbody = surveydata.fetch_question_body(qkey)
        refined_qbody = surveydata.refine_question_body(qkey)
        return (
            qkey,
            {"original_qbody": original_qbody, "refined_qbody": refined_qbody},
            None,
        )
    except Exception as e:
        print(f"--> process_qkey: failed to work on qkey {qkey}.")
        return qkey, None, str(e)


if __name__ == "__main__":

    ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
    input_dir = ROOT_DIR / "data" / "subpop-train"
    os.makedirs(input_dir / "processed", exist_ok=True)

    with open(input_dir / f"subpop-train_parsed-qkeys.json", "r") as f:
        qkey_dict = json.load(f)
        for wave_number, qkeys in qkey_dict.items():
            qkey_dict[wave_number] = list(set(qkeys))

    refined_qbody_dict = {}
    error_qkeys_list = []

    for wave_idx, qkeys_list in qkey_dict.items():
        print(f"--> main: working on wave {wave_idx}.")
        wave_number = int(wave_idx.replace("W", ""))

        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(process_qkey, qkey, wave_number): qkey
                for qkey in qkeys_list
            }
            for future in as_completed(futures):
                qkey = futures[future]
                try:
                    qkey, refined_data, error = future.result()
                    if refined_data:
                        refined_qbody_dict[qkey] = refined_data
                    if error:
                        error_qkeys_list.append(qkey)
                except Exception as e:
                    print(f"--> main: unhandled exception for qkey {qkey}: {e}")
                    error_qkeys_list.append(qkey)

        with open(input_dir / "processed" / f"refined_qkey_dict.json", "w") as f:
            json.dump(refined_qbody_dict, f, indent=4)
        if error_qkeys_list:
            with open(input_dir / "processed" / f"error_qkeys_list.json", "w") as f:
                json.dump(error_qkeys_list, f, indent=4)
