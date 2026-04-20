from enum import Enum

class SteeringPromptType(Enum):
    BIO = ["bio_prompt"]
    QA = ["qa_prompt"]
    PORTRAY = ["portray_prompt"]
    ALL = ["bio_prompt", "qa_prompt", "portray_prompt"]