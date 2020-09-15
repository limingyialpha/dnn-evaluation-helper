from pathlib import Path
from typing import List

from app.questionnaire.questionnaire import Questionnaire


class QuestionnaireStatsGenerator:
    """This class generates statistics from a collection of questionnaires"""

    def __init__(self, q_list: List[Questionnaire], target_path: Path):
        self._q_list = q_list
        self._target_path = target_path

    # TODO
    def generate_labelled_images(self):
        pass

    # TODO
    def generate_aggregated_data_excel(self):
        pass
