from pathlib import Path
from typing import List

import numpy as np

from app.questionnaire.questionnaire import Questionnaire
from app.reference.ref import Reference
from app.utilities.excel import ExcelSheet


class QuestionnaireStatsGenerator:
    """This class generates statistics from a collection of questionnaires"""

    # the default top level folder name in target saving path for generated results
    _default_top_level_folder = 'report'
    # the default labelled image folder name in top level folder for generated results
    _default_labelled_image_folder_name = 'labelled'
    # the default excel file name
    _default_excel_file_name = 'report.xlsx'

    @staticmethod
    def _get_default_labelled_image_name(questionnaire: Questionnaire) -> str:
        """Gets the default labelled image name

        This function modifies the image file name to the labelled image name
        by adding '_labelled' before its extension
        For example image2.png -> image2_labelled.png

        Parameters
        ----------
        questionnaire:
            a questionnaire object

        Returns
        -------
        labelled_image_name:
            the labelled image name
        """
        return questionnaire.get_stem() + '_labelled' + questionnaire.get_suffix()

    def __init__(self, q_list: List[Questionnaire], target_path: Path):
        """Initializes the generator

        Parameters
        ----------
        q_list:
            a list of questionnaires
        target_path:
            the target image saving path
        """
        self._q_list = q_list
        self._target_path = target_path

    def generate_labelled_images(self):
        """Generates labelled images

        Two groups of elements of an questionnaire image will be labelled:
        --reference points of significance
        --crossed boxes
        """
        for q in self._q_list:
            image_to_save = q.get_labelled_image()
            save_path = self._target_path.joinpath(self._default_top_level_folder,
                                                   self._default_labelled_image_folder_name,
                                                   self._get_default_labelled_image_name(q))
            image_to_save.save(save_path)

    def generate_aggregated_data_excel(self):
        """Generates an excel file with sum of crosses per option per question

        The rows represents questions and the columns represents options
        The number in cells represents the sum of all crosses in questionnaires
        """
        sheet = ExcelSheet.create()
        # create context for rows and columns
        for question in range(1, Reference.default_question_num + 1):
            sheet.set_cell(question + 1, 1, f'Question {question}')
        for option in range(1, Reference.default_option_num + 1):
            sheet.set_cell(1, option + 1, f'Option {option}')
        # initializes a array to compute the sum of crosses per question per option
        aggregated_data = np.zeros((Reference.default_question_num, Reference.default_option_num))
        for q in self._q_list:
            for question in range(1, Reference.default_question_num + 1):
                for option in range(1, Reference.default_option_num + 1):
                    if q.is_crossed(question, option):
                        aggregated_data[question - 1][option - 1] += 1
        # writing the sum to the excel
        for question in range(1, Reference.default_question_num + 1):
            for option in range(1, Reference.default_option_num + 1):
                sheet.set_cell(question + 1, option + 1, aggregated_data[question - 1][option - 1])
        sheet.save(self._target_path.joinpath(self._default_top_level_folder, self._default_excel_file_name))
