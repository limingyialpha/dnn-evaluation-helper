# This import fixes the problem that specifying the type of an object
# in its module definition raises error
from __future__ import annotations

from pathlib import Path

from app.dnn.dnn import BoxClassifier
from app.questionnaire.distortion import Distortion
from app.reference.ref import Reference
from app.utilities.img import Image
from app.utilities.pixel import Pixel


# from app.dnn.dnn import default_option_image_size


class Questionnaire:
    """Represents a questionnaire

    A questionnaire has a certain number of questions.
    This number is specified by the Reference.default_question_num.
    An object of this class holds references to:
    -- the path to its image
    -- the reference questionnaire
    -- the deep neural network box classifier
    and uses these to automatically extract data from its image.
    """

    def __init__(self, image_path: Path, ref: Reference, bc: BoxClassifier):
        """ Initialize a questionnaire and automatically extract data from its image

        Parameters
        ----------
        image_path :
            the path to its image
        ref :
            the reference questionnaire
        bc :
            the deep neural network box classifier
        """
        # saves the reference to the image path, reference and box classifier
        self._path = image_path
        self._ref = ref
        self._bc = bc
        # initialize the distortion with the image path and the reference
        self._distortion = Distortion(image_path, ref)
        # initialize the questions and its options
        self._question_num = Reference.default_question_num
        self._option_num = Reference.default_option_num
        self._questions = [_QuestionEntry(self._option_num) for i in range(0, self._question_num)]
        # loads the option images with the help of the distortion
        self._load_option_image()
        # processes the option images with the help of the box classifier
        self._process_option_image()

    def get_question_num(self) -> int:
        """Gets number of questions"""
        return self._question_num

    def get_option_num(self) -> int:
        """Gets number of options per question

        Every question has the same number of options
        """
        return self._option_num

    def get_path(self):
        """Gets the full path of its image file"""
        return self._path

    def get_stem(self) -> str:
        """Gets the image file name without the suffix"""
        return self._path.stem

    def get_suffix(self) -> str:
        """Gets the image file suffix"""
        return self._path.suffix

    def _cross(self, question_num: int, option_num: int):
        """Classify an option into 'crossed'

        Parameters
        ----------
        question_num:
            The question number starting with index 1
        option_num:
            The option number starting with index 1
        """
        self._questions[question_num - 1].cross(option_num)

    def is_crossed(self, question_num: int, option_num: int) -> bool:
        """Returns True if the option is classified into 'crossed'

        Parameters
        ----------
        question_num:
            The question number starting with index 1
        option_num:
            The option number starting with index 1

        Returns
        -------
        True if the option is classified into 'crossed'
        """
        return self._questions[question_num - 1].is_crossed(option_num)

    def _get_ques_opt_image(self, question_num: int, option_num: int) -> Image:
        """Gets the image of an option with given question number and option number

        Parameters
        ----------
        question_num:
            The question number starting with index 1, ends with question number
        option_num:
            The option number starting with index 1, ends with option number

        Returns
        -------
        option_image:
            The image of the option, containing either an empty box or a crossed one.
            The image is resized to match "_default_length_after_resizing"
        """
        return self._questions[question_num - 1].get_option_image(option_num)

    def set_ques_opt_image(self, question_num: int, option_num: int, option_image: Image):
        """Sets the image of an option with given question number and option number

        Parameters
        ----------
        question_num:
            The question number starting with index 1
        option_num:
            The option number starting with index 1
        option_image:
            The image of the option, containing either an empty box or a crossed one.
            The image should be resized to match "_default_length_after_resizing"
        """
        self._questions[question_num - 1].set_option_image(option_num, option_image)

    # TODO
    # use the information from Distortion object to
    # get the corresponding box location pixel that matches the question number
    # and the option number
    def get_box_loc_pixel(self, question_num: int, option_num: int) -> Pixel:
        pass

    # TODO
    # use the information from Distortion object to
    # get the corresponding list of the matching points to
    # "reference points of significance"
    def get_matching_ref_pixels(self):
        pass

    # TODO
    # use the information from Distortion object to
    # solve "Aufgabe (Kaestchen finden)"
    def _load_option_image(self):
        pass

    def _process_option_image(self):
        """Uses the box classifier to classify all the boxes in the questionnaire"""
        for question in range(1, Reference.default_question_num + 1):
            for option in range(1, Reference.default_option_num):
                if self._bc.is_crossed(self._get_ques_opt_image(question, option)):
                    self._cross(question, option)


class _QuestionEntry:
    """Represents a question entry of a questionnaire

    A question entry consists a certain number of options.
    This number is specified by the "Reference.default_option_num"
    """

    def __init__(self, option_num: int):
        """initializes the entry with options

        Parameters
        ----------
        option_num:
            number of options in the question entry
        """
        self._options = [_Option() for i in range(0, option_num)]

    def cross(self, option_num: int):
        """Crosses an option

        Parameters
        ----------
        option_num:
            an option number
        """
        self._options[option_num - 1].cross()

    def is_crossed(self, option_num: int) -> bool:
        """Returns true if an option is crossed

        Parameters
        ----------
        option_num:
            the option number

        Returns
        -------
        is_crossed:
            true if an option is crossed
        """
        return self._options[option_num - 1].is_crossed()

    def get_option_image(self, option_num: int) -> Image:
        """Gets the image of an option

        The size of the square shape option image should be:
        default_option_image_size

        Parameters
        ----------
        option_num:
            the option number

        Returns
        -------
        image:
            the square shape option image
        """
        return self._options[option_num - 1].get_image()

    def set_option_image(self, option_number: int, option_image: Image):
        """Sets the image of an option

        The size of the square shape option image should be:
        default_option_image_size

        Parameters
        ----------
        option_number:
            the option number
        option_image:
            the option image
        """
        self._options[option_number - 1].set_image(option_image)


class _Option:
    """Represents an option of a question in a questionnaire"""

    def __init__(self):
        """Initialize an option

        its image after initialization is None
        the option after initialization is not crossed
        """
        self._image = None
        self._is_crossed = False

    def cross(self):
        """Crosses this option"""
        self._is_crossed = True

    def is_crossed(self) -> bool:
        """Returns true if this option is crossed

        Returns
        -------
        is_crossed:
            true if this option is crossed
        """
        return self._is_crossed

    def get_image(self) -> Image:
        """gets the image of this option

        The size of the square shape option image should be:
        default_option_image_size

        Returns
        -------
        image:
            its option image
        """
        return self._image

    def set_image(self, option_image: Image):
        """Sets the option image

        The size of the square shape option image should be:
        default_option_image_size

        Parameters
        ----------
        option_image:
            the option image
        """
        self._image = option_image
