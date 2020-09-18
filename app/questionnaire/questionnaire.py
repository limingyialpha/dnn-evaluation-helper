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

    # the default box square radius that labels a point
    _default_point_label_box_radius = 20
    # the default box square radius that labels a box
    _default_box_label_box_radius = 30
    # the default box square radius when cropping out option images from the questionnaire image
    _default_cropping_radius = 30
    # the default box square length to resize to from the cropped out option images
    # after that the option image will be saved to this questionnaire object's corresponding option
    _default_length_after_resizing = 40

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
        # initialize the distortion with the image path and the reference and
        # label the image to be labelled with reference points of significance
        self._distortion = Distortion(image_path, ref)
        # initialize the questions and its options
        self._question_num = Reference.default_question_num
        self._option_num = Reference.default_option_num
        self._questions = [_QuestionEntry(self._option_num) for i in range(0, self._question_num)]
        # loads the option images with the help of the distortion
        self._load_option_image()
        # processes the option images with the help of the box classifier
        self._process_option_image()
        # labels the image
        self._labelled_image = self._label_image(self._default_point_label_box_radius,
                                                 self._default_box_label_box_radius)

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

    def get_name(self) -> str:
        """ Gets the image file name with suffix"""
        return self._path.name

    def get_stem(self) -> str:
        """Gets the image file name without the suffix"""
        return self._path.stem

    def get_suffix(self) -> str:
        """Gets the image file suffix"""
        return self._path.suffix

    def get_labelled_image(self) -> Image:
        """Gets the image with matching reference points of significance and crossed boxes labelled"""
        return self._labelled_image

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

    def _get_box_loc_pixel(self, question_num: int, option_num: int) -> Pixel:
        return self._ref.get_box_center_pixel(question_num, option_num).transform(self._distortion.get_lin_tran_mtx(),
                                                                                  self._distortion.get_translation_vec())

    def _load_option_image(self):
        """Loads the option images into the object's options

        The option image's location will first be determined by the distortion
        Then it will then crops out the option image and resize it.
        After that, the option image will be saved into the object's option
        """
        for question in range(1, self.get_question_num() + 1):
            for option in range(1, self.get_option_num() + 1):
                matching_box_center_pixel = self._get_box_loc_pixel(question, option)
                # convert the image into grayscale
                image = Image.open_extern(self._path).to_gray_scale()
                cropped = image.crop_around_middle_point(matching_box_center_pixel, self._default_cropping_radius)
                resized = cropped.resize(self._default_length_after_resizing, self._default_length_after_resizing)
                self.set_ques_opt_image(question, option, resized)

    def _process_option_image(self):
        """Uses the box classifier to classify all the boxes in the questionnaire"""
        for question in range(1, self.get_question_num() + 1):
            for option in range(1, self.get_option_num() + 1):
                if self._bc.is_crossed(self._get_ques_opt_image(question, option)):
                    self._cross(question, option)

    def _label_image(self, point_box_radius: int, box_box_radius: int) -> Image:
        """labels the questionnaire image

        the following two things will be labelled
        --reference points of significance
        --crossed boxes

        Parameters
        ----------
        point_box_radius:
            the box radius to label a point
        box_box_radius:
            the box radius to label a box

        Returns
        -------
        image:
            the labelled image
        """
        to_label = Image.open_extern(self._path)
        half_labelled = self._label_ref_p_of_sig(to_label, point_box_radius)
        return self._label_crossed_box(half_labelled, box_box_radius)

    def _label_ref_p_of_sig(self, to_label: Image, box_radius: int) -> Image:
        """Labels the questionnaire image's reference points of significance

        Parameters
        ----------
        to_label:
            the image to be labelled
        box_radius:
            the box radius that labels a point

        Returns
        -------
        image:
            the labelled image
        """
        matching_p_list = self._distortion.get_matching_point_list()
        labelled = to_label.label_points(matching_p_list, box_radius)
        return labelled

    def _label_crossed_box(self, to_label: Image, box_radius: int) -> Image:
        """Labels the questionnaire image's crossed boxes

        Parameters
        ----------
        to_label:
            the image to be labelled
        box_radius:
        the box radius to label a box

        Returns
        -------
        image:
            the labelled image
        """
        chosen_box_pixels = []
        for question in range(1, self.get_question_num() + 1):
            for option in range(1, self.get_option_num() + 1):
                if self.is_crossed(question, option):
                    matching_chosen_box_pixel = self._get_box_loc_pixel(question, option)
                    chosen_box_pixels.append(matching_chosen_box_pixel)
        labelled = to_label.label_areas(chosen_box_pixels, box_radius)
        return labelled


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
