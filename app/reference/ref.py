"""This module encapsulates the logic of the reference questionnaire

It consists of 3 class with 2 protected and 1 public as
the interface of the "reference" package:
2 protected classes:
    class "_ReferencePointsOfSignificanceLoader":
            responsible for loading the resource related to the reference
            points of significance and auto generating the masks.
    class "_ReferenceBoxLoader":
            responsible for loading the resource related to the reference boxes
1 public class:
    class "Reference"
            this class is the only interface of the "reference" package
            it wraps up the 2 protected loader classes and hides their implementations
            behind the interface to regulate and simplify the access to reference
            questionnaire related data.
"""
from typing import Iterator

import numpy as np

from app.utilities.excel import ExcelSheet
from app.utilities.img import Image, gray_scale_white, gray_scale_black
from app.utilities.pixel import Pixel

# the default radius of a mark around but excluding a middle point
default_mark_radius = 20
# the threshold in the mark edge filtering algorithm
_default_mark_edge_filtering_threshold = 200


class Reference:
    """Represents the whole reference questionnaire

    The only interface of the "reference package"
    It wraps up the 2 protected loader classes and hides their implementations
    behind the interface to regulate and simplify the access to reference
    questionnaire related data.
    """
    # the default number of questions in a questionnaire
    default_question_num = 14
    # the default number of options in each question
    default_option_num = 5
    # the default package name of the reference questionnaire image
    _default_ref_image_package_name = 'app.resources.ref'
    # the default resource file name of the reference questionnaire image
    _default_ref_image_resource_name = 'reference.jpg'

    def __init__(self, package: str = _default_ref_image_package_name,
                 resource: str = _default_ref_image_resource_name):
        """Initialize a reference questionnaire

        Parameters
        ----------
        package:
            the package name of the reference questionnaire image
            (default is _default_ref_image_package_name)
        resource:
            the resource file name of the reference questionnaire image
            (default is _default_ref_image_resource_name)
        """

        # opens the reference questionnaire image and convert it into gray scale
        ref_image = Image.open_resource(package, resource).to_gray_scale()
        # initializes the both loader
        self._ref_p_of_sig_loader = _ReferencePointsOfSignificanceLoader(ref_image)
        self._ref_box_loader = _ReferenceBoxLoader()

    def iter_ref_p_of_sig(self) -> Iterator[Pixel]:
        """Returns the iterator that iterates through the reference points of significance"

        Returns
        -------
        iterator:
            iterates through the reference points of significance
        """
        return self._ref_p_of_sig_loader.iter_pixel()

    def get_mask(self, ref_p_of_sig: Pixel) -> np.ndarray:
        """Gets the auto-generated mask from a reference point of significance

        the mask will be generated as a square shape image-converted
        numpy ndarray with the given reference point of significance
        in the middle from the reference questionnaire image after
        edge filtering algorithm
        If the given Pixel is not a reference point of significance,
        an Exception will be raised

        Parameters
        ----------
        ref_p_of_sig:
            a reference point of significance

        Returns
        -------
        numpy ndarray:
            a square shape image-converted numpy ndarray
        """
        return self._ref_p_of_sig_loader.get_mask(ref_p_of_sig)

    def get_box_center_pixel(self, question_num: int, option_num: int) -> Pixel:
        """Gets the pixel that is in the middle of an option box

        Parameters
        ----------
        question_num:
            the question number of the box
        option_num
            the option number of the box
        Returns
        -------
        pixel:
            the pixel that is in the middle of the option box
        """
        return self._ref_box_loader.get_box_center_pixel(question_num, option_num)


class _ReferencePointsOfSignificanceLoader:
    """Loads the reference point of significance related resource and auto generates their masks"""

    # the default package name of the reference points of significance excel
    _default_ref_p_of_sig_excel_package_name = 'app.resources.ref.sig'
    # the default resource file name of the reference points of significance excel
    _default_ref_p_of_sig_excel_resource_name = 'ref_pos_sig.xlsx'

    def __init__(self, ref_image: Image, package: str = _default_ref_p_of_sig_excel_package_name,
                 resource: str = _default_ref_p_of_sig_excel_resource_name):
        """Initializes the loader

        Resources related to reference points of significance will be loaded
        Masks will be auto generated

        Parameters
        ----------
        ref_image:
            The reference questionnaire image
        package: optional
            The package name of the reference points of significance excel
            (default is _default_ref_p_of_sig_excel_package_name)
        resource:
            The resource file name of the reference points of significance excel
            (default is _default_ref_p_of_sig_excel_resource_name)

        """
        # open the excel sheet
        sheet = ExcelSheet.open_resource(package, resource)
        self._pixel_mask_dict = {}
        # iterate through the x, y coordinates values of each reference points of significance
        for row_items in sheet.iter_row_values():
            p = Pixel(row_items[0], row_items[1])
            # crop the image with default_mark_radius
            raw_mask_image = ref_image.crop_around_middle_point(p, default_mark_radius)
            # mask will be generated after applying the edge filtering algorithm
            mask = self._filter_out_edge(raw_mask_image)
            # link the reference point of significance to its mask
            self._pixel_mask_dict[p] = mask

    def iter_pixel(self) -> Iterator[Pixel]:
        """Iterates through the reference points of significance

        Returns
        -------
        iterator:
            iterates through the reference points of significance
        """
        return self._pixel_mask_dict.__iter__()

    def get_mask(self, ref_p_of_sig: Pixel) -> np.ndarray:
        """Gets the auto-generated mask from a reference point of significance

        Parameters
        ----------
        ref_p_of_sig:
            a reference point of significance

        Returns
        -------
        numpy ndarray:
            a square shape image-converted numpy ndarray
        """
        return self._pixel_mask_dict[ref_p_of_sig]

    @staticmethod
    def _filter_out_edge(image: Image,
                         edge_filtering_threshold: int = _default_mark_edge_filtering_threshold) -> np.ndarray:
        """Applying the edge filtering algorithm

        The algorithm scans the grayscale value of each pixel,
        if the value is smaller than the threshold(darker than the threshold color),
        the grayscale value will be set to absolute black: gray_scale_black
        else the grayscale value will be set to absolute white: gray_scale_white
        As described above, the algorithm filters out the edge of a grayscale image


        Parameters
        ----------
        image:
            the input image
        edge_filtering_threshold:
            the threshold used in edge filtering algorithm

        Returns
        -------
        image(mask):
            a new filtered image, supposed to be used as mask
        """
        image_data = image.to_data_ndarray()
        for i in range(0, len(image_data)):
            for j in range(0, len(image_data[0])):
                if image_data[i][j] < edge_filtering_threshold:
                    image_data[i][j] = gray_scale_black
                else:
                    image_data[i][j] = gray_scale_white
        return image_data


class _ReferenceBoxLoader:
    """Loads the reference box related resource"""

    # the default package name of the reference box excel
    _default_ref_box_excel_package_name = 'app.resources.ref.box'
    # the default resource file name of the reference box excel
    _default_ref_box_excel_resource_name = 'ref_box.xlsx'

    def __init__(self, package: str = _default_ref_box_excel_package_name,
                 resource: str = _default_ref_box_excel_resource_name):
        """Initializes the loader

        Resource related to the reference boxes will be loaded

        Parameters
        ----------
        package:
            The package name of the reference box excel
            (default is _default_ref_box_excel_package_name)
        resource
            The resource file name of the reference box excel
            (default is _default_ref_box_excel_resource_name)
         """
        sheet = ExcelSheet.open_resource(package, resource)
        self._box_middle_pixel_array = []
        for row_items in sheet.iter_row_values():
            self._box_middle_pixel_array.append(
                [Pixel(row_items[2 * i], row_items[2 * i + 1]) for i in range(0, Reference.default_option_num)])

    def get_box_center_pixel(self, question_num: int, option_num: int) -> Pixel:
        """Gets the pixel that is in the middle of an option box

        Parameters
        ----------
        question_num:
            the question number of the box
        option_num
            the option number of the box
        Returns
        -------
        pixel:
            the pixel that is in the middle of the option box
        """
        return self._box_middle_pixel_array[question_num - 1][option_num - 1]
