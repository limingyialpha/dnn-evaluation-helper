import sys
from itertools import product
from pathlib import Path
from typing import Iterator, Dict, Tuple, List

import numpy as np
from numpy.linalg import norm

from app.reference.ref import Reference
from app.reference.ref import default_mark_radius
from app.utilities.img import Image
from app.utilities.img import gray_scale_white, gray_scale_black
from app.utilities.pixel import Pixel

# the default search radius of the square region around a reference point of significance
# a matching point is expected to be found in this square region
_default_search_radius = 25

# the default threshold when filtering the edge
# see method "_filter_out_edge"
_default_data_edge_filtering_threshold = 200


class Distortion:
    """The distortion class uses the reference questionnaire to compute the distortion of a given questionnaire

    The distortion is an affine transformation, which includes two parameters:
    --A: a 2*2 linear transformation matrix
    --b: a 2*1 translation vector
    The distortion of a questionnaire to the reference questionnaire can be computed by:
    pixel = A @ pixel_ref + b (computation is pixelwise)
    """

    def __init__(self, path_to_image: Path, ref: Reference):
        """Initializes the distortion and compute the parameters

        Parameters
        ----------
        path_to_image:
            the path to the given questionnaire image
        ref:
            the reference object
        """
        # open and convert the image into gray scale
        image = Image.open_extern(path_to_image).to_gray_scale()
        # filter out the edge of the interested regions of image to data in ndarray
        filtered_image_data = self._filter_out_edge(image, ref.iter_ref_p_of_sig(),
                                                    _default_data_edge_filtering_threshold,
                                                    _default_search_radius + default_mark_radius)
        # using a dictionary to hold the reference points of significance and there matching points
        # reason of using a dictionary: preserving the semantic of "matching"
        self._ref_and_matching_p_of_sig_dict: Dict[Pixel, Pixel] = {}
        # finding matching points
        for pixel in ref.iter_ref_p_of_sig():
            matching_p = self._find_matching_point_of_sig(filtered_image_data, pixel, ref)
            self._ref_and_matching_p_of_sig_dict[pixel] = matching_p
        # compute the distortion parameters with matching points
        self._lin_tran_mtx, self._translation_vec = self._get_distortion_params(self._ref_and_matching_p_of_sig_dict)

    def get_matching_point_list(self) -> List[Pixel]:
        """Gets the list of the matching points

        Returns
        -------
        list:
            the list of the matching points
        """
        return list(self._ref_and_matching_p_of_sig_dict.values())

    def get_lin_tran_mtx(self) -> np.ndarray:
        """Get's the 2*2 linear transformation matrix A

        Returns
        -------
        A:
            the 2*2 linear transformation matrix
        """
        return self._lin_tran_mtx

    def get_translation_vec(self) -> np.ndarray:
        """Get's the 2*1 translation vector b

        Returns
        -------
        b:
            the 2*1 translation vector
        """
        return self._translation_vec

    @staticmethod
    def _get_distortion_params(ref_and_matching_p_of_sig_dict: Dict[Pixel, Pixel]) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the distortion parameters

        the algorithm is described in the guide
        parameters include:
        --A: a 2*2 linear transformation matrix
        --b: a 2*1 translation vector
        """
        num_of_points = len(ref_and_matching_p_of_sig_dict)
        c_matrix = np.zeros((2 * num_of_points, 6))
        d_vector = np.zeros((2 * num_of_points, 1))
        current_index = 0
        for key in ref_and_matching_p_of_sig_dict:
            c_matrix[current_index][0] = key.x
            c_matrix[current_index][1] = key.y
            c_matrix[current_index][4] = 1
            c_matrix[current_index + 1][2] = key.x
            c_matrix[current_index + 1][3] = key.y
            c_matrix[current_index + 1][5] = 1
            d_vector[current_index][0] = ref_and_matching_p_of_sig_dict[key].x
            d_vector[current_index + 1][0] = ref_and_matching_p_of_sig_dict[key].y
            current_index += 2
        best_fit = np.linalg.lstsq(c_matrix, d_vector, rcond=None)[0]
        linear_transformation_matrix = np.array([[best_fit[0][0], best_fit[1][0]], [best_fit[2][0], best_fit[3][0]]])
        translation_vector = np.array([[best_fit[4][0]], [best_fit[5][0]]])
        return linear_transformation_matrix, translation_vector

    @staticmethod
    def _find_matching_point_of_sig(filtered_image_data: np.ndarray, ref_point: Pixel, ref: Reference) -> Pixel:
        """Finds the matching point of a reference point of significance

        the algorithm is described in the guide

        Parameters
        ----------
        filtered_image_data:
            the filtered image data in numpy ndarray
        ref_point:
            the reference point of significance
        ref:
            the reference questionnaire

        Returns
        -------
        the matching point:
            the matching point in the given questionnaire image to the reference point of significance
        """
        # initialize a container to record the current minimum value and the matching point
        current_min_dif_quad_sum = sys.maxsize
        current_best_matching_point = None
        # Careful! the first index of the data matrix is y coordinate and the second is x coordinate!
        for y, x in Distortion._iter_search_area(ref_point, _default_search_radius):
            filtered_sub_data = Distortion._get_sub_array_around_middle_point(filtered_image_data, y, x,
                                                                              default_mark_radius)
            dif_quad_sum = Distortion._get_dif_fro_norm(filtered_sub_data, ref.get_mask(ref_point))
            if dif_quad_sum < current_min_dif_quad_sum:
                current_min_dif_quad_sum = dif_quad_sum
                current_best_matching_point = Pixel(x, y)
        return current_best_matching_point

    @staticmethod
    def _iter_search_area(middle_point: Pixel, search_radius: int) -> Iterator:
        """Iterates through the pixel indexes in the searching area

        Parameters
        ----------
        middle_point:
            the middle point of the searching area
        search_radius:
            the square radius of the searching area

        Returns
        -------
        Iterator:
            iterates through the pixel indexes in the searching area
        """
        x_mid = middle_point.x
        y_mid = middle_point.y
        x_search_range = range(x_mid - search_radius, x_mid + search_radius + 1)
        y_search_range = range(y_mid - search_radius, y_mid + search_radius + 1)
        return product(y_search_range, x_search_range)

    @staticmethod
    def _get_sub_array_around_middle_point(array: np.ndarray, middle_index_y: int, middle_index_x: int,
                                           radius: int) -> np.ndarray:
        """Gets the sub square array around a middle pixel
        
        This method is used to get the numpy ndarray of the interested
        region of a questionnaire image
        
        Parameters
        ----------
        array:
            the original array
        middle_index_y:
            the index of the middle point in y dimension
        middle_index_x:
            the index of the middle point in x dimension
        radius:
            the square radius

        Returns
        -------
        sub array:
            a sub array around the middle point with the given square radius
        """
        return array[middle_index_y - radius:middle_index_y + radius + 1,
               middle_index_x - radius:middle_index_x + radius + 1]

    @staticmethod
    def _get_dif_fro_norm(array_1: np.ndarray, array_2: np.ndarray) -> float:
        """Gets the Frobenius norm of the pixelwise difference of two numpy ndarrays

        Parameters
        ----------
        array_1:
            Minuend numpy ndarray
        array_2
            Subtrahend numpy ndarray
        Returns
        -------
        float
            the Frobenius norm of the pixelwise difference of two numpy ndarrays
        """
        return norm(array_1 - array_2, ord='fro')

    @staticmethod
    def _filter_out_edge(image: Image, pixels: Iterator[Pixel],
                         edge_filtering_threshold: int, radius: int) -> np.ndarray:
        """Applying the edge filtering algorithm

        The algorithm scans the grayscale value of each pixel,
        if the value is smaller than the threshold(darker than the threshold color),
        the grayscale value will be set to absolute black: gray_scale_black
        else the grayscale value will be set to absolute white: gray_scale_white
        As described above, the algorithm filters out the edge of a grayscale image

        Parameters
        ----------
        image:
            the image to filter
        pixels:
            the pixels around which the regions should be filtered
        edge_filtering_threshold:
            the edge filtering threshold (default is _default_data_edge_filtering_threshold = 200)
        radius:
            the radius of the square around the pixels
            only the pixels in these square regions will be filtered

        Returns
        -------
        filtered image:
            the filtered image
        """
        image_data = image.to_data_ndarray()
        for pixel in pixels:
            i_range = range(pixel.y - radius, pixel.y + radius + 1)
            j_range = range(pixel.x - radius, pixel.x + radius + 1)
            for i in i_range:
                for j in j_range:
                    if image_data[i][j] < edge_filtering_threshold:
                        image_data[i][j] = gray_scale_black
                    else:
                        image_data[i][j] = gray_scale_white
        return image_data
