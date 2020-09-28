from pathlib import Path
from typing import List

import numpy as np

from app.reference.ref import Reference
from app.utilities.pixel import Pixel
from app.utilities.img import Image, gray_scale_white, gray_scale_black

# the default radius of a mark around but excluding a middle point
default_mark_radius = 20
# the threshold in the mark edge filtering algorithm
_default_mark_edge_filtering_threshold = 200

class Distortion:
    # TODO
    # each Distortion object belongs to a questionnaire object
    # and holds the following data:
    # 1. matching points to "reference points of significance" as marked in the resource:
    #    app/resources/ref/sig/reference_point_of_significance_labelled.jpg
    # 2. the linear transformation matrix "A" as required in the guide
    # 3. the translation vector "b" as required in the guide

    # implement the following classes so that Distortion will
    # compute the values above automatically in __init__
    # you may freely choose the best data structures and algorithm to solve the
    # problem of finding matching points
    # Attention! the "Aufgabe(Referenzbogen erstellen) (c)" in the guide
    # may possibly incorrect, applying filter to the marks and the to-analyse
    # questionnaire maybe possible, still confused? -> ASK Li
    def __init__(self, path_to_image: Path, ref: Reference):
        """

        Parameters
        ----------
        path_to_image
        ref
        """
        #load Image
        img = Image.open_extern(path_to_image).to_gray_scale()
        p_o_s = p_o_s_finder(ref,img)
        reference_p_o_s = list(ref.iter_ref_p_of_sig())
        A,b = calculate_affine_transformation(p_o_s,reference_p_o_s)
        self._p_o_s = p_o_s
        self._A = A
        self._b = b


    def get_matching_point_list(self) -> List[Pixel]:
        return self._p_o_s

    def get_lin_tran_mtx(self) -> np.ndarray:
        return self._A

    def get_translation_vec(self) -> np.ndarray:
        return self._b


""" This first batch of functions is to derive the matching points
"""


def filter(image: Image,
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

def p_o_s_finder(R: Reference,image: Image) -> List[Pixel]:
    """
    finds a point of significance in an image

    Parameters
    ----------
    R: a Reference object
    image: the image to be compared to the reference image

    Returns
    --------
    the pixels in the image witch correspond to their respective one on the Reference object
    """
    #gets the reference points of significance
    p_o_s = R.iter_ref_p_of_sig()
    # the list witch will be returned
    l =  []
    for p in p_o_s:
        #the reference mask
        mask = R.get_mask(p)
        #placeholder for the lowest value
        diff = 1E10
        #Iterate through a 15x15 block of candidates
        for i in range(-7,8):
            for j in range(-7,8):
                p_test = Pixel(p.x+i,p.y+j)

                image_test = image.crop_around_middle_point(p_test,default_mark_radius)
                image_test = filter(image_test)
                #difference via frobenius
                x = np.linalg.norm(mask - image_test)
                if diff > x:
                    diff = x
                    #save koordinates
                    p_new = p_test
        #save the new pixels
        l.append(p_new)
    return l

""" This batch of functions derive the affirm transformation
"""


def calculate_affine_transformation(x: List[Pixel],y: List[Pixel]) -> List[np.ndarray]:
    """Tries to map x onto y using "Referenzbogen erstellen" d) and gaussian normal equations

    Parameters:
    -----------
    x,y in Pixel^n
    Returns:
    -----------
    A: R^2,2
    b: R^2
    for affine transformation A*x[i] + b ~ y[i] (i<=n)
   """
    C,d  = get_c_d(x,y)
    x = gaussian_normal_eq(C,d)
    A = np.array([[x[0],x[1]],[x[2],x[3]]])
    b = np.array([x[4],x[5]])
    return A,b

def gaussian_normal_eq(A: np.ndarray, b:np.ndarray) -> np.ndarray:
    """calculates the least square solution in the euclidian norm of an linear equation
    Parameters:
    ----------
    A: R^m,n matrix
    b: R^n  vector
    Returns:
    ----------
    x: solution vector R^n
    """
    AT = A.transpose()
    A = np.matmul(AT,A)
    b = np.matmul(AT,b)
    print(A)
    return np.linalg.solve(A,b)

def get_c_d(x:List[Pixel],y:List[Pixel]) ->List[np.ndarray]:
    """

    Parameters
    ----------
    x,y in R^2,n

    Returns
    -------
    matrix C in R^2n,6 and d in R^2*n for the calcuation of the affine transformation: C = (x_1,x_2,  0,  0,  1,  0
                                                                                            0,  0,x_1,x_2,  0,  1)
                                                                                    d  = (y_1
                                                                                          y_2)

    """
    n = len(x)
    C  = np.zeros((2*n,6))
    y_1 = np.zeros(n)
    y_2 = np.zeros(n)
    for i in range(n):
        C [i][0] = x[i].x
        C [i][1] = x[i].y
        C [n+i][2] = x[i].x
        C [n+i][3] = x[i].y
        C [i][4] = 1.0
        C [n+i][5] = 1.0
        y_1[i] = y[i].x
        y_2[i] = y[i].y
    d = np.append(y_1,y_2)
    return C,d



