# This import fixes the problem that specifying the type of an object
# in its module definition raises error
from __future__ import annotations

import numpy as np


class Pixel:
    """Represents a pixel in a 2D image

    This class encapsulates all the data structures and logics
    of a pixel in a 2D image. Whenever a pixel is referred
    to in this project, it is encouraged to use this class to
    represent it.
    """

    def __init__(self, x: int, y: int):
        """Initialize the pixel with coordinates

        Parameters
        ----------
        x:
            The horizontal x coordinate
        y:
            The vertical y coordinate
        """
        self._x = x
        self._y = y

    @property
    def x(self) -> int:
        """Gets the horizontal x coordinate

        Returns
        -------
        x:
            the horizontal x coordinate
        """
        return self._x

    @property
    def y(self) -> int:
        """Gets the vertical y coordinate

        Returns
        -------
        y:
            the vertical y coordinate
        """
        return self._y

    @x.setter
    def x(self, value: int):
        """Sets the horizontal x coordinate

        Parameters
        ----------
        value:
            the horizontal x coordinate
        """
        if value < 0:
            raise ValueError("x coordinate value can not be less than zero!")
        self._x = value

    @y.setter
    def y(self, value: int):
        """Sets the vertical y coordinate

                Parameters
                ----------
                value:
                    the vertical y coordinate
                """
        if value < 0:
            raise ValueError("y coordinate value can not be less than zero!")
        self._y = value

    def transform(self, lin_tran_mtx: np.ndarray, translation_vec: np.ndarray) -> Pixel:
        """Transforms the pixel to another pixel

        The transformation is parameterized by:
        -- a 2*2 linear transformation matrix A
        -- a 2*1 translation vector b
        The transformation is characterized by the following formula:
        pixel_new = A @ pixel_old + b where @ stands for matrix multiplication

        Parameters
        ----------
        lin_tran_mtx:
            a 2*2 linear transformation matrix
        translation_vec:
            a 2*1 translation vector b

        Returns
        -------
        pixel_new:
            a new pixel transformed from the old one
        """
        loc_vec = np.array([[self._x], [self._y]])
        loc_vec_after = lin_tran_mtx @ loc_vec + translation_vec
        return Pixel(round(loc_vec_after[0][0]), round(loc_vec_after[1][0]))

    def __str__(self):
        """Returns the string representation of the pixel

        Example: Pixel 3:2
        """
        return "Pixel {}:{}".format(self._x, self._y)
