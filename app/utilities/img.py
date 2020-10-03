# This import fixes the problem that specifying the type of an object
# in its module definition raises error
from __future__ import annotations

import os
from importlib.resources import path
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image as ImageLib
from PIL import ImageDraw

from app.utilities.pixel import Pixel

# represents color black in grayscale specified with 1 Byte
gray_scale_black = 0
# represents color white in grayscale specified with 1 Byte
gray_scale_white = 255
# represents color red in RGB color mode specified with 1 Byte per color channel
_rgb_red = (255, 0, 0)
# the default width of a labelling box
_default_box_line_width = 2
# the default radius of a labelling dot
_default_dot_radius = 1


class Image:
    """Represents a 2D image

    This class encapsulates all the attributes and behaviours of
    a 2D image.
    It hides, regulates and standardizes the access to other image
    processing libraries, thus protecting developers from making
    mistakes and simplifies the image processing in the project.
    It is encouraged to use this class to represent all the images
    in the project.
    """

    def __init__(self, content: ImageLib.Image):
        """Initialize an object with its content, an opened image from an image library

        It is not encouraged to use the constructor directly to construct new objects.
        Instead, use the following factory methods:
        -- open_resource:
            opens an image from the project's resources
        -- open_extern
            opens an image outside the project's scope

        Parameters
        ----------
        content:
            an opened image from an image library
        """
        self._content = content

    @classmethod
    def open_resource(cls, package: str, resource: str) -> Image:
        """Opens an image from the project's resources

        Factory method to initialize an object

        Parameters
        ----------
        package:
            package name, starting from the top level package to the package
            containing the resource, using . as separator
            For example: 'app.utilities'
        resource
            resource file name with extension
            For example: 'i_hate_programming.txt'

        Returns
        -------
        image:
            an Image object with the opened image
        """
        with path(package, resource) as p:
            return cls(ImageLib.open(os.fspath(p)))

    @classmethod
    def open_extern(cls, path_to_img: Path) -> Image:
        """Opens an image outside the project's scope

        Factory method to initialize an object

        Parameters
        ----------
        path_to_img:
            the path object specifying the location of the image

        Returns
        -------
            an Image object with the opened image

        """
        return cls(ImageLib.open(os.fspath(path_to_img)))

    def save(self, save_path: Path, img_format='JPEG'):
        """Saves the image to the given path

        if the given path does not exist,
        this path will be automatically generated

        Parameters
        ----------
        save_path:
            the path object specifying the saving location
        img_format: optional
            the format of the image (default is JPEG)
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self._content.save(os.fspath(save_path), format=img_format)

    def show(self):
        """Shows the image by display it with a default 3rd party image gallery app

        Mainly for debugging purposes
        """
        self._content.show()

    def resize(self, width: int, height: int) -> Image:
        """Resizes the image

        Parameters
        ----------
        width:
            the horizontal width in pixels after resizing
        height:
            the vertical height in pixels after resizing

        Returns
        -------
        image:
            a new resized image
        """
        return Image(self._content.resize((width, height)))

    def to_data_ndarray(self) -> np.ndarray:
        """Returns a numpy ndarray object corresponding to the image

        Attention!
        the height of the image corresponds to the first index(row number) of the array
        while the width corresponds to the second index(column number)!
        For example:
        After conversion, to access the image pixel data with horizontal x coordinate 3
        and vertical y coordinate 11, array[11][3] should be called
        """
        return np.array(self._content)

    def to_rgb(self) -> Image:
        """Converts the image to RGB color mode

        Each pixel contains 3 Byte data specifying the Red, Green and
        Blue channel

        Returns
        -------
        image:
            a new image after conversion
        """
        return Image(self._content.convert(mode='RGB'))

    def to_gray_scale(self) -> Image:
        """Converts the image to grayscale color mode

        Each pixel contains only 1 Byte data specifying the gray scale channel

        Returns
        -------
        image:
            a new image after conversion

        """
        return Image(self._content.convert(mode='L'))

    def crop_around_middle_point(self, center_pixel: Pixel, radius: int) -> Image:
        """Crops the image into a square shape around a middle point(pixel)

        Parameters
        ----------
        center_pixel:
            the center pixel around which it should be cropped
        radius:
            the radius of the square after cropping, excluding the
            center pixel itself. For example, if radius is 10,
            then the width of the square will be 10+1+10 = 21

        Returns
        -------
        image:
            a new cropped image
        """
        x = center_pixel.x
        y = center_pixel.y
        return Image(self._content.crop((x - radius, y - radius, x + radius + 1, y + radius + 1)))

    def label_areas(self, center_pixels: List[Pixel], radius: int) -> Image:
        """Labels areas of images around middle points

        each label is a red squares bounding the area around the middle points
        the line width of the labelling square is given by the:
        _default_box_line_width

        Parameters
        ----------
        center_pixels:
            a list of center pixels around whose area it should be labelled
        radius:
            the radius of each square, excluding the
            center pixel itself. For example, if radius is 10,
            then the width of the square will be 10+1+10 = 21

        Returns
        -------
        image:
            a new labelled image
        """
        self._content = self._content.convert(mode='RGB')
        draw = ImageDraw.Draw(self._content)
        for middle_point in center_pixels:
            x = middle_point.x
            y = middle_point.y
            draw.rectangle([x - radius, y - radius, x + radius, y + radius], fill=None, outline=_rgb_red,
                           width=_default_box_line_width)
        return Image(self._content)

    def label_points(self, points: List[Pixel], radius: int) -> Image:
        """Labels points of images

        each label consists of 2 parts:
        -- a red dot right on the specified point itself
            the dot radius, excluding the specified point itself, is given by the:
            _default_dot_radius
        -- a red bounding square with the specified point in the middle
            the line width of the labelling square is given by the:
            _default_box_line_width


        Parameters
        ----------
        points:
            a list of points(pixels) to be labelled
        radius:
            the radius of each square, excluding the
            center pixel itself. For example, if radius is 10,
            then the width of the square will be 10+1+10 = 21

        Returns
        -------
        image:
            a new labelled image
        """
        self._content = self._content.convert(mode='RGB')
        draw = ImageDraw.Draw(self._content)
        for middle_point in points:
            x = middle_point.x
            y = middle_point.y
            for j in range(y - _default_dot_radius, y + _default_dot_radius + 1):
                for i in range(x - _default_dot_radius, x + _default_dot_radius + 1):
                    draw.point((i, j), fill=_rgb_red)
            draw.rectangle([x - radius, y - radius, x + radius, y + radius], fill=None, outline=_rgb_red,
                           width=_default_box_line_width)
        return Image(self._content)
