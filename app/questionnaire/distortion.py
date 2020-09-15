from pathlib import Path
from typing import List

import numpy as np

from app.reference.ref import Reference
from app.utilities.pixel import Pixel


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
        pass

    def get_matching_point_list(self) -> List[Pixel]:
        pass

    def get_lin_tran_mtx(self) -> np.ndarray:
        pass

    def get_translation_vec(self) -> np.ndarray:
        pass
