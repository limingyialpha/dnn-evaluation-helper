import pickle
import time
from importlib.resources import path
from pathlib import Path

import numpy as np

from app.dnn.data import TrainingBatchDistributor
from app.dnn.data import crossed_vec
from app.utilities.img import Image

# the default package name where the dnn locates
_default_dnn_package_name = 'app.resources.dnn'
# the default resource file name where the dnn locates
_default_dnn_resource_name = 'dnn_50_1.0_400_40_0.9928571428571429'
# the default path to save a trained dnn
_default_dnns_saving_path = ''


class BoxClassifier:
    """The interface of the dnn package"""

    def __init__(self):
        """Initializes the object by automatically loads the trained _BoxClassifierDNN object"""
        with path(_default_dnn_package_name, _default_dnn_resource_name) as p:
            with open(p, 'rb') as dnn_file:
                self._dnn: _BoxClassifierDNN = pickle.load(dnn_file)

    def is_crossed(self, image_to_analyse: Image) -> bool:
        """Returns True if an image is crossed

        The image is by default 40*40 Pixels in gray scale

        Parameters
        ----------
        image_to_analyse:
            the image to analyse

        Returns
        -------
        bool:
            True if an image is crossed
        """
        return self._dnn.is_crossed(image_to_analyse)


class _BoxClassifierDNN:
    """The actual deep neural network for box classification"""

    # number of neurons in the 0 layer
    _l0_neuron_num = 1600
    # number of neurons in the 1 layer
    _l1_neuron_num = 2
    # the default pattern recognition threshold
    # if the difference between an output and the expectation is
    # less than this threshold, they are considered to be the same
    _default_pattern_recognition_threshold = 0.001

    @staticmethod
    def sigmoid(x: float) -> float:
        """The sigmoid function

        the divide between the two cases below prevents an float overflow
        """
        if x < 0:
            return np.exp(x) / (1 + np.exp(x))
        else:
            return 1 / (1 + np.exp(-x))

    def __init__(self):
        """Initialize the dnn, as described in the guide/Neuronale Netze als Evaluationshelfer.pdf"""
        self._batch_distributor = TrainingBatchDistributor()
        self._atv_f = np.vectorize(self.sigmoid)
        self._weight_mtx: np.ndarray = np.random.normal(0, 1, (self._l1_neuron_num, self._l0_neuron_num)) / np.sqrt(
            self._l0_neuron_num)
        self._bias: np.ndarray = np.random.normal(0, 1, (self._l1_neuron_num, 1))

    def feedforward(self, input_array: np.ndarray) -> np.ndarray:
        """the feedforward, as described in the guide/Neuronale Netze als Evaluationshelfer.pdf"""
        return self._atv_f(self._weight_mtx @ input_array + self._bias)

    def train(self, batch_size: int, learning_rate: float, epoch: int):
        """the training process, as described in the guide/Neuronale Netze als Evaluationshelfer.pdf

        training uses only a part of the training data
        """
        for gen in range(0, epoch):
            weight_grad_sum = np.zeros(self._weight_mtx.shape, dtype=float)
            bias_grad_sum = np.zeros(self._bias.shape, dtype=float)
            for data in self._batch_distributor.iter_next_training_batch(batch_size):
                dif_out_exp = self.feedforward(data.input) - data.expectation
                weight_grad_sum = weight_grad_sum + np.multiply(dif_out_exp, data.input.transpose())
                bias_grad_sum = bias_grad_sum + dif_out_exp
            self._bias = self._bias - (learning_rate / batch_size) * bias_grad_sum
            self._weight_mtx = self._weight_mtx - (learning_rate / batch_size) * weight_grad_sum

    def save(self, batch_size_i: int, learning_rate_i: float, epoch_i: int, gen_i: int,
             correctness_rate_i: float):
        """Saves the dnn with parameters"""
        params = ['dnn', str(batch_size_i), str(learning_rate_i), str(epoch_i), str(gen_i), str(correctness_rate_i)]
        file_name = '_'.join(params)
        saving_path = Path(_default_dnns_saving_path).joinpath(file_name)
        with open(saving_path, 'wb') as obj_saving_file:
            pickle.dump(self, obj_saving_file, protocol=pickle.DEFAULT_PROTOCOL)

    def _validate(self, input_data: np.ndarray, exp: np.ndarray) -> bool:
        """Validates the dnn with the rest of the traning data"""
        output_data = self.feedforward(input_data)
        if np.linalg.norm(output_data - exp, 'fro') < self._default_pattern_recognition_threshold:
            return True
        else:
            return False

    def is_crossed(self, input_image: Image) -> bool:
        """Returns true if the input image is crossed"""
        input_data = input_image.to_gray_scale().to_data_ndarray().reshape((self._l0_neuron_num, 1))
        return self._validate(input_data, crossed_vec)

    def get_correctness(self) -> float:
        """Gets the correctness by validating the rest of the training data"""
        correct = 0.0
        image_sum = 0.0
        for data in self._batch_distributor.iter_rest():
            image_sum += 1
            if self._validate(data.input, data.expectation):
                correct += 1
        return correct / image_sum


class DNNSelector:


    """This class natural selects the best _BoxClassifierDNN

    A part of the data is used to train the dnn
    and another part is used to validate the correctness
    This class generates a number of dnns and repeatedly
    selects the one with the highest correctness rate.
    In the end, only the best dnn will be saved using pickle module
    """


def __init__(self, batch_size: int, learning_rate: float, epoch: int, generation_threshold: int,
             correctness_threshold: float):
    """Initializes the selector with params

    Parameters
    ----------
    batch_size:
        the batch size
    learning_rate:
        the learning rate
    epoch:
        the epoch
    generation_threshold:
        the threshold for the generation
        the iteration stops after this generation
    correctness_threshold:
        the threshold for the correctness
        the iteration stops earlier if the
        correctness of the dnn has reached this threshold
    """
    self._batch_size = batch_size
    self._learning_rate = learning_rate
    self._epoch = epoch
    self._generation_threshold = generation_threshold
    self._correctness_threshold = correctness_threshold
    self._current_best_dnn: _BoxClassifierDNN = None
    self._current_best_generation = 0
    self._current_highest_correctness_rate: float = 0.5


def natural_select(self):
    """Natural selects thebest dnn"""
    t_start = time.time()
    print(
        f"The natural selection starts: batch size:{self._batch_size}, learning rate:{self._learning_rate}, epoch:{self._epoch},generation threshold:{self._generation_threshold}, correctness threshold: {self._correctness_threshold}")
    current_generation = 1
    while self._current_highest_correctness_rate <= self._correctness_threshold and current_generation <= self._generation_threshold:
        dnn = _BoxClassifierDNN()
        dnn.train(self._batch_size, self._learning_rate, self._epoch)
        new_correctness_rate = dnn.get_correctness()
        if new_correctness_rate > self._current_highest_correctness_rate:
            self._current_best_dnn = dnn
            self._current_best_generation = current_generation
            self._current_highest_correctness_rate = new_correctness_rate
            message = "New generation"
        else:
            message = "Dead generation"
        print(
            message + f": {current_generation},correctness: {new_correctness_rate:.4f} , time spent:{time.time() - t_start:.2f}")
        current_generation += 1
    print(
        f"Natural Selection ended: the best generation: {self._current_best_generation} with correctness rate:{self._current_highest_correctness_rate}")
    self._current_best_dnn.save(self._batch_size, self._learning_rate, self._epoch, self._current_best_generation,
                                self._current_highest_correctness_rate)
