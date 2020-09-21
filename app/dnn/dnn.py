import random
import numpy as np
import pickle
from app.utilities.img import Image
from app.dnn.data import TrainingBatchDistributor

# Here is  where you place your best-performing
# dnn, in app\resources\dnn\dnn <- the last is simply the file name
# this file "dnn" is an object of type _BoxClassifierDNN "pickled"
# by the library "pickle"
default_dnn_saving_package_name = 'app.resources.dnn'
default_dnn_saving_resource_name = 'dnn'
# the default option image size in the training data
default_option_image_size = 40
default_TrainingBatch_number = 100
train_batch = TrainingBatchDistributor(default_TrainingBatch_number)

def sigmoid(x):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

# TODO
# This class is the (and the only) interface of the app.dnn package
# This class will be called by the "questionnaire class" to determine
# whether a box is crossed or empty
class BoxClassifier:
    # initialize the classifier by loading the saved best-performing
    # with library "pickle"
    def __init__(self):
        with open("dnn_object.pkl", 'rb') as obj_loading_file:
            self.trained_dnn = pickle.load(obj_loading_file)


    # TODO
    # the only public methods that will be called from outside the package
    # this method uses the loaded dnn to get the output array like
    # [[0.67][0.99]] and defines an algorithm to put it either in category
    # crossed or empty. if its crossed, return True.
    def is_crossed(self, image_to_analyse: Image) -> bool:
        gray_image = image_to_analyse.to_gray_scale()
        input_array = gray_image.to_data_ndarray().reshape(1600, 1)
        a_l1 = self.trained_dnn.feedforward(input_array)

        error = a_l1 - np.array([[0], [1]])

        # after calculation ,if the square error is less than 0.02, it will be recognised as crossed box.
        if (error[1, 1]**2 + error[2, 1]**2) <= 0.02:
            return True
        else:
            return False


# TODO
# The mother nature is cruel to its children just like this class to
# _BoxClassifierDNN. Failing dnns will be abandoned and successful ones
# will be saved(by calling its save() member method).(but save to where?)
# You will probably initialize and validate 100 dnns one by one and select the
# one with the best correctness rate in validation process.(and of course save it)
# Please do not save all the selected dnns in the project, the repository will be
# polluted, please save it in your local computer and manually choose the best-performing
# one in the project
# still confused -> ask Li
class DNNSelector:

    def __init__(self):
        self.box_classifier_dnn = _BoxClassifierDNN()

    def dnn_select(self, times):
        best_dnn = 10
        for i in times:
            random_number = random.randint(0, 1)
            if random_number == 0:
                input_array, y = train_batch.get_crossed_boxes()
            if random_number == 1:
                input_array, y = train_batch.get_empty_boxes()
            self.box_classifier_dnn.train()
            output_a_l1 = self.box_classifier_dnn.feedforward(input_array)
            cost = self.cross_entropy_cost(output_a_l1, y)
            if cost < best_dnn:
                best_dnn = cost
                self.box_classifier_dnn.save_object()

    def cross_entropy_cost(self, a_l1, y):
        return -(np.dot(y.T, np.log(a_l1)) + np.dot((1-y).T, np.log(1-a_l1)))

# TODO
# this class implements the real dnn entity.
# this class defines the functions and implements the neural network algorithm
# this class should be able to train the dnn to learn the cross/empty pattern
# this class however does not care about the successfulness of the trained
# neural network
# other methods should be implemented
# think about where should I put or get the parameters like batch_size, learning rate,
# and epochs.......hummmmm
class _BoxClassifierDNN:
    _l0_neuron_num = 1600
    _l1_neuron_num = 2

    def __init__(self):
        # weights
        self.weight_array = np.random.rand(2,1600)
        # biases
        self.biases_array = np.random.rand(2,1)

    # dates is a normalised gray-value array(1600,1)
    def feedforward(self, input_array):

        output = sigmoid(np.dot(self.weight_array, input_array) + self.biases_array)

        return output


    def train(self):
        learn_rate = 0.1
        epochs = 10
        sum_dC_dw = 0
        sum_dC_db = 0


        # set a random number (0,1) to casual choose the box from either crossed boxes or empty boxes
        for epoch in range(epochs):
            random_number = random.randint(0,1)
            if random_number == 0:
                input_array, y = train_batch.get_crossed_boxes()
            if random_number == 1:
                input_array, y = train_batch.get_empty_boxes()

            # - - - Do a feedforward
            for i in range(train_batch.batch_number):
                z_l1 = np.dot(self.weight_array, input_array) + self.biases_array
                a_l1 = sigmoid(z_l1)         # a_l1 is a matrix (2,1)
                dC_dz = a_l1 - y
                dz_dw = input_array
                dz_db = 1

                dC_dw = dC_dz * dz_dw
                dC_db = dC_dz * dz_db

                sum_dC_dw += dC_dw
                sum_dC_db += dC_db

            self.weight_array -= (learn_rate/train_batch.batch_number) * sum_dC_dw
            self.biases_array -= (learn_rate/train_batch.batch_number) * sum_dC_db

        #return self.weight_array, self.biases_array



    # TODO
    # use the "pickle" library to save the object
    # (When the DNNSelector finds its correctness is high)
    def save_object(self):
        with open("dnn_object", 'wb') as obj_saving_file:
            pickle.dump(self, obj_saving_file, protocol=pickle.HIGHEST_PROTOCOL)



