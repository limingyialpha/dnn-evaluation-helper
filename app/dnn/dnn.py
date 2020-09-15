from app.utilities.img import Image

# Here is  where you place your best-performing
# dnn, in app\resources\dnn\dnn <- the last is simply the file name
# this file "dnn" is an object of type _BoxClassifierDNN "pickled"
# by the library "pickle"
default_dnn_saving_package_name = 'app.resources.dnn'
default_dnn_saving_resource_name = 'dnn'
# the default option image size in the traning data
default_option_image_size = 40


# TODO
# This class is the (and the only) interface of the app.dnn package
# This class will be called by the "questionnaire class" to determine
# whether a box is crossed or empty
class BoxClassifier:
    # initialize the classifier by loading the saved best-performing
    # with library "pickle"
    def __init__(self):
        pass

    # TODO
    # the only public methods that will be called from outside the package
    # this method uses the loaded dnn to get the output array like
    # [[0.67][0.99]] and defines an algorithm to put it either in category
    # crossed or empty. if its crossed, return True.
    def is_crossed(self, image_to_analyse: Image) -> bool:
        pass


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
    pass


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
        pass

    # TODO
    # use the "pickle" library to save the object
    # (When the DNNSelector finds its correctness is high)
    def save_object(self):
        pass
