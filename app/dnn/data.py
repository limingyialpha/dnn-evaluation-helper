from app.utilities.img import *
import numpy as np
crossed_vec = np.array([[0], [1]])
empty_vec = np.array([[1], [0]])

# TODO
# this class provides an interface for the dnn.py in the same package
# to access the training data (crossed boxes and empty ones too)
# please do not save the training data in the project itself, or else
# it will take hours to "commit"
# Public and protected methods, data structures, classes and so on
# must be implemented to achieve the goal above.
# Please pay attention to the randomness of the training data given to the
# dnn.py and have in mind that 75% of the data will be used to train the dnn
# and the rest 25% should be use to validate the correctness of this dnn.
# If you have problems -> Ask Li
class TrainingBatchDistributor:
    def __init__(self, batch_number):
        self.batch_number = batch_number
        self.n = 0
        self.m = 0
        self.gray_value_array = np.array([1600, 1])

    def get_crossed_boxes(self):
        package_crossed_path = "app.resources.crosses.work_type_crossed"  # folder directory
        files = os.listdir(package_crossed_path)  # get all filenames in this path
        image = Image.open_resource(package_crossed_path, files[self.n])  # open box images
        gray_image = image.to_gray_scale()
        self.gray_value_array = gray_image.to_data_ndarray().reshape(1600, 1)
        self.n += 1
        # 75% of dates will be used for training Dnn
        if self.n % self.batch_number == int(0.75*len(files)):
            self.n = 0
        normalised_array = self.gray_value_array / 255
        return normalised_array, crossed_vec

    def get_empty_boxes(self):
        package_empty_path = "app.resources.crosses.work_type_empty"  # folder directory
        files = os.listdir(package_empty_path)  # get all filenames in this path

        image = Image.open_resource(package_empty_path, files[self.m])  # open box images
        gray_image = image.to_gray_scale()
        self.gray_value_array = gray_image.to_data_ndarray().reshape(1600,1)
        self.m += 1
        # 75% of dates will be used for training Dnn
        if self.m % self.batch_number == int(0.75*len(files)):
            self.m = 0
        normalised_array = self.gray_value_array/255

        return normalised_array, empty_vec

