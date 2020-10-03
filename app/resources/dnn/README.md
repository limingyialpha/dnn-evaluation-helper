#About this directory
This directory is where the trained deep neural network should be placed.

#About the deep neural network
The trained neural network has as an example following file name:

"dnn_50_1.0_400_40_0.9928571428571429"

The file name includes 5 parameters which is used in the traning of this deep neural network:
* 50: The batch size
* 1.0: The learning rate
* 400: The epoch
* 40: The generation when this best deep neural network emerges
* 0.9928571428571429: The correctness in the validation process

The app by default uses around (and maximum) 75% of the training data to train and the rest 25% to validate.
The file name is auto generated with user/developer specified parameters

To train your own deep neural network, you should do the followings:
1. go to app/dnn/data.py and complete the following two attributes to specify the path
of the training data:
    * _default_training_data_crossed_box_folder_name
    * _default_training_data_empty_box_folder_name
2. go to app/dnn/dnn and complete the following one attrbute to specify the path
to save the trained deep neural network:
    * _default_dnns_saving_path
3. go to app/\_\_main\_\_.py
4. comment the main() out and uncomment the rest of the code
6. enter the parameters in DNNSelector to start training. 
Remember that the app uses by default around (and maximum) 75% of data to train and the rest 25% to validate.
So please make sure that batch size multiples epoch is around but less than 75% of the training data size.
7. the command line will output logs while training
8. after the training, the trained dnn will be save to the path specified by "_default_dnns_saving_path"
9. please see if the result is satisfying, if not, discard it and train again.
10. else, save the dnn to app/resources/dnn and replace the attribute "_default_dnn_resource_name"
in app/dnn/dnn with the file name of the newly trained deep neural network
11. now uncomment the main() and comment the rest of the code out,
 the application is now using the newly trained deep neural network