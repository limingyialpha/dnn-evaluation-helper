# Deep Neural Network Evaluation Helper
This project is the final project of "Einführung in Python" Module in SS2020 provided by KIT.

### Project description
"Deep Neural Network Evaluation helper" automatically generates statistics from scanned questionnaires with handwritten crosses.

The formal description can be found in "guide/Neuronale Netze als Evaluationshelfer.pdf".


### User guide
To test the functionality of the project, you should do the followings:
1. Download Git: https://git-scm.com/downloads
2. Open Command Prompt and clone the Repository with: git clone https://github.com/limingyialpha/dnn-evaluation-helper.git
3. Open the project top-level folder with the file explore and unzip the templates/templates.zip
4. Download Anaconda, open the Anaconda Navigator and create a new environment
5. In the newly created environment, install the following packages:

    1. libopencv
    2. numpy
    3. opencv
    4. openpyxl
    5. pillow
    6. py-opencv
    7. tqdm
    
6. After the installation of the pacakges above, activate this encironment and install the CMD.exe Prompt in this environment
7. Open the CMD.exe Prompt and change directory to the top-level directory of the cloned Repo
8. Run the following command: python -m app
9. Follow the instructions in the CMD.exe Prompt and enter the folder path where the unzipped questionnaire images are at for the both questions
10. The generated output are labelled questionnaire images and an excel file with the aggregated data.
