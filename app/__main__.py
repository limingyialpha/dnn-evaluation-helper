from concurrent import futures
from pathlib import Path

from tqdm import tqdm

from app.stats.stats import QuestionnaireStatsGenerator
from app.task import generate_questionnaire
from app.dnn.dnn import DNNSelector

# the accepted image formats
accepted_image_format_extensions = [".jpg", ".jpeg", ".png"]


class ImageNotFound(Exception):
    """Thrown when no image is found or the found image format is not accepted"""
    pass


def main():
    """The main logic of the application"""
    source_path = Path(input('Please enter the directory path where the questionnaire images will be analysed:'))
    print("Scanning directory for images...")
    # initialize an list to hold all paths to images
    q_path_list = [path for path in source_path.iterdir() if path.suffix in accepted_image_format_extensions]
    q_num = len(q_path_list)
    # raise exception if no image is found or the found image format is not accepted
    if q_num == 0:
        raise ImageNotFound(
            f"The current directory contains no images or the image format is not accepted.\n Accepted image formats: {accepted_image_format_extensions}")
    target_path = Path(input('Please enter the directory path for the generated files:'))
    # print initial empty progress bar
    print("Analysing questionnaire images...(this might take a while)")
    print(
        "Multiple progress bars will probably be generated due to collisions between tqdm package and concurrent package")
    # initialize multiple processes to analyse images
    with futures.ProcessPoolExecutor() as ex:
        q_list = list(tqdm(ex.map(generate_questionnaire, q_path_list), total=q_num,
                           bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'))
    # generate statistics
    qsa = QuestionnaireStatsGenerator(q_list, target_path)
    print("Generating output files...")
    qsa.generate_labelled_images()
    qsa.generate_aggregated_data_excel()
    print(f"Process ended. Output files can be found in {target_path}")


if __name__ == '__main__':
    main()

    """The code below is used to train a DNN, Please only train dnn here, or else pickle package will not work
    selector = DNNSelector(batch_size=50, learning_rate=1.0, epoch=400, generation_threshold=1,
                           correctness_threshold=0.999)
    selector.natural_select()
    """
