"""A workaround to stop concurrent.futures raising BrokenProcessPool Exception

There is an unsolved problem in multiprocessing.Pool
See: https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror
The concurrent.futures.ProcessPoolExecutor which internally uses multiprocessing.Pool
are raising BrokenProcessPool Exception due to the unreasonable AttributeError
from multiprocessing.Pool
The workaround is having the task submitted to the process pool executor
separated into another file"""
from pathlib import Path

from app.questionnaire.questionnaire import Questionnaire


def generate_questionnaire(path: Path) -> Questionnaire:
    """The task that should be submitted to the ProcessPoolExecutor in __main__.py"""
    return Questionnaire(path)
