from setuptools import setup

setup(
    name="dnn-evaluation-helper",
    version="0.1.0",
    packages=["dnn_evaluation-helper-package"],
    entry_points={
        "console_scripts": [
            "dnn_eva_helper = app.__main__:main"
        ]
    },
)