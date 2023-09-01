import os
import shutil

import pytest


@pytest.fixture(scope="session")
def prepare_enviroment():
    tmp = os.system("pip install -r tests/requirements_tests.txt")
    os.system("mkdir tests/tmp")
    yield tmp
    # with open("tests/requirements_tests.txt", 'r') as f:
    #     for package in f:
    #         os.system(f"pip-autoremove -y {package}")
    # shutil.rmtree("tests/tmp")


@pytest.fixture
def read_dataset():
    import pandas as pd
    return pd.read_csv("tests/data/train_data.csv"), pd.read_csv("tests/data/y.csv")


@pytest.fixture
def compare_objects_results_data():
    obj = {
        "expected":
            {
                "vector_1": {"data": [
                    {"datep": "2016.01.01", "predict": 36},
                    {"datep": "2016.01.01", "predict": 42},
                    {"datep": "2016.01.01", "predict": 42},
                ]},
                "vector_2": {"data": [
                    {"datep": "2016.01.01", "predict": 36},
                    {"datep": "2016.01.01", "predict": 42},
                    {"datep": "2016.01.01", "predict": 42},
                ]},
                "result_feature": "predict",
                "ratio": 1.0,
            },
        "unexpected":
            {

                "vector_1": {"data": [
                    {"datep": "2016.01.01", "predict": 36},
                    {"datep": "2016.01.01", "predict": 42},
                    {"datep": "2016.01.01", "predict": 42},
                ]},
                "vector_2": {"data": [
                    {"datep": "2016.01.01", "predict": 36},
                    {"datep": "2016.01.01", "predict": 48},
                    {"datep": "2016.01.01", "predict": 42},
                ]},
                "result_feature": "predict",
                "ratio": 1.0,
            },
        "invalid": {
                "vector_1": {"data": [
                    {"datep": "2016.01.01", "predict": 36},
                    {"datep": "2016.01.01", "predict": 42},
                    {"datep": "2016.01.01", "predict": 42},
                ]},
                "vector_2": {"data": [
                    {"datep": "2016.01.01", "predict": 36},
                    {"datep": "2016.01.01", "predict": 42},
                    {"datep": "2016.01.01", "predict": 42},
                ]},
                "result_feature": [1, 2,3 ],
                "ratio": "test",
            },
    }

    return obj
