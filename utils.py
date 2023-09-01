import csv
import json
import pathlib
from collections.abc import Collection
from typing import Union, Tuple, Dict, List

from template import Template
import validation


def model_config(model="") -> Dict:
    config = {
        "CatBoostClassifier": {
            "import": ("from catboost import CatBoostClassifier",),
            "init": ("self.model = CatBoostClassifier()", f"self.model.load_model('{model}')"),
        },
        "RandomForestClassifier": {
            "import": ("from sklearn.ensemble import RandomForestClassifier",),
            "init": (f"self.model = pickle.load(open('{model}', 'rb'))",),
        },
        "LinearRegression": {
            "import": ("from sklearn.linear_model import LinearRegression",),
            "init": (f"self.model = pickle.load(open('{model}', 'rb'))",),
        },
        "LogisticRegression": {
            "import": ("from sklearn.linear_model import LogisticRegression",),
            "init": (f"self.model = pickle.load(open('{model}', 'rb'))",),
        },
        "GaussianNB": {
            "import": ("from sklearn.naive_bayes import GaussianNB",),
            "init": (f"self.model = pickle.load(open('{model}', 'rb'))",),
        },
        "SVC": {
            "import": ("from sklearn.svm import SVC",),
            "init": (f"self.model = pickle.load(open('{model}', 'rb'))",),
        },
        "DecisionTreeRegressor": {
            "import": ("from sklearn.tree import DecisionTreeRegressor",),
            "init": (f"self.model = pickle.load(open('{model}', 'rb'))",),
        },
        "KNeighborsRegressor": {
            "import": ("from sklearn.neighbors import KNeighborsRegressor",),
            "init": (f"self.model = pickle.load(open('{model}', 'rb'))",),
        },
        "KNeighborsClassifier": {
            "import": ("from sklearn.neighbors import KNeighborsClassifier",),
            "init": (f"self.model = pickle.load(open('{model}', 'rb'))",),
        },
        "GradientBoostingClassifier": {
            "import": ("from sklearn.ensemble import GradientBoostingClassifier",),
            "init": (f"self.model = pickle.load(open('{model}', 'rb'))",),
        },
        "XGBClassifier": {
            "import": ("from xgboost import XGBClassifier",),
            "init": (f"self.model = pickle.load(open('{model}', 'rb'))",),
        },
        "XGBRegressor": {
            "import": ("from xgboost import XGBRegressor",),
            "init": (f"self.model = pickle.load(open('{model}', 'rb'))",),
        },
    }
    return config


def wrap(model_type: str, model: str, wrapped_name: str):
    config = model_config(model)
    if model_type not in config:
        raise KeyError(f"There is no {model_type}")
    model = Template(model_config=config[model_type], name=wrapped_name)
    model.save()


def read_file(file: str) -> Tuple[int, Union[str, Dict, List]]:
    path_to_file = pathlib.Path(file)
    if not path_to_file.exists():
        return 1, f"File {file} not found"
    with open(path_to_file, 'r') as f:
        try:
            result = json.load(f)
            return 0, result
        except json.decoder.JSONDecodeError:
            if file.endswith(".csv"):
                try:
                    with open(path_to_file, 'r') as f:
                        result = []
                        reader = csv.DictReader(f)
                        for row in map(dict, reader):
                            result.append(row)
                        return 0, result
                except Exception:
                    return 1, f"file {file} extension is not json or csv"
    return 1, f"file {file} extension is not json or csv"


def save_report(obj: Dict):
    with open("ift_report.json", 'w') as f:
        f.write(json.dumps(obj))


def validate_acceptable_range(expected_result: Collection, actual_result: Collection) -> bool:
    if len(expected_result) != len(actual_result):
        return False
    for expected, actual in zip(expected_result, actual_result):
        if actual == expected:
            continue
        elif isinstance(expected, list) and isinstance(actual, (int, float)) and expected[0] <= actual <= expected[1]:
            continue
        else:
            return False
    return True


def validate_config(config, test_name):
    exit_code, obj = read_file(file=config)
    if exit_code:
        return exit_code, obj
    if not obj.get(test_name):
        return 1, f"There is no {test_name} key in config"
    exit_code, obj = validation.validate_config(obj=obj)
    return exit_code, obj


def make_ift(config: str, model: str, acceptable_range: bool) -> Tuple[int, Union[Dict, str]]:
    exit_code, obj = validate_config(config=config, test_name="ift")
    if exit_code:
        return exit_code, obj
    model = Template.load(model=model)
    total_vectors = len(obj["ift"])
    matched_results = 0
    for row in obj["ift"]:
        result = model.predict(row["vector"])
        if not acceptable_range:
            matched_results += result == row["expected_result"]
        else:
            matched_results += validate_acceptable_range(row["expected_result"], result)

    percentage = matched_results / total_vectors
    ift_result = {"total_vectors": total_vectors, "matched_vectors": matched_results, "percentage": percentage}
    save_report(ift_result)
    return 0, ift_result


def compare_objects_results(vector_1: Dict, vector_2: Dict, result_feature: str, ratio: float) -> bool:
    result = True
    if isinstance(vector_1, dict) and result_feature in vector_1:
        return vector_1[result_feature] * ratio >= vector_2[result_feature]
    if isinstance(vector_1, dict):
        for key in vector_1:
            if isinstance(vector_1[key], (list, dict)):
                result = result and compare_objects_results(vector_1[key], vector_2[key], result_feature, ratio)
    elif isinstance(vector_1, list):
        for i in range(len(vector_1)):
            if isinstance(vector_1[i], dict):
                result = result and compare_objects_results(vector_1[i], vector_2[i], result_feature, ratio)
    return result


def compare_lists_results(vector_1: List, vector_2: List, ratio: float) -> bool:
    for x, y in zip(vector_1, vector_2):
        if isinstance(x, (int, float)) and isinstance(x, (int, float)) and x * ratio < y:
            return False
    return True


def validate_categorical_feature(model, config):
    exit_code, obj = validate_config(config=config, test_name="validate_categorical_feature")
    if exit_code:
        return exit_code, obj
    model = Template.load(model=model)
    total_vectors = len(obj["validate_categorical_feature"])
    matched_results = 0
    for row in obj["validate_categorical_feature"]:
        vector = row["vector"]
        tmp = []
        feature = row["feature"]
        result_feature = row.get("result_feature")
        for value in row["values"]:
            vector[feature] = value
            tmp.append(model.predict(vector))
        for i in range(len(row["ratio"])):
            ratio = row["ratio"][i]
            vector_1 = tmp[i]
            vector_2 = tmp[i + 1]
            if result_feature:
                schema = validation.CategoricalFeatureConfig()
                dct = schema.load({"vector_1": vector_1, "vector_2": vector_2,
                                   "result_feature": result_feature, "ratio": ratio})
                result = compare_objects_results(**dct)
            elif isinstance(vector_1, list):
                result = compare_lists_results(vector_1, vector_2, ratio)
            else:
                result = False
            matched_results += result
    return 0, {"total_vectors": total_vectors, "matched_results": matched_results}
