import pytest
from click.testing import CliRunner

from template import Template
from testframework import wrap_model

runner = CliRunner()


def init_predict(model):
    model = Template().load(model=model)
    vector = {"Pclass": 1, "SibSp": 0, "Parch": 0, "Sex_female": 1, "Sex_male": 0}
    return model.predict(vector)


@pytest.mark.order(after="test_train_model.py::test_train_cb_classifier")
def test_wrap_api_cb_classifier():
    path_to_model = "tests/tmp/cb_model"
    result = runner.invoke(wrap_model, ["--model_type", "CatBoostClassifier", "--path_to_model", path_to_model,
                                        "--wrapped_name", "tests/tmp/cb_classifier_wrapped"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code
    init_predict("tests/tmp/cb_classifier_wrapped")


@pytest.mark.order(after="test_train_model.py::test_train_rf_classifier")
def test_wrap_api_rf_classifier():
    path_to_model = "tests/tmp/rf_model"
    result = runner.invoke(wrap_model, ["--model_type", "RandomForestClassifier", "--path_to_model", path_to_model,
                                        "--wrapped_name", "tests/tmp/rf_classifier_wrapped"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code
    init_predict("tests/tmp/rf_classifier_wrapped")


@pytest.mark.order(after="test_train_model.py::test_train_linear_regression")
def test_wrap_api_linear_regression():
    path_to_model = "tests/tmp/reg_model"
    result = runner.invoke(wrap_model, ["--model_type", "LinearRegression", "--path_to_model", path_to_model,
                                        "--wrapped_name", "tests/tmp/linear_reg_wrapped"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code
    init_predict("tests/tmp/linear_reg_wrapped")


@pytest.mark.order(after="test_train_model.py::test_train_logistic_regression")
def test_wrap_api_logistic_regression():
    path_to_model = "tests/tmp/log_reg_model"
    result = runner.invoke(wrap_model, ["--model_type", "LogisticRegression", "--path_to_model", path_to_model,
                                        "--wrapped_name", "tests/tmp/log_reg_wrapped"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code
    init_predict("tests/tmp/log_reg_wrapped")


@pytest.mark.order(after="test_train_model.py::test_train_gaussian_nb")
def test_wrap_api_gaussian_nb():
    path_to_model = "tests/tmp/gaussian_nb_model"
    result = runner.invoke(wrap_model, ["--model_type", "GaussianNB", "--path_to_model", path_to_model,
                                        "--wrapped_name", "tests/tmp/gaussian_nb_wrapped"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code
    init_predict("tests/tmp/gaussian_nb_wrapped")


@pytest.mark.order(after="test_train_model.py::test_train_svc")
def test_wrap_api_svc():
    path_to_model = "tests/tmp/svc_model"
    result = runner.invoke(wrap_model, ["--model_type", "SVC", "--path_to_model", path_to_model,
                                        "--wrapped_name", "tests/tmp/svc_wrapped"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code
    init_predict("tests/tmp/svc_wrapped")


@pytest.mark.order(after="test_train_model.py::test_train_dtree_regressor")
def test_wrap_api_dtree_regressor():
    path_to_model = "tests/tmp/dtree_regressor_model"
    result = runner.invoke(wrap_model, ["--model_type", "DecisionTreeRegressor", "--path_to_model", path_to_model,
                                        "--wrapped_name", "tests/tmp/dtree_regressor_wrapped"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code
    init_predict("tests/tmp/dtree_regressor_wrapped")


@pytest.mark.order(after="test_train_model.py::test_train_knn_regressor")
def test_wrap_api_knn_regressor():
    path_to_model = "tests/tmp/knn_regressor_model"
    result = runner.invoke(wrap_model, ["--model_type", "KNeighborsRegressor", "--path_to_model", path_to_model,
                                        "--wrapped_name", "tests/tmp/knn_regressor_wrapped"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code
    init_predict("tests/tmp/knn_regressor_wrapped")


@pytest.mark.order(after="test_train_model.py::test_train_knn_classifier")
def test_wrap_api_knn_classifier():
    path_to_model = "tests/tmp/knn_classifier_model"
    result = runner.invoke(wrap_model, ["--model_type", "KNeighborsClassifier", "--path_to_model", path_to_model,
                                        "--wrapped_name", "tests/tmp/knn_classifier_wrapped"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code
    init_predict("tests/tmp/knn_classifier_wrapped")


@pytest.mark.order(after="test_train_model.py::test_train_gradient_boosting_classifier")
def test_wrap_api_gradient_boosting_classifier():
    path_to_model = "tests/tmp/gradient_boosting_classifier_model"
    result = runner.invoke(wrap_model, ["--model_type", "GradientBoostingClassifier", "--path_to_model", path_to_model,
                                        "--wrapped_name", "tests/tmp/gradient_boosting_classifier_wrapped"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code
    init_predict("tests/tmp/gradient_boosting_classifier_wrapped")


@pytest.mark.order(after="test_train_model.py::test_train_xgboost_classifier")
def test_wrap_api_xgboost_classifier():
    path_to_model = "tests/tmp/xgboost_classifier_model"
    result = runner.invoke(wrap_model, ["--model_type", "XGBClassifier", "--path_to_model", path_to_model,
                                        "--wrapped_name", "tests/tmp/xgboost_classifier_wrapped"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code
    init_predict("tests/tmp/xgboost_classifier_wrapped")


@pytest.mark.order(after="test_train_model.py::test_train_xgboost_regressor")
def test_wrap_api_xgboost_regressor():
    path_to_model = "tests/tmp/xgboost_regressor_model"
    result = runner.invoke(wrap_model, ["--model_type", "XGBRegressor", "--path_to_model", path_to_model,
                                        "--wrapped_name", "tests/tmp/xgboost_regressor_wrapped"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code
    init_predict("tests/tmp/xgboost_regressor_wrapped")
