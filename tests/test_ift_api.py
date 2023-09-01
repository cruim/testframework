import pytest
from click.testing import CliRunner
from testframework import ift


runner = CliRunner()


@pytest.mark.order(after="test_wrap_api.py::test_wrap_api_cb_classifier")
def test_ift_api_cb_classifier():
    result = runner.invoke(ift, ["--path_to_model", "tests/tmp/cb_classifier_wrapped",
                                 "--config", "tests/data/ift.json"])
    assert not result.exception, result.stdout
    assert result.exit_code == 0, result.exit_code


@pytest.mark.order(after="test_wrap_api.py::test_wrap_api_rf_classifier")
def test_ift_api_rf_classifier():
    result = runner.invoke(ift, ["--path_to_model", "tests/tmp/rf_classifier_wrapped",
                                 "--config", "tests/data/ift.json"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code


@pytest.mark.order(after="test_wrap_api.py::test_wrap_api_linear_regression")
def test_ift_api_linear_regression():
    result = runner.invoke(ift, ["--path_to_model", "tests/tmp/linear_reg_wrapped", "--config", "tests/data/ift.json"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code


@pytest.mark.order(after="test_wrap_api.py::test_wrap_api_logistic_regression")
def test_ift_api_logistic_regression():
    result = runner.invoke(ift, ["--path_to_model", "tests/tmp/log_reg_wrapped", "--config", "tests/data/ift.json"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code


@pytest.mark.order(after="test_wrap_api.py::test_wrap_api_gaussian_nb")
def test_ift_api_gaussian_nb():
    result = runner.invoke(ift, ["--path_to_model", "tests/tmp/gaussian_nb_wrapped", "--config", "tests/data/ift.json"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code


@pytest.mark.order(after="test_wrap_api.py::test_wrap_api_svc")
def test_ift_api_svc():
    result = runner.invoke(ift, ["--path_to_model", "tests/tmp/svc_wrapped", "--config", "tests/data/ift.json"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code


@pytest.mark.order(after="test_wrap_api.py::test_wrap_api_dtree_regressor")
def test_ift_api_dtree_regressor():
    result = runner.invoke(ift, ["--path_to_model", "tests/tmp/dtree_regressor_wrapped",
                                 "--config", "tests/data/ift_acceptable_range.json", "--acceptable_range", True])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code


@pytest.mark.order(after="test_wrap_api.py::test_wrap_api_knn_regressor")
def test_ift_api_knn_regressor():
    result = runner.invoke(ift, ["--path_to_model", "tests/tmp/knn_regressor_wrapped",
                                 "--config", "tests/data/ift.json"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code


@pytest.mark.order(after="test_wrap_api.py::test_wrap_api_knn_classifier")
def test_ift_api_knn_classifier():
    result = runner.invoke(ift, ["--path_to_model", "tests/tmp/knn_classifier_wrapped",
                                 "--config", "tests/data/ift.json"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code


@pytest.mark.order(after="test_wrap_api.py::test_wrap_api_gradient_boosting_classifier")
def test_ift_api_gradient_boosting_classifier():
    result = runner.invoke(ift, ["--path_to_model", "tests/tmp/gradient_boosting_classifier_wrapped",
                                 "--config", "tests/data/ift.json"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code


@pytest.mark.order(after="test_wrap_api.py::test_wrap_api_xgboost_classifier")
def test_ift_api_xgboost_classifier():
    result = runner.invoke(ift, ["--path_to_model", "tests/tmp/xgboost_classifier_wrapped",
                                 "--config",  "tests/data/ift.json"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code


@pytest.mark.order(after="test_wrap_api.py::test_wrap_api_xgboost_regressor")
def test_ift_api_xgboost_regressor():
    result = runner.invoke(ift, ["--path_to_model", "tests/tmp/xgboost_regressor_wrapped",
                                 "--config", "tests/data/ift.json"])
    assert not result.exception, result.exception
    assert result.exit_code == 0, result.exit_code


