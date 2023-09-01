import pytest
from click.testing import CliRunner

from testframework import validate_categorical_feature

runner = CliRunner()


@pytest.mark.order(after="test_train_model.py::test_train_dtree_regressor")
def test_dtree_regressor():
    result = runner.invoke(validate_categorical_feature, ["--path_to_model", "tests/tmp/cb_classifier_wrapped",
                                                          "--config", "tests/data/ift_acceptable_range.json"])
    assert not result.exception, result.stdout
    assert result.exit_code == 0, result.exit_code
