from click.testing import CliRunner

from testframework import available_models, available_tests

runner = CliRunner()


def test_available_models_api():
    result = runner.invoke(available_models)
    assert not result.exception, result.exception
    assert result.exit_code == 0


def test_available_tests_api():
    result = runner.invoke(available_tests)
    assert not result.exception, result.exception
    assert result.exit_code == 0
    assert isinstance(result.output, str), result.output
