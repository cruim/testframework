import utils
import validation


def test_compare_objects_results_expected(compare_objects_results_data):
    data = compare_objects_results_data["expected"]
    status_code, data = validation.validate_compare_objects_results(data)
    assert status_code == 0, status_code
    res = utils.compare_objects_results(**data)
    assert res is True, res


def test_compare_objects_results_unexpected(compare_objects_results_data):
    data = compare_objects_results_data["unexpected"]
    status_code, data = validation.validate_compare_objects_results(data)
    assert status_code == 0, status_code
    res = utils.compare_objects_results(**data)
    assert res is False, res


def test_compare_objects_results_invalid(compare_objects_results_data):
    data = compare_objects_results_data["invalid"]
    status_code, data = validation.validate_compare_objects_results(data)
    assert status_code == 1, status_code
    assert isinstance(data, str)
