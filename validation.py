from marshmallow import Schema, fields, ValidationError, validate, validates_schema
from typing import Union, Tuple, Dict


class CategoricalFeatureConfig(Schema):
    vector = fields.Dict(required=True)
    feature = fields.String(required=True)
    values = fields.List(fields.Raw, validate=validate.Length(min=2), required=True)
    ratio = fields.List(fields.Float(), validate=validate.Length(min=1), required=True)
    result_feature = fields.String(required=False)

    @validates_schema
    def validate_feature(self, data, **kwargs):
        if data["feature"] not in data["vector"]:
            raise ValidationError(f"Field feature {data['feature']} should be in field vector")

    @validates_schema
    def validate_ratio(self, data, **kwargs):
        if len(data["values"]) - len(data["ratio"]) != 1:
            raise ValidationError("Length of field ratio should be less then length of field values by  exactly 1")


class IFT(Schema):
    vector = fields.Raw(required=True)
    expected_result = fields.Raw(required=True)
    actual_result = fields.Raw(required=False, allow_none=True)


class TestConfig(Schema):
    ift = fields.List(fields.Nested(nested=IFT), required=False)
    validate_categorical_feature = fields.List(fields.Nested(nested=CategoricalFeatureConfig), required=False)


def validate_config(obj: Dict) -> Tuple[int, Union[str, Dict]]:
    schema = TestConfig()
    try:
        schema.load(obj)
    except ValidationError as e:
        return 1, str(e)
    else:
        return 0, obj


class CompareObjectsResults(Schema):
    vector_1 = fields.Dict(required=True)
    vector_2 = fields.Dict(required=True)
    result_feature = fields.String(required=True)
    ratio = fields.Float(required=True)


def validate_compare_objects_results(obj: Dict) -> Tuple[int, Union[str, Dict]]:
    schema = CompareObjectsResults()
    try:
        schema.load(obj)
    except ValidationError as e:
        return 1, str(e)
    else:
        return 0, obj
