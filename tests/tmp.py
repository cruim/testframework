vector1 = {
    "report_date": "01.10.2021",
    "latitude": 56.857255,
    "longitude": 52.196086,
    "zone_id": 86716658,
    "object_class": 1,
    "comissioning_date": "30.06.2-21",
    "walls_material_id": 3,
    "decoration_id": 2,
    "number_of_floors": 16
}

vector2 = {
    "report_date": "01.10.2021",
    "latitude": 56.857255,
    "longitude": 52.196086,
    "zone_id": 86716658,
    "object_class": 4,
    "comissioning_date": "30.06.2-21",
    "walls_material_id": 3,
    "decoration_id": 2,
    "number_of_floors": 16
}

(vector1, vector2, 1.1)



# генерация векторов на основе train_test_split, посчитать метрику на основе тестовых векторов, например усреднится по
# ответам и сравнить, что значение не меньше какого-то значения

# сравнение результатов идентичных векторов с различными значениями фичей, например vector1 отличается от vector2,
# только object_class

# compare_metrics

# compare_ift_accuracy

vector1 = {"Pclass": 1, "SibSp": 1, "Parch": 0, "Sex_female": 0, "Sex_male": 1}
vector2 = {"Pclass": 2, "SibSp": 1, "Parch": 0, "Sex_female": 0, "Sex_male": 1}

x,y = [0.3423, 0.123]

d1 = {"data": [
    {"datep": "2016.01.01", "predict": 36},
    {"datep": "2016.01.01", "predict": 42},
    {"datep": "2016.01.01", "predict": 42},
]}

d2 = {"data": [
    {"datep": "2016.01.01", "predict": 36},
    {"datep": "2016.01.01", "predict": 42},
    {"datep": "2016.01.01", "predict": 42},
]}


def compare_results(obj1, obj2, feature, ratio):
    result = True
    if isinstance(obj1, dict) and feature in obj1:
        return obj1[feature]*ratio >= obj2[feature]
    if isinstance(obj1, dict):
        for key in obj1:
            if isinstance(obj1[key], (list, dict)):
                result = result and compare_results(obj1[key], obj2[key], feature, ratio)
    elif isinstance(obj1, list):
        for i in range(len(obj1)):
            if isinstance(obj1[i], dict):
                result = result and compare_results(obj1[i], obj2[i], feature, ratio)
    return result
