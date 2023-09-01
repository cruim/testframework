import pytest
import pickle


@pytest.mark.order(0)
def test_train_cb_classifier(prepare_enviroment, read_dataset):
    from catboost import CatBoostClassifier
    cb_classifier = CatBoostClassifier(one_hot_max_size=4, iterations=10, random_seed=0, allow_writing_files=False)
    X, y = read_dataset
    cb_classifier.fit(X, y)
    assert cb_classifier.feature_names_ == ['Pclass', 'SibSp', 'Parch', 'Sex_female', 'Sex_male']
    path_to_model = "tests/tmp/cb_model"
    cb_classifier.save_model(path_to_model)


@pytest.mark.order(1)
def test_train_rf_classifier(read_dataset):
    from sklearn.ensemble import RandomForestClassifier
    rf_classifier = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=1)
    X, y = read_dataset
    rf_classifier.fit(X, y)
    path_to_model = "tests/tmp/rf_model"
    with open(path_to_model, "wb") as f:
        pickle.dump(rf_classifier, f)


@pytest.mark.order(2)
def test_train_linear_regression(read_dataset):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    X, y = read_dataset
    reg.fit(X, y)
    path_to_model = "tests/tmp/reg_model"
    with open(path_to_model, "wb") as f:
        pickle.dump(reg, f)


@pytest.mark.order(3)
def test_train_logistic_regression(read_dataset):
    from sklearn.linear_model import LogisticRegression
    reg = LogisticRegression()
    X, y = read_dataset
    reg.fit(X, y)
    path_to_model = "tests/tmp/log_reg_model"
    with open(path_to_model, "wb") as f:
        pickle.dump(reg, f)


@pytest.mark.order(4)
def test_train_gaussian_nb(read_dataset):
    from sklearn.naive_bayes import GaussianNB
    reg = GaussianNB()
    X, y = read_dataset
    reg.fit(X, y)
    path_to_model = "tests/tmp/gaussian_nb_model"
    with open(path_to_model, "wb") as f:
        pickle.dump(reg, f)


@pytest.mark.order(5)
def test_train_svc(read_dataset):
    from sklearn.svm import SVC
    reg = SVC()
    X, y = read_dataset
    reg.fit(X, y)
    path_to_model = "tests/tmp/svc_model"
    with open(path_to_model, "wb") as f:
        pickle.dump(reg, f)


@pytest.mark.order(6)
def test_train_dtree_regressor(read_dataset):
    from sklearn.tree import DecisionTreeRegressor
    reg = DecisionTreeRegressor()
    X, y = read_dataset
    reg.fit(X, y)
    path_to_model = "tests/tmp/dtree_regressor_model"
    with open(path_to_model, "wb") as f:
        pickle.dump(reg, f)


@pytest.mark.order(7)
def test_train_knn_regressor(read_dataset):
    from sklearn.neighbors import KNeighborsRegressor
    reg = KNeighborsRegressor()
    X, y = read_dataset
    reg.fit(X, y)
    path_to_model = "tests/tmp/knn_regressor_model"
    with open(path_to_model, "wb") as f:
        pickle.dump(reg, f)


@pytest.mark.order(8)
def test_train_knn_classifier(read_dataset):
    from sklearn.neighbors import KNeighborsClassifier
    reg = KNeighborsClassifier()
    X, y = read_dataset
    reg.fit(X, y)
    path_to_model = "tests/tmp/knn_classifier_model"
    with open(path_to_model, "wb") as f:
        pickle.dump(reg, f)

@pytest.mark.order(9)
def test_train_gradient_boosting_classifier(read_dataset):
    from sklearn.ensemble import GradientBoostingClassifier
    reg = GradientBoostingClassifier()
    X, y = read_dataset
    reg.fit(X, y)
    path_to_model = "tests/tmp/gradient_boosting_classifier_model"
    with open(path_to_model, "wb") as f:
        pickle.dump(reg, f)


@pytest.mark.order(10)
def test_train_xgboost_classifier(read_dataset):
    from xgboost import XGBClassifier
    reg = XGBClassifier()
    X, y = read_dataset
    reg.fit(X, y)
    path_to_model = "tests/tmp/xgboost_classifier_model"
    with open(path_to_model, "wb") as f:
        pickle.dump(reg, f)


@pytest.mark.order(11)
def test_train_xgboost_regressor(read_dataset):
    from xgboost import XGBRegressor
    reg = XGBRegressor()
    X, y = read_dataset
    reg.fit(X, y)
    path_to_model = "tests/tmp/xgboost_regressor_model"
    with open(path_to_model, "wb") as f:
        pickle.dump(reg, f)
