import pickle

import pandas as pd
import numpy as np


class Template:
    """
    Template for wrapping ML models
    """

    def __init__(self, model_config=None, name="wrapped_model"):
        if model_config:
            self.name = name
            self.config = model_config
            for _import in self.config["import"]:
                exec(_import)
            for _init in self.config["init"]:
                exec(_init)

    def predict(self, vector):
        predict = getattr(self.model, "predict")
        vector = pd.DataFrame([vector])
        result = predict(vector)
        result = result.tolist() if isinstance(result, np.ndarray) else result
        return result

    def get_metrics(self):
        pass

    def save(self):
        pickle.dump(self, open(self.name, "wb"))

    def load(model):
        obj = pickle.load(open(model, "rb"))
        return obj

    load = staticmethod(load)
