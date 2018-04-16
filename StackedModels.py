__author__ = 'franzi'

import numpy as np
from sklearn.base import clone

class StackingModels():

    def __init__(self, base_models, meta_model):

        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X, y):
        self.base_models_ = [clone(x) for x in self.base_models]

        for base_model in self.base_models_:
            base_model.fit(X, y)

        predictions = np.column_stack([
            base_model.predict(X) for base_model in self.base_models_
        ])

        self.meta_model_ = clone(self.meta_model)

        self.meta_model_.fit(predictions, y)

        return self

    def predict(self, X):

        predictions = np.column_stack([
            base_model.predict(X) for base_model in self.base_models_
        ])

        return self.meta_model_.predict(predictions)

