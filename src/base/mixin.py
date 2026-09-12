class ClassifierMixin:
    def score(self, X, y):
        from metrics import accuracy_score

        return accuracy_score(y, self.predict(X))  # type: ignore


class RegressorMixin:
    def score(self, X, y):
        from metrics import r2_score

        return r2_score(y, self.predict(X))  # type: ignore


class TransformerMixin:
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)  # type: ignore
