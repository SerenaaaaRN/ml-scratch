class BaseEstimator:
    def get_params(self, deep: bool = True):
        import inspect

        sig = inspect.signature(self.__init__)
        params = {}

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            params[name] = getattr(self, name, param.default)

        return params

    def set_params(self, **params):
        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key} for estimator {self.__class__.__name__}"
                    f"Valid parameter are: {list(valid_params.keys())}"
                )
            setattr(self, key, value)
        return self

    def _check_is_fitted(self, attribute=None):
        if attribute is None:
            fitted_attrs = [
                attr
                for attr in dir(self)
                if attr.endswith("_") and not attr.startswith("_")
            ]
            if not fitted_attrs:
                raise ValueError(
                    f"This {self.__class__.__name__} instance is not fitted yet"
                    "Call 'fit' with appropriate arguments before using this estimator"
                )
        else:
            missing = [attr for attr in attribute if not hasattr(self, attr)]
            if missing:
                raise ValueError(
                    f"This {self.__class__.__name__} instance is not fitted yet"
                    f"Missing attribute: {missing}"
                )

    def __repr__(self) -> str:
        params = self.get_params()
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"
