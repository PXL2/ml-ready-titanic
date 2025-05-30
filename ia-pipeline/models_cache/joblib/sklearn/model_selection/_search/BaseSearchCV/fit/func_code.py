# first line: 1372
        @functools.wraps(fit_method)
        def wrapper(estimator, *args, **kwargs):
            global_skip_validation = get_config()["skip_parameter_validation"]

            # we don't want to validate again for each call to partial_fit
            partial_fit_and_fitted = (
                fit_method.__name__ == "partial_fit" and _is_fitted(estimator)
            )

            if not global_skip_validation and not partial_fit_and_fitted:
                estimator._validate_params()

            with config_context(
                skip_parameter_validation=(
                    prefer_skip_nested_validation or global_skip_validation
                )
            ):
                return fit_method(estimator, *args, **kwargs)
