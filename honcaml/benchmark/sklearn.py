from honcaml.benchmark import base


class SklearnBenchmark(base.BaseBenchmark):

    @classmethod
    def _clean_search_space(cls, search_space: dict) -> dict:
        """
        Given a dict with a search space for a model, this function gets the
        module of model to import and the hyperparameters search space and
        ensures that method exists.

        Args:
            search_space (Dict): a dict with the search space to explore

            Example of 'search_space' input parameter:
            {
                'n_estimators':
                    method: randint
                    values: [2, 110],
                  max_features:
                    method: choice
                    values: [sqrt, log2]
            }

        Returns:
            (Dict): a dict where for each hyperparameter the corresponding
            method to generate all possible values during the search.
        """
        cleaned_search_space = {}
        for hyper_parameter, space in search_space.items():
            cleaned_search_space[
                hyper_parameter] = super()._clean_parameter_search_space(space)
        return cleaned_search_space
