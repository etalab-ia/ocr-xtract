import logging
from time import time

from skopt.space import Categorical, Integer, Real
from tqdm import tqdm


class tqdm_skopt(object):
    def __init__(self, **kwargs):
        self._bar = tqdm(**kwargs)

    def __call__(self, res):
        self._bar.update()


def create_dimensions_from_parameters(parameters):
    """
    This function creates dimensions for the scikit optimize experiment :param parameters: config file :return: list of
    dimensions for scikit optimize
    """
    dim1 = Integer(
        name="k_retriever",
        low=min(parameters["k_retriever"]),
        high=max(parameters["k_retriever"]) + 1,
    )
    dim2 = Integer(
        name="k_title_retriever",
        low=min(parameters["k_title_retriever"]),
        high=max(parameters["k_title_retriever"]) + 1,
    )
    dim3 = Integer(
        name="k_reader_per_candidate",
        low=min(parameters["k_reader_per_candidate"]),
        high=max(parameters["k_reader_per_candidate"]) + 1,
    )
    dim4 = Integer(
        name="k_reader_total",
        low=min(parameters["k_reader_total"]),
        high=max(parameters["k_reader_total"]) + 1,
    )
    dim5 = Categorical(
        name="reader_model_version", categories=parameters["reader_model_version"]
    )
    dim6 = Categorical(
        name="retriever_model_version", categories=parameters["retriever_model_version"]
    )
    dim7 = Categorical(name="retriever_type", categories=parameters["retriever_type"])
    dim8 = Categorical(name="squad_dataset", categories=parameters["squad_dataset"])
    dim9 = Categorical(name="filter_level", categories=parameters["filter_level"])
    dim10 = Categorical(name="preprocessing", categories=parameters["preprocessing"])
    dim11 = Integer(
        name="boosting",
        low=min(parameters["boosting"]),
        high=max(parameters["boosting"]) + 1,
    )
    dim12 = Categorical(name="split_by", categories=parameters["split_by"])
    dim13 = Integer(
        name="split_length",
        low=min(parameters["split_length"]),
        high=max(parameters["split_length"]) + 1,
    )

    dimensions = [
        dim1,
        dim2,
        dim3,
        dim4,
        dim5,
        dim6,
        dim7,
        dim8,
        dim9,
        dim10,
        dim11,
        dim12,
        dim13,
    ]

    return dimensions


class LoggingCallback(object):
    """
    Callback to control the verbosity.

    Parameters ---------- n_init : int, optional Number of points provided by the user which are yet to be evaluated.
    This is equal to `len(x0)` when `y0` is None

    n_random : int, optional Number of points randomly chosen.

    n_total : int Total number of func calls.

    Attributes ---------- iter_no : int Number of iterations of the optimization routine.
    """

    def __init__(self, n_total, n_init=0, n_random=0):
        self.n_init = n_init
        self.n_random = n_random
        self.n_total = n_total
        self.iter_no = 1

        self._start_time = time()
        self._print_info(start=True)

    def _print_info(self, start=True):
        iter_no = self.iter_no
        if start:
            status = "started"
            eval_status = "Evaluating function"
            search_status = "Searching for the next optimal point."

        else:
            status = "ended"
            eval_status = "Evaluation done"
            search_status = "Search finished for the next optimal point."

        if iter_no <= self.n_init:
            logging.info(
                "Iteration No: %d %s. %s at provided point."
                % (iter_no, status, eval_status)
            )

        elif self.n_init < iter_no <= (self.n_random + self.n_init):
            logging.info(
                "Iteration No: %d %s. %s at random point."
                % (iter_no, status, eval_status)
            )

        else:
            logging.info("Iteration No: %d %s. %s" % (iter_no, status, search_status))

    def __call__(self, res):
        """
        Parameters ---------- res : `OptimizeResult`, scipy object The optimization as a OptimizeResult object.
        """
        time_taken = time() - self._start_time
        self._print_info(start=False)

        curr_y = res.func_vals[-1]
        curr_min = res.fun

        logging.info("Time taken: %0.4f" % time_taken)
        logging.info("Function value obtained: %0.4f" % curr_y)
        logging.info("Current minimum: %0.4f" % curr_min)

        self.iter_no += 1
        if self.iter_no <= self.n_total:
            self._print_info(start=True)
            self._start_time = time()
