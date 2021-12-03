import inspect
import numpy as np

from geodesic_shooting.utils.logger import getLogger


class BaseStepsizeController:
    """Base class for methods to control stepsizes and other line search parameters.

    These algorithms adjust parameters like minimal or maximal stepsize before performing the next
    optimization step. Furthermore, additional parameters of the line search algorithm can also be
    changed using these algorithms.
    """
    def __init__(self, line_search, log_frequency=10, log_level='INFO'):
        """Constructor.

        Parameters
        ----------
        line_search
            Line search algorithm that is used in the optimization.
        log_frequency
            Frequency of logging the stepsize.
        log_level
            Level of the log messages to display (required by the logger).
        """
        self.line_search = line_search

        self.log_frequency = log_frequency
        self.number_iterations_without_logging = 0

        self.logger = getLogger('stepsize_controller', level=log_level)

    def _check_parameters(self, parameters_line_search):
        """Function to check whether the parameters given to the stepsize controller can fit the
        parameters used in the line search algorithm.

        Parameters
        ----------
        parameters_line_search
            Dictionary provided for updating the stepsize and that is supposed to be checked
            whether it can be used by the line search algorithm.
        """
        args = inspect.getfullargspec(self.line_search.__call__)[0]
        assert all(param in args for param in parameters_line_search)

    def update(self, parameters_line_search, current_stepsize):
        """Function that updates the minimal and maximal stepsize or other parameters of the line
        search algorithm.

        This function should always first check if the parameters fit to the line search algorithm,
        i.e. it should first call `_check_parameters`.

        Parameters
        ----------
        parameters_line_search
            Dictionary with additional information for the line search algorithm that is to be
            updated.
        current_stepsize
            The last stepsize that was returned by the line search algorithm.

        Returns
        -------
        A dictionary with the adjusted parameters.
        """
        self._check_parameters(parameters_line_search)
        assert 'max_stepsize' in parameters_line_search
        assert 'min_stepsize' in parameters_line_search

        old_max_stepsize = parameters_line_search['max_stepsize']
        old_min_stepsize = parameters_line_search['min_stepsize']

        parameters_line_search['max_stepsize'] = min(parameters_line_search['max_stepsize'],
                                                     current_stepsize)
        parameters_line_search['min_stepsize'] = min(parameters_line_search['min_stepsize'],
                                                     parameters_line_search['max_stepsize'])

        if not np.isclose(old_max_stepsize, parameters_line_search['max_stepsize']):
            self.logger.info(f'Updating maximum stepsize to {parameters_line_search["max_stepsize"]:.3e} ...')

        if not np.isclose(old_min_stepsize, parameters_line_search['min_stepsize']):
            self.logger.info(f'Updating minimum stepsize to {parameters_line_search["min_stepsize"]:.3e} ...')

        if self.number_iterations_without_logging >= self.log_frequency:
            self.logger.info(f'Current maximum stepsize: {parameters_line_search["max_stepsize"]:.3e}')
            self.logger.info(f'Current minimum stepsize: {parameters_line_search["min_stepsize"]:.3e}')
            self.number_iterations_without_logging = 0

        self.number_iterations_without_logging += 1

        return parameters_line_search


class PatientStepsizeController(BaseStepsizeController):
    """Stepsize controller that waits some iterations before changing stepsize.

    A fixed number of iterations with reduced stepsize before finally adjusting minimal
    and maximal stepsize can be prescribed. The new maximal stepsize will be set to the
    maximal stepsize that occured during waiting before adjusting the stepsizes.
    """
    def __init__(self, line_search, patience=10, log_frequency=10, log_level='INFO'):
        """Constructor.

        Parameters
        ----------
        line_search
            Line search algorithm that is used in the optimization.
        patience
            Number of iterations to wait before changing stepsize.
        log_frequency
            Frequency of logging the stepsize.
        log_level
            Level of the log messages to display (required by the logger).
        """
        self.line_search = line_search

        self.patience = patience
        self.maximal_reduced_stepsize = None
        self.number_iterations_without_changing_stepsize = 0

        self.log_frequency = log_frequency
        self.number_iterations_without_logging = 0

        self.logger = getLogger('stepsize_controller', level=log_level)

    def update(self, parameters_line_search, current_stepsize):
        """Function that updates the minimal and maximal stepsize of the line search algorithm.

        Whether or not to update the stepsize depends on the number of consecutive iterations in
        which the stepsize was reduced.

        Parameters
        ----------
        parameters_line_search
            Dictionary with additional information for the line search algorithm that is to be
            updated.
        current_stepsize
            The last stepsize that was returned by the line search algorithm.

        Returns
        -------
        A dictionary with the adjusted parameters.
        """
        self._check_parameters(parameters_line_search)
        assert 'max_stepsize' in parameters_line_search
        assert 'min_stepsize' in parameters_line_search

        old_max_stepsize = parameters_line_search['max_stepsize']
        old_min_stepsize = parameters_line_search['min_stepsize']

        if not np.isclose(parameters_line_search['max_stepsize'], current_stepsize):
            if self.number_iterations_without_changing_stepsize < self.patience:
                if self.maximal_reduced_stepsize is None:
                    self.maximal_reduced_stepsize = current_stepsize
                else:
                    self.maximal_reduced_stepsize = max(self.maximal_reduced_stepsize, current_stepsize)
                self.number_iterations_without_changing_stepsize += 1
            else:
                parameters_line_search['max_stepsize'] = self.maximal_reduced_stepsize
                assert not np.isclose(parameters_line_search['max_stepsize'], old_max_stepsize)
                self.logger.info(f'Updating maximum stepsize to {parameters_line_search["max_stepsize"]:.3e} ...')

                parameters_line_search['min_stepsize'] = min(parameters_line_search['min_stepsize'],
                                                             parameters_line_search['max_stepsize'])
                if not np.isclose(old_min_stepsize, parameters_line_search['min_stepsize']):
                    self.logger.info(f'Updating minimum stepsize to {parameters_line_search["min_stepsize"]:.3e} ...')

                self.number_iterations_without_changing_stepsize = 0
                self.maximal_reduced_stepsize = None
        else:
            self.number_iterations_without_changing_stepsize = 0
            self.maximal_reduced_stepsize = None

        if self.number_iterations_without_logging >= self.log_frequency:
            self.logger.info(f'Current maximum stepsize: {parameters_line_search["max_stepsize"]:.3e}')
            self.logger.info(f'Current minimum stepsize: {parameters_line_search["min_stepsize"]:.3e}')
            self.number_iterations_without_logging = 0

        self.number_iterations_without_logging += 1

        return parameters_line_search
