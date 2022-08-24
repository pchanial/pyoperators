"""
Implements iterative algorithm class.
"""
from copy import copy

import numpy as np

from .criterions import huber_criterion, norm2, quadratic_criterion
from .linesearch import optimal_step

__all__ = [
    'ConjugateGradient',
    'HuberConjugateGradient',
    'acg',
    'hacg',
    'fletcher_reeves',
    'polak_ribiere',
]


# DEPRECATED
class Algorithm:
    """
    Abstract class to define iterative algorithms.

    Attributes
    ----------

    niterations : int
        Current iteration number.

    Methods
    -------

    initialize : Set variables to initial state.

    run : performs the optimization until stop_condition is reached or
          Ctrl-C is pressed.

    next : perform one iteration and return current solution.

    callback : user-defined function to print status or save variables.

    cont : continue the optimization skipping initialiaztion.

    """

    def initialize(self):
        print('The Algorithm class is deprecated.')
        self.iter_ = 0
        self.current_solution = None

    def callback(self):
        pass

    def iterate(self, n=1):
        """
        Perform n iterations and return current solution.
        """
        for i in range(n):
            self.iter_ += 1
            self.callback(self)
        return self.current_solution

    def run(self):
        """
        Perform the optimization.
        """
        self.initialize()
        self.iterate()  # at least 1 iteration
        self.cont()
        self.at_exit()
        return self.current_solution

    def cont(self):
        """
        Continue an interrupted estimation (like call but avoid
        initialization).
        """
        while not self.stop_condition(self):
            self.iterate()
        return self.current_solution

    def at_exit(self):
        """
        Perform some task at exit.
        Does nothing by default.
        """
        pass

    def __call__(self):
        print("Deprecation warning: use 'run' method instead.")
        self.run()


# defaults
TOL = 1e-6
GTOL = 1e-6
MAXITER = None


# stop conditions
class StopCondition:
    """
    A class defining stop conditions for iterative algorithms.
    Must be called with an Algorithm instance as argument.
    """

    def _test_maxiter(self, algo):
        return algo.iter_ >= self.maxiter

    def _test_tol(self, algo):
        self.resid = np.abs(algo.last_criterion - algo.current_criterion)
        self.resid /= algo.first_criterion
        return self.resid < self.tol

    def _test_gtol(self, algo):
        return algo.current_gradient_norm < self.gtol

    _all_tests = [_test_maxiter, _test_tol, _test_gtol]

    def __init__(self, maxiter=None, tol=None, gtol=None, cond=np.any):
        """
        Generate a StopCondition instance.

        Parameters
        ----------
        maxiter: int (None)
            If not None, stops after a fixed number of iterations.
        tol: float (None)
            If not None, stops when the criterion decreases by less than
            tol times the first criterion value.
        gtol: float (None)
            If not None, stops when the norm of the gradient falls below
            gtol.
        cond: np.any, np.all
            If cond==np.any, stops when any of the above condition is True.
            If cond==np.all, stops when all of the above condition is True.
        """
        self.cond = cond
        self.maxiter = maxiter
        self.tol = tol
        self.gtol = gtol
        self.all_val = [self.maxiter, self.tol, self.gtol]
        # filter out tests with None values
        self.tests_val = [val for val in self.all_val if val is not None]
        self.tests = [
            test for test, val in zip(self._all_tests, self.all_val) if val is not None
        ]
        # store values for printing
        self.resid = None

    def __call__(self, algo):
        return self.cond([test(self, algo) for test in self.tests])

    def str(self, algo):
        """
        Returns a string with current condition values.
        """
        if self.resid is not None and self.tol is not None:
            return f'\t {self.resid:1.2e} / {self.tol:1.2e}'
        else:
            return '\t Residual'


default_stop = StopCondition(maxiter=MAXITER, tol=TOL, gtol=GTOL)

# update types


def fletcher_reeves(algo):
    """
    Fletcher-Reeves descent direction update method.
    """
    return algo.current_gradient_norm / algo.last_gradient_norm


def polak_ribiere(algo):
    """
    Polak-Ribiere descent direction update method.
    """
    b = np.dot(algo.current_gradient.T, (algo.current_gradient - algo.last_gradient))
    b /= np.norm(algo.last_gradient)
    return b


# callback function


class Callback:
    """
    A Callback instance is called by an Algorithm at each iteration
    with the Algorithm instance as input. It can be used to display
    convergence information at each iteration (iteration number,
    criterion value), display the current solution or store it on
    disk.
    """

    def __init__(self, verbose=False, savefile=None, shape=()):
        """
        Parameters
        ----------
        verbose: boolean (default False)
            If True, iteration number and criterion value are displayed.
        savefile: str or file
            If not None, the current iteration, criterion value and solution
            are stored with numpy savez function.
        shape: 2-tuple
            Shape of the solution.
            If not empty tuple, plot or imshow are called to display
            current solution (solution should be 1D or 2D).

        Returns
        -------
        None

        """
        self.verbose = verbose
        self.savefile = savefile
        self.shape = shape
        self.im = None

    def print_status(self, algo):
        if self.verbose:
            if algo.iter_ == 1:
                print('Iteration \t Criterion')
            print_str = '\t%i \t %e' % (algo.iter_, algo.current_criterion)
            print_str += algo.stop_condition.str(algo)
            print(print_str)

    def save(self, algo):
        if self.savefile is not None:
            var_dict = {
                'iter': algo.iter_,
                'criterion': algo.current_criterion,
                'solution': algo.current_solution,
            }
            np.savez(self.savefile, **var_dict)

    def imshow(self, algo):
        import matplotlib.pyplot as mp

        if algo.iter_ == 1:
            self.im = mp.imshow(algo.current_solution.reshape(self.shape))
        else:
            self.im.set_data(algo.current_solution.reshape(self.shape))
        mp.draw()
        mp.show()

    def plot(self, algo):
        import matplotlib.pyplot as mp

        if algo.iter_ == 1:
            self.im = mp.plot(algo.current_solution)[0]
        else:
            y = algo.current_solution
            self.im.set_ydata(y)
            mp.ylim((y.min(), y.max()))
        mp.draw()
        mp.show()

    def __call__(self, algo):
        if self.verbose:
            self.print_status(algo)
        if self.savefile is not None:
            self.save(algo)
        if self.shape is not None:
            if len(self.shape) == 1:
                self.plot(algo)
            elif len(self.shape) == 2:
                self.imshow(algo)


default_callback = Callback()

# algorithms


class ConjugateGradient(Algorithm):
    """
    Apply the conjugate gradient algorithm to a Criterion instance.

    Parameters
    ----------

    criterion : Criterion
        A Criterion instance. It should have following methods and attributes:
            __call__ : returns criterion values at given point
            diff : returns gradient (1st derivative) of criterion at given point
            shapein: the shape of the input of criterion

    x0 : ndarray (None)
        The first guess of the algorithm.

    callback : function (default_callback)
        Perform some printing / saving operations at each iteration.

    stop_condition : function (default_stop)
        Defines when the iterations should stop

    update_type : function (fletcher_reeves)
        Type of descent direction update : e.g. fletcher_reeves, polak_ribiere

    line_search : function (optimal step)
        Line search method to find the minimum along each direction at each
        iteration.

    Returns
    -------

    Returns an algorithm instance. Optimization is performed by
    calling this instance.

    """

    def __init__(
        self,
        criterion,
        x0=None,
        callback=default_callback,
        stop_condition=default_stop,
        update_type=fletcher_reeves,
        line_search=optimal_step,
        **kwargs,
    ):
        self.criterion = criterion
        self.gradient = criterion.diff
        self.shapein = self.criterion.shapein
        # functions
        self.callback = callback
        self.stop_condition = stop_condition
        self.update_type = update_type
        self.line_search = line_search
        self.kwargs = kwargs
        # to store values
        self.current_criterion = np.inf
        self.current_solution = None
        self.current_gradient = None
        self.current_gradient_norm = None
        self.current_descent = None
        self.last_criterion = np.inf
        self.last_solution = None
        self.last_gradient = None
        self.last_gradient_norm = None
        self.last_descent = None

    def initialize(self):
        """
        Initialize required values.
        """
        Algorithm.initialize(self)
        self.first_guess()
        self.first_criterion = self.criterion(self.current_solution)
        self.current_criterion = self.first_criterion

    def first_guess(self, x0=None):
        """
        Sets current_solution attribute to initial value.
        """
        if x0 is None:
            self.current_solution = np.zeros(np.prod(self.shapein))
        else:
            self.current_solution = copy(x0)

    # update_* functions encode the actual algorithm
    def update_gradient(self):
        self.last_gradient = copy(self.current_gradient)
        self.current_gradient = self.gradient(self.current_solution)

    def update_gradient_norm(self):
        self.last_gradient_norm = copy(self.current_gradient_norm)
        self.current_gradient_norm = norm2(self.current_gradient)

    def update_descent(self):
        if self.iter_ == 0:
            self.current_descent = -self.current_gradient
        else:
            self.last_descent = copy(self.current_descent)
            b = self.update_type(self)
            self.current_descent = -self.current_gradient + b * self.last_descent

    def update_solution(self):
        self.last_solution = copy(self.current_solution)
        a = self.line_search(self)
        self.current_solution += a * self.current_descent

    def update_criterion(self):
        self.last_criterion = copy(self.current_criterion)
        self.current_criterion = self.criterion(self.current_solution)

    def iterate(self):
        """
        Update all values.
        """
        self.update_gradient()
        self.update_gradient_norm()
        self.update_descent()
        self.update_solution()
        self.update_criterion()
        Algorithm.iterate(self)

    def at_exit(self):
        self.current_solution.resize(self.criterion.shapein)


class QuadraticConjugateGradient(ConjugateGradient):
    """
    A subclass of ConjugateGradient using a QuadraticCriterion.
    """

    def __init__(self, model, data, priors=[], hypers=[], covariances=None, **kwargs):
        criterion = quadratic_criterion(
            model, data, hypers=hypers, priors=priors, covariances=covariances
        )
        ConjugateGradient.__init__(self, criterion, **kwargs)


class HuberConjugateGradient(ConjugateGradient):
    """
    A subclass of ConjugateGradient using an HuberCriterion.
    """

    def __init__(self, model, data, priors=[], hypers=[], deltas=None, **kwargs):
        criterion = huber_criterion(
            model, data, hypers=hypers, priors=priors, deltas=deltas
        )
        ConjugateGradient.__init__(self, criterion, **kwargs)


# for backward compatibility
def define_stop_condition(**kwargs):
    defaults = {'maxiter': None, 'tol': TOL, 'gtol': GTOL, 'cond': np.any}
    new_kwargs = {k: kwargs.get(k, v) for k, v in defaults.items()}
    return StopCondition(**new_kwargs)


def define_callback(**kwargs):
    defaults = {'verbose': False, 'savefile': None, 'shape': ()}
    new_kwargs = {k: kwargs.get(k, v) for k, v in defaults.items()}
    return Callback(**new_kwargs)


def acg(
    model, data, priors=[], hypers=[], covariances=None, return_algo=False, **kwargs
):
    stop_condition = define_stop_condition(**kwargs)
    callback = define_callback(**kwargs)
    algorithm = QuadraticConjugateGradient(
        model,
        data,
        priors=priors,
        hypers=hypers,
        covariances=covariances,
        stop_condition=stop_condition,
        callback=callback,
        **kwargs,
    )
    sol = algorithm.run()
    if return_algo:
        return sol, algorithm
    else:
        return sol


def hacg(model, data, priors=[], hypers=[], deltas=None, return_algo=False, **kwargs):
    stop_condition = define_stop_condition(**kwargs)
    callback = define_callback(**kwargs)
    algorithm = HuberConjugateGradient(
        model,
        data,
        priors=priors,
        hypers=hypers,
        deltas=deltas,
        stop_condition=stop_condition,
        callback=callback,
        **kwargs,
    )
    sol = algorithm.run()
    return sol


# other


def normalize_hyper(hyper, y, x):
    """
    Normalize hyperparamaters so that they are independent of pb size
    """
    nx = float(x.size)
    ny = float(y.size)
    return np.asarray(hyper) * ny / nx
