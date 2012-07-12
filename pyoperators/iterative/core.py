f
from __future__ import division, print_function

__all__ = ['Algorithm']

class Algorithm(object):
    """
    Abstract class to define iterative algorithms.

    Attributes
    ----------

    iter_ : int
        Current iteration number.

    Methods
    -------

    initialize : Set variables to initial state.

    run : performs the optimization until stop_condition is reached or
          Ctrl-C is pressed.

    iterate : perform one iteration and return current solution.

    callback : user-defined function to print status or save variables.

    cont : continue the optimization skipping initialiaztion.

    """
    def initialize(self):
        self.iter_ = 0
        self.current_solution = None

    def callback(self):
        pass

    def iterate(self, n=1):
        """
        Perform n iterations and return current solution.
        """
        for i in xrange(n):
            self.iter_ += 1
            self.callback(self)
        return self.current_solution

    def run(self):
        """
        Perform the optimization.
        """
        self.initialize()
        self.iterate() # at least 1 iteration
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

