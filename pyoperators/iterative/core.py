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

    iterate : perform one iteration and return current solution.

    callback : user-defined function to print status or save variables.

    cont : continue the optimization skipping initialiaztion.

    __call__ : performs the optimization until stop_condition is reached.

    """

    def initialize(self):
        self.iter_ = 0
        self.current_solution = None

    def callback(self):
        pass

    def iterate(self):
        """
        Perform one iteration and returns current solution.
        """
        self.iter_ += 1
        self.callback(self)
        # return value not used in loop but usefull in "interactive mode"
        return self.current_solution

    def at_exit(self):
        """
        Perform some task at exit.
        Does nothing by default.
        """
        pass

    def __call__(self):
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
