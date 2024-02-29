# knapsack_solver.py
import numpy as np
from scipy.optimize import linprog

class KnapsackSolver:
    import numpy as np
    from scipy.optimize import linprog
    
    def __init__(self, capacity, weights, profits):
        self.capacity = capacity
        self.weights = weights
        self.profits = profits
        self.n = len(weights)
        self.maxProfit = 0
        self.bestItems = np.zeros(self.n)
    
    def solve_linear(self, include=None, exclude=None):
        if include is None:
            include = []
        if exclude is None:
            exclude = []

        num_items = len(self.weights)
        c = -np.array(self.profits)  # Coefficients for the objective function (negative for maximization)
        A = np.array([self.weights]) # Constraints matrix
        b = np.array([self.capacity]) # Constraints bounds
        x_bounds = [(0, 1) if i not in include and i not in exclude else (1, 1) if i in include else (0, 0) for i in range(num_items)]
        res = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds, method='highs') # Solve the linear programming problem
        return res.x, -res.fun
    
    def branch_and_bound(self):
        currentItems = np.zeros(self.n)
        return self._branch_and_bound(0, 0, 0, currentItems)

    def _branch_and_bound(self, level, currentWeight, currentProfit, currentItems):
        # Use linear relaxation to obtain an upper bound at this node
        _, linear_relaxation_profit = self.solve_linear(exclude=np.where(currentItems[:level] == 0)[0].tolist())
        
        # Prune the branch if the upper bound is not better than the current max profit
        if linear_relaxation_profit <= self.maxProfit:
            return self.maxProfit, self.bestItems
        
        if level == self.n:
            if currentProfit > self.maxProfit:
                self.maxProfit = currentProfit
                self.bestItems = np.copy(currentItems)
            return self.maxProfit, self.bestItems

        # Include the current item, if it does not exceed the capacity
        if currentWeight + self.weights[level] <= self.capacity:
            currentItems[level] = 1
            self._branch_and_bound(level + 1, currentWeight + self.weights[level], currentProfit + self.profits[level], currentItems)
        
        # Exclude the current item
        currentItems[level] = 0
        self._branch_and_bound(level + 1, currentWeight, currentProfit, currentItems)
        
        return self.maxProfit, self.bestItems

        