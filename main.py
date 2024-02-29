import numpy as np
from scipy.optimize import linprog

class KnapsackSolver:
    def __init__(self, capacity, weights, profits):
        self.capacity = capacity
        self.weights = weights
        self.profits = profits
    
    def solve_linear(self):
        num_items = len(self.weights)
        c = -np.array(self.profits)  # Coefficients for the objective function (negative for maximization)
        A = np.array([self.weights]) # Constraints matrix
        b = np.array([self.capacity]) # Constraints bounds
        x_bounds = [(0, 1) for _ in range(num_items)] # Bounds for each variable

        res = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds, method='highs') # Solve the linear programming problem
        return res.x, res.fun

def load_data(filename):
    with open(filename, 'r') as file:
        data = file.read().splitlines()
    return [int(line) for line in data]

def main():
    capacity = load_data('p02_c.txt')[0]
    weights = load_data('p02_w.txt')
    profits = load_data('p02_p.txt')
    optimal_selection = load_data('p02_s.txt') # For validation

    solver = KnapsackSolver(capacity, weights, profits)
    solution, max_profit = solver.solve_linear()
    normal_solution = [x for x in solution]

    # Calculate profit from the optimal solution
    optimal_profit = sum(p * x for p, x in zip(profits, optimal_selection))

    print(f"Solution: {normal_solution}")
    print(f"Max Profit: {-max_profit}") # Negate because we minimized the negative profit
    print(f"Optimal selection: {optimal_selection}")
    print(f"Optimal profit: {optimal_profit}")
if __name__ == "__main__":
    main()