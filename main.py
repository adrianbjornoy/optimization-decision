import numpy as np
from knapsack_solver import KnapsackSolver

def load_data(filename):
    with open(filename, 'r') as file:
        data = file.read().splitlines()
    return [int(line) for line in data]

def solve_as_simplex(dataset): 
    capacity = load_data(f"p0{dataset}_c.txt")[0]
    weights = load_data(f"p0{dataset}_w.txt")
    profits = load_data(f"p0{dataset}_p.txt")
    optimal_selection = load_data(f"p0{dataset}_s.txt") # For validation

    solver = KnapsackSolver(capacity, weights, profits)
    solution, max_profit = solver.solve_linear()
    normal_solution = [x for x in solution]

    # Calculate profit from the optimal solution
    optimal_profit = sum(p * x for p, x in zip(profits, optimal_selection))

    print(f"Solution: {normal_solution}")
    print(f"Max Profit: {-max_profit}") # Negate because we minimized the negative profit
    print(f"Optimal selection: {optimal_selection}")
    print(f"Optimal profit: {optimal_profit}")




#solve_as_simplex(7)