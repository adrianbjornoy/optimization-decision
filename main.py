import numpy as np
import time
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

    start_time = time.time()
    solver = KnapsackSolver(capacity, weights, profits)
    solution, max_profit = solver.solve_linear()
    normal_solution = [x for x in solution]
    end_time = time.time()

    # Calculate profit from the optimal solution
    optimal_profit = sum(p * x for p, x in zip(profits, optimal_selection))

    print(f"Solution: {normal_solution}")
    print(f"Max Profit: {max_profit}") # Negate because we minimized the negative profit
    print(f"Optimal selection: {optimal_selection}")
    print(f"Optimal profit: {optimal_profit}")
    print(f"Elapsed time: {end_time-start_time} seconds")

def solve_as_branch_and_bound(dataset):
    capacity = load_data(f"p0{dataset}_c.txt")[0]
    weights = load_data(f"p0{dataset}_w.txt")
    profits = load_data(f"p0{dataset}_p.txt")
    optimal_selection = load_data(f"p0{dataset}_s.txt") # For validation

    #Initialize the solver with the problem data
    start_time = time.time()
    solver = KnapsackSolver(capacity, weights, profits)
    
    # Calculate the optimal solution using Branch and Bound
    max_profit, best_items = solver.branch_and_bound()
    end_time = time.time()
    # Print the results
    print("Optimal Solution (Selected Items):", best_items)
    print("Maximum Profit:", max_profit)
    print(f"Elapsed time: {end_time-start_time} seconds")
    print("Number of times linear programming was done:", solver.lp_counter)

def solve_as_genetic_algorithm(dataset):
    capacity = load_data(f"p0{dataset}_c.txt")[0]
    weights = load_data(f"p0{dataset}_w.txt")
    profits = load_data(f"p0{dataset}_p.txt")
    optimal_selection = load_data(f"p0{dataset}_s.txt") # For validation

    start_time = time.time()
    solver = KnapsackSolver(capacity, weights, profits)
    solution, profit, weight = solver.genetic_algorithm()
    end_time = time.time()

    print("GA Solution:", solution)
    print("GA Profit:", profit)
    print("GA Weight:", weight )
    print(f"Elapsed time: {end_time-start_time} seconds")

#solve_as_simplex(2)
#solve_as_branch_and_bound(8)
solve_as_genetic_algorithm(7)