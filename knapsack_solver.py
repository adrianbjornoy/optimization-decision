# knapsack_solver.py
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

class KnapsackSolver:
    import numpy as np
    from scipy.optimize import linprog
    import matplotlib.pyplot as plt
    
    def __init__(self, capacity, weights, profits):
        self.capacity = capacity
        self.weights = weights
        self.profits = profits
        self.n = len(weights)
        self.maxProfit = 0
        self.bestItems = np.zeros(self.n)
        self.lp_counter = 0
    
    def solve_linear(self, include=None, exclude=None):
        if include is None:
            include = []
        if exclude is None:
            exclude = []
        self.lp_counter += 1
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
        _,linear_relaxation_profit = self.solve_linear(exclude=np.where(currentItems[:level] == 0)[0].tolist())
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

    def genetic_algorithm(self, population_size=20, mutation_rate=0.1, generations=100):
        
        def create_individual():
            return np.random.choice([0, 1], size=(self.n,))
    
        def compute_fitness(individual, weights, profits, capacity, penalty_rate=0.1):
            weight = np.sum(individual * weights)
            profit = np.sum(individual * profits)
            if weight > capacity:
                return 0
            else:
                return profit

        def crossover(parent1, parent2):
            crossover_point = np.random.randint(1, self.n-1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            return child1, child2
        
        def mutate(individual):
            for i in range(self.n):
                if np.random.rand() < mutation_rate:
                    individual[i] = 1 - individual[i]
            return individual

        # Initialize population
        population = [create_individual() for _ in range(population_size)]
        best_solution = None
        best_fitness = -np.inf
        max_fitness_over_generations = []
        diversity_over_generations = []

        for generation in range(generations):
            # Evaluate fitness for every individual in the current generation
            fitnesses = [compute_fitness(individual, self.weights, self.profits, self.capacity) for individual in population]

            # Selection
            parents_indices = np.random.choice(population_size, size=population_size, replace=True, p=np.array(fitnesses)/sum(fitnesses))
            parents = [population[i] for i in parents_indices]
            
            # Crossover and mutation
            next_population = []
            for i in range(0, population_size, 2):
                parent1, parent2 = parents[i], parents[i+1]
                child1, child2 = crossover(parent1, parent2)
                child1, child2 = mutate(child1), mutate(child2)
                next_population.extend([child1, child2])

            population = next_population
        
            # Evaluate fitness for every individual in the current generation
            fitnesses = [compute_fitness(individual, self.weights, self.profits, self.capacity) for individual in population]
            current_best_fitness = max(fitnesses)
            #print(current_best_fitness)
            current_best_idx = fitnesses.index(current_best_fitness)
            current_best_solution = population[current_best_idx]
            #print(current_best_solution)

            diversity = len(np.unique(np.vstack(population), axis=0))
            max_fitness_over_generations.append(current_best_fitness)
            diversity_over_generations.append(diversity)

            if current_best_fitness > best_fitness:
                print(f"Updating best fitness: {best_fitness} -> {current_best_fitness}")
                best_fitness = current_best_fitness
                best_solution = current_best_solution

                actual_profit = np.sum(best_solution * self.profits)
                actual_weight = np.sum(best_solution * self.weights)
        
        # Plotting the maximum fitness over generations
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
        plt.plot(max_fitness_over_generations, label='Max Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Maximum Fitness over Generations')
        plt.legend()

        # Plotting the diversity over generations
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
        plt.plot(diversity_over_generations, label='Diversity', color='orange')
        plt.xlabel('Generation')
        plt.ylabel('Diversity')
        plt.title('Population Diversity over Generations')
        plt.legend()

        plt.tight_layout()
        plt.show()
        return best_solution, actual_profit, actual_weight