"""Neural Architecture Evolution

Genetic algorithms and neural architecture search.
"""

import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ArchitectureGenome:
    """Represents a neural architecture genome"""
    
    def __init__(self, layers: List[int]):
        self.layers = layers
        self.fitness = 0.0
        
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate architecture"""
        for i in range(len(self.layers)):
            if np.random.random() < mutation_rate:
                change = np.random.choice([-1, 0, 1])
                self.layers[i] = max(10, self.layers[i] + change * 10)


class NeuralEvolution:
    """Evolve neural architectures"""
    
    def __init__(self, population_size: int = 20):
        self.population: List[ArchitectureGenome] = []
        self.generation = 0
        
        # Initialize population
        for _ in range(population_size):
            layers = [np.random.randint(50, 200) for _ in range(3)]
            self.population.append(ArchitectureGenome(layers))
            
    def evolve(self, fitness_fn, generations: int = 10):
        """Evolve architectures over generations"""
        for gen in range(generations):
            # Evaluate fitness
            for genome in self.population:
                genome.fitness = fitness_fn(genome.layers)
                
            # Selection
            self.population.sort(key=lambda g: g.fitness, reverse=True)
            survivors = self.population[:len(self.population)//2]
            
            # Crossover and mutation
            offspring = []
            for _ in range(len(self.population) - len(survivors)):
                parent1, parent2 = np.random.choice(survivors, 2, replace=False)
                child_layers = [
                    p1 if np.random.random() < 0.5 else p2
                    for p1, p2 in zip(parent1.layers, parent2.layers)
                ]
                child = ArchitectureGenome(child_layers)
                child.mutate()
                offspring.append(child)
                
            self.population = survivors + offspring
            self.generation += 1
            
            best_fitness = max(g.fitness for g in self.population)
            logger.info(f"Generation {gen}: Best fitness = {best_fitness:.3f}")
            
        return self.population[0]  # Return best architecture


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def fitness(layers):
        return -sum(layers) * 0.001 + np.random.random()
        
    evolution = NeuralEvolution()
    best = evolution.evolve(fitness, generations=5)
    print(f"Best architecture: {best.layers}")
