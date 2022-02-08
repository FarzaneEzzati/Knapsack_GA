'''Solving Knapsack Problem using genetic algorithm '''
''' Import Libraries '''
import pandas as pd
import numpy as np
import random
from copy import copy
import matplotlib.pyplot as plt
import time

''' Swapper for crossover'''
def swapper(x,y):
    return (copy(y),copy(x))

''' Feasible Solutions Creation OR Import '''
def Feasible_Solution(allowed_weight,weights):

    feas_solus = np.zeros((feas_solu_num, len(weights)), dtype=int)  # to store feasible solutions

    # creating random feasible solutions
    for solution in range(feas_solu_num):

        weight_sum = 0 # to store the temporary weight
        remained_items = list(range(item_num)) # selecting items from this list

        for iter in range(feas_solu_num):
            flag = True
            while flag:
                random_var = random.randint(0,item_num)
                if random_var in remained_items:
                    if weight_sum + weights[random_var] <= allowed_weight:
                        feas_solus[solution][random_var] = 1
                        weight_sum += weights[random_var]
                        remained_items.remove(random_var);
                        flag = False
                    else:
                        remained_items.remove(random_var)
                        flag = False
                else:
                   continue

    # check for duplicates
    for i in range(feas_solu_num):
        for j in range(feas_solu_num):
            subtraction = np.subtract(feas_solus[i], feas_solus[j])
            if 1 or -1 in subtraction:
                continue
            else:
                print(str(i) + ' ' + str(j) + ' are duplicates. Please run again')

    # return the feasible solutions
    return feas_solus

''' Fitness Calculation'''
def Fitness(solution):
    return np.sum(np.dot(solution,importance))

''' PickUp TO Mate Function'''
def PickUpToMate(generation, new_generation):
    # first: pickup two couples to determin the winner of tornament
    # second: send the winners to cross over function
    # third: send them to mutation function
    # fourht: send them to new generation function
    # fifth: return the generated generation

    tornament_list = list(range(feas_solu_num))

    tornament_list1 = sorted(tornament_list, key= lambda k: random.random()) # sort randomly
    tornament_list2 = sorted(tornament_list, key= lambda k: random.random()) # sort randomly

    for parent in range(int(len(tornament_list2)/2)):
        ''' first tornament '''
        tornament1_1 = generation[tornament_list1[2*parent]]
        tornament1_2 = generation[tornament_list2[2*parent+1]]

        if Fitness(tornament1_1) > Fitness(tornament1_2): # parent with a higher fittnes is selected
            parent1 = tornament1_1
        else:
            parent1 = tornament1_2

        ''' second tornament '''
        tornament2_1 = generation[tornament_list2[2*parent]]
        tornament2_2 = generation[tornament_list2[2*parent+1]]

        if Fitness(tornament2_1) > Fitness(tornament2_2): # parent with a higher fittnes is selected
            parent2 = tornament2_1
        else:
            parent2 = tornament2_2

        child1, child2 = CrossOver(parent1, parent2)
        child1, child2 = Mutation(child1, child2)
        new_generation = NewGeneration(child1, child2, new_generation, parent)
    return new_generation

''' Cross Over Function'''
def CrossOver(parent1,parent2):

    middle = int(len(parent1)/2)-1 # find the middle of parents to crossover
    parent1[middle:], parent2[middle:] = swapper(parent1[middle:], parent2[middle:])  # crossing over
    child1 = parent1 # change the names
    child2 = parent2 # change the names

    weight1 = np.sum(np.dot(child1, weights))  # weight of child 1
    weight2 = np.sum(np.dot(child2, weights))  # weight of child 1

    if weight1 > allowed_weight or weight2 > allowed_weight:
        # created children must be feasible
        # new feasible child should be a good one

        child1_feasibility = weight1 > allowed_weight
        child2_feasibility = weight2 > allowed_weight

        if child1_feasibility: # if child 1 is not feasible
            importance1 = child1*importance
            importance1[np.where(importance1 == 0)] = 2 # just to find minimum importance better
            while weight1 > allowed_weight: # find less important items to remove and be feasible
                child1[np.argmin(importance1)] = 0
                weight1 -= weights[np.argmin(importance1)]

        if child2_feasibility: # if child 2 is not feasible
            importance2 = child2*importance
            importance2[np.where(importance2 == 0)] = 2 # just to find minimum importance better
            while weight2 > allowed_weight: # find less important items to remove and be feasible
                child2[np.argmin(importance2)] = 0
                weight2 -= weights[np.argmin(importance2)]
    # Now, children are generated and are feasible
    return child1, child2

''' Mutation Function'''
def Mutation(child1,child2):
    weight1 = np.sum(np.dot(child1,weights)) # child solution 1 weight
    weight2 = np.sum(np.dot(child2,weights)) # child solution 2 weight

    for item in range(item_num):
        random_figure1 = random.random() # a random number for item in child1
        random_figure2 = random.random() # a random number for item in child2

        if random_figure1 >= mutation_rate:
            if weight1 + weights[item] <= allowed_weight: # if adding new item is feasible do it, otherwise do'nt add it
                child1[item] = 1 - child1[item] # genome 1 to 0 | genome 0 to 1
        if random_figure2 >= mutation_rate:
            if weight2 + weights[item] <= allowed_weight: # if adding new item is feasible do it, otherwise do'nt add it
                child2[item] = 1 - child2[item] # genome 1 to 0 | genome 0 to 1

    # Now, our children have been muted and are ready to pass to a new generation
    return child1, child2

''' Create New Generation Function '''
def NewGeneration(child1, child2, add_to_generation, parent):
    add_to_generation[2*parent] = child1
    add_to_generation[2*parent+1] = child2
    return add_to_generation

''' Main '''
if __name__ == '__main__':
    ''' Initialization '''
    importance = [0.2,0.35,0.9,0.1,0.6,0.9,0.45,0.65,0.41,0.5,0.4,0.4,0.6,0.5,0.1,0.2,0.3,0.4,0.8,0.9] # [0,1]
    sum_importance = np.sum(importance) # maximum fitness
    allowed_weight = 4500       # grams
    weights = [120,250,200,350,500,1000,600,310,50,145,140,220,180,400,1000,150,240,160,150,200] # grams
    feas_solu_num = 10          # number of solutions created at first
    item_num = len(weights)     # total numbre of items
    mutation_rate = 0.7         # rate of mutation
    threshold = .8              # termination threshold
    generation_fitness = []     # to save fittness of solutions in each generation
    feas_solus = Feasible_Solution(allowed_weight,weights) # now we have feasible solutions and their fitness
    generation = feas_solus

    # caculate fitness of a each solution in generation
    for solution in range(len(generation)):
        generation_fitness.append(Fitness(generation[solution]))
    max_fitness = np.max(generation_fitness)

    ''' GA Loop '''
    generation_number = 1
    while max_fitness <= sum_importance*threshold: # termination condition

        # initialize
        generation_fitness = []
        new_generation = np.zeros((feas_solu_num,len(weights)))

        # GA
        generation = PickUpToMate(generation, new_generation)

        # caculate fitness of a each solution in generation
        for solution in range(len(generation)):
            generation_fitness.append(Fitness(generation[solution]))
        max_fitness = np.max(generation_fitness)

        generation_number += 1

    # Now we have the maximum fitness which terminated the loop
    ''' Evaluation '''
    print('Optimal found in Generation %s' % generation_number)
    print('The optimal solution is: ', generation[np.argmax(generation_fitness)], 'with fitness: ', np.max(generation_fitness), '/', sum_importance )
    print('Total weight is: ', np.sum(np.dot(generation[np.argmax(generation_fitness)], weights)), '/' , str(allowed_weight))
