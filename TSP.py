
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math
import random

# coordinates = pd.read_csv('input_0.csv')
# coordinates = pd.read_csv('input_1.csv')
# coordinates = pd.read_csv('input_2.csv')
# coordinates = pd.read_csv('input_3.csv')
# coordinates = pd.read_csv('input_4.csv')
coordinates = pd.read_csv('input_5.csv')
# coordinates = pd.read_csv('input_6.csv')
# coordinates = pd.read_csv('input_7.csv')

w,h = len(coordinates),len(coordinates.columns)
coordinates = coordinates.values

# aquire distance matrix

distance = np.zeros((w, w))
for i in range(w):
    for j in range(w):
        distance[i, j] = distance[j, i] = np.linalg.norm(coordinates[i] - coordinates[j])

count = 300
iter_time = 1000
retain_rate = 0.3
random_select_rate = 0.5
mutation_rate = 0.1
gailiang_N = 3000

def get_total_distance(x):
    dista = 0
    for i in range(len(x)):
        if i == len(x) - 1:
            dista += distance[x[i]][x[0]]
        else:
            dista += distance[x[i]][x[i + 1]]
    return dista

def gailiang(x):
    distance = get_total_distance(x)
    gailiang_num = 0
    while gailiang_num < gailiang_N:
        while True:
            a = random.randint(0, len(x) - 1)
            b = random.randint(0, len(x) - 1)
            if a != b:
                break
        new_x = x.copy()
        temp_a = new_x[a]
        new_x[a] = new_x[b]
        new_x[b] = temp_a
        if get_total_distance(new_x) < distance:
            x = new_x.copy()
        gailiang_num += 1

def nature_select(population):
    grad = [[x, get_total_distance(x)] for x in population]
    grad = [x[0] for x in sorted(grad, key=lambda x: x[1])]
    retain_length = int(retain_rate * len(grad))
    parents = grad[: retain_length]
    for ruozhe in grad[retain_length:]:
        if random.random() < random_select_rate:
            parents.append(ruozhe)
    return parents

def crossover(parents):
    target_count = count - len(parents)
    children = []
    while len(children) < target_count:
        while True:
            male_index = random.randint(0, len(parents)-1)
            female_index = random.randint(0, len(parents)-1)
            if male_index != female_index:
                break
        male = parents[male_index]
        female = parents[female_index]
        left = random.randint(0, len(male) - 2)
        right = random.randint(left, len(male) - 1)
        gen_male = male[left:right]
        gen_female = female[left:right]
        child_a = []
        child_b = []

        len_ca = 0
        for g in male:
            if len_ca == left:
                child_a.extend(gen_female)
                len_ca += len(gen_female)
            if g not in gen_female:
                child_a.append(g)
                len_ca += 1

        len_cb = 0
        for g in female:
            if len_cb == left:
                child_b.extend(gen_male)
                len_cb += len(gen_male)
            if g not in gen_male:
                child_b.append(g)
                len_cb += 1

        children.append(child_a)
        children.append(child_b)
    return children

def mutation(children):
    for i in range(len(children)):
        if random.random() < mutation_rate:
            while True:
                u = random.randint(0, len(children[i]) - 1)
                v = random.randint(0, len(children[i]) - 1)
                if u != v:
                    break
            temp_a = children[i][u]
            children[i][u] = children[i][v]
            children[i][v] = temp_a


def get_result(population):
    grad = [[x, get_total_distance(x)] for x in population]
    grad = sorted(grad, key=lambda x: x[1])
    return grad[0][0], grad[0][1]


population = []
index = [i for i in range(w)]
for i in range(count):
    x = index.copy()
    random.shuffle(x)
    gailiang(x)
    population.append(x)

distance_list = []
result_cur_best, dist_cur_best = get_result(population)
distance_list.append(dist_cur_best)

i = 0
while i < iter_time:
    parents = nature_select(population)
    children = crossover(parents)
    mutation(children)
    population = parents + children
    result_cur_best, dist_cur_best = get_result(population)
    distance_list.append(dist_cur_best)
    i = i + 1
print(dist_cur_best)

for i in range(len(result_cur_best)):
    result_cur_best[i] += 1

result_path = result_cur_best
result_path.append(result_path[0])

X = []
Y = []
for index in result_path:
    X.append(coordinates[index-1, 0])
    Y.append(coordinates[index-1, 1])

plt.figure(1)
plt.plot(X, Y, '-o')
for i in range(len(X)):
    plt.text(X[i] + 0.05, Y[i] + 0.05, str(result_path[i]), color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Route')

#input_0: 3291.6217214092458
#input_1: 3778.7154164925378
#input_2: 4494.417962262894
#input_3: 11381.949193167071
#input_4: 25074.07549278131
#input_5: 156288.000608771
