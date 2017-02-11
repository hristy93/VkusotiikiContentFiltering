import csv
import numpy

from heapq import heappush, heappop, nlargest
from math import sqrt
from random import sample, choice

from sklearn.model_selection import train_test_split

from VkusotiikiContentFiltering import prepare_data


def group_classes(elements, user_likes):
    bucket = {}
    for element in elements:
        score, trainer_id = element
        like = user_likes[trainer_id]
        if like in bucket:
            bucket[like] += 1
        else:
            bucket[like] = 1
    return bucket


def get_recipes_names_by_id(data, ids):
    res = []
    for id in ids:
        res.append(data[id].get('name') if id < len(data) else '')
    return res


def calculate_euclid_dist(item, recipe):
    current_sum = 0
    for i in range(len(recipe)):
        current_sum += (float(item[i]) - float(recipe[i])) ** 2
    return sqrt(current_sum)


def generate_queue(item, trainers_ids, tf_data, k, user_likes):
    queue = []

    for trainer_id in trainers_ids:
        recipe = tf_data[trainer_id]
        
        euclid_dist = calculate_euclid_dist(item, recipe)

        heappush(queue, (euclid_dist, trainer_id))

    queue.sort()
    new_elements = queue[:k]
        
    most_spread_class = group_classes(new_elements, user_likes)

    if len(set(most_spread_class.values())) == len(set(most_spread_class.keys())):
        found_class = sorted(list(most_spread_class.items()), key=lambda x: x[1])[-1][0]
    else:
        k = max(most_spread_class.values())
        classes = [item for item, cl in most_spread_class.items() if cl == k]
        closest_cl = heappop(new_elements)
        found_class = closest_cl if closest_cl in classes else choice(classes)

    return found_class


def main():
    count = 0

    set_length = input('Enter traning + test sets length. Default value is {}. Your value: '.format(100))
    set_length = int(set_length if set_length else 100)

    k = input('Enter value for k. Default is k = {}. Your value: '.format(int(sqrt(set_length))))
    k = int(k if k else sqrt(set_length))
    print()

    fetched_data = prepare_data()
    data = fetched_data.get('data')
    tf_data = fetched_data.get('tf_data')
    user_likes = fetched_data.get('user_likes')
    fav_recipe_ids = fetched_data.get('fav_recipe_ids')

    train_recipe_ids, test_recipe_ids = train_test_split(sample(range(len(data)), set_length))

    print('\n\nkNN with k = {}\n'.format(k))
    
    print('Total number of recipes to train and test {}'.format(set_length))
    
    print('Count of recipes in TRAIN set {}. Num of liked recipes from TRAIN set: {}'.format(
        len(train_recipe_ids),
        len([i for i in train_recipe_ids if user_likes[i]])))
     
    print('Count of recipes in TEST set {}. Num liked recipes from TEST set: {}'.format(
        len(test_recipe_ids),
        len([i for i in test_recipe_ids if user_likes[i]])))

    print()

    accuracy = 0
    for recipe_id in test_recipe_ids:
        item = tf_data[recipe_id]

        found_class = generate_queue(item, train_recipe_ids, tf_data, k, user_likes)

        count = count + 1 if found_class == user_likes[recipe_id] else count

        if found_class == user_likes[recipe_id] and found_class:
            print('Found match with id = {} called: {}'.format(
                recipe_id, data[recipe_id].get('name')))

    print('Accuracy: {0:.2f}%'.format((100 * count) / len(test_recipe_ids)))


if __name__ == '__main__':
    main()

