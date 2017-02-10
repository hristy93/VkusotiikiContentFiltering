import numpy

from heapq import heappush, heappop, nlargest
from random import sample, choice
from time import time

from kd_tree import kdtree, Tree
from VkusotiikiContentFiltering import prepare_data


LEFT_IDX = 1
RIGHT_IDX = 2


def group_classes(queue, user_likes):
    bucket, pref = {}, {}
    for score, recipe_info in queue:
        recipe, index = recipe_info
        like = user_likes[index]
        if like in bucket:
            bucket[like] += 1
            pref[like].append((score, recipe_info))
        else:
            bucket[like] = 1
            pref[like] = [(score, recipe_info)]
    return bucket, pref


def get_neighbours(subtree, res=[]):
    if subtree is None:
        return res

    for item in subtree:
        if item is None:
            continue
        if type(item) is Tree and not all(map(lambda x: x is None, subtree)):
            return get_neighbours(item, res=res)
        elif type(item) is list:
            res.append(item)
    return res


def generate_queue(item, tf_data, depth=1000):
    start = time()
    tree = kdtree(tf_data)
    print('\nGenerating KD tree took:', time() - start, 's')

    neighbours = []
    # copy tree
    subtree = Tree(**tree._asdict())
    k = 0

    for level, attr in enumerate(item):
        if subtree and not all(map(lambda x: x is None, subtree)):
            idx = LEFT_IDX if subtree[0] is not None and attr < subtree[0][level] else RIGHT_IDX
            subtree = subtree[idx]
            if k == depth - 1:
                break
            k += 1

    neighbours = get_neighbours(subtree)
    neighbours_indexes = [tf_data.index(n) for n in neighbours]
    # print('neighbours', neighbours)

    return neighbours, neighbours_indexes


def kNN(k, item, neighbours, indexes, user_likes):
    queue = []
    for neighbour, indx in zip(neighbours, indexes):
        dist = numpy.dot(item, neighbour)
        if dist:
            heappush(queue, (dist, (neighbour, indx)))

    # queue.sort()
    # queue = queue[:k]

    # We calculate cos between vectors, so bigger cos => closer recipes
    queue = nlargest(k, queue)

    most_spread_class, pref = group_classes(queue, user_likes)
    print(most_spread_class)
    # print('PREF', pref)
    return [recipe_info for recipe_info in pref.get(1, [])]

    if len(set(most_spread_class.values())) == 1:
        found_class = sorted(list(most_spread_class.items()), key=lambda x: x[1])[-1][0]
    else:
        found = max(most_spread_class.values())
        classes = [obj for obj, cl in most_spread_class.items() if cl == found]
        closest_cl = heappop(queue)
        found_class = closest_cl if closest_cl in classes else choice(classes)
    return found_class


def main():
    count = 0
    k = int(input('Enter k = '))
    dataset = read_data('data.csv')
    test_data = sample(dataset, 20)
    trainers = set(dataset).difference(set(test_data))
    for item in test_data:
        item = list(item)
        cl = item.pop()
        found_class = generate_queue(item, trainers, k)
        count = count + 1 if cl == found_class else count
        print(item, found_class, cl == found_class)
    print('Accuracy: {}%'.format(100 * count / len(test_data)))


def recipes_main():
    # load data
    fetched_data = prepare_data()
    data = fetched_data.get('data')
    # ingredients = fetched_data.get('ingredients')
    # ingredients_count = fetched_data.get('ingredients_count')
    # data_count = fetched_data.get('data_count')
    tf_data = fetched_data.get('tf_data')
    # idf_data = fetched_data.get('idf_data')
    # tfidf_data = fetched_data.get('tfidf_data')
    user_likes = fetched_data.get('user_likes')
    # best_user_pref_count = fetched_data.get('best_user_pref_count')
    # use_random_likes = fetched_data.get('use_random_likes')
    # recipe_ids_test = fetched_data.get('recipe_ids_test')
    # best_recipe_count = fetched_data.get('best_recipe_count')
    # recipe_ids_train = fetched_data.get('recipe_ids_train')
    # ingredient_data = fetched_data.get('ingredient_data')

    fav_meat_recipe_ids = fetched_data.get('fav_meat_recipe_ids')

    item_id = 354
    k = 3

    item = tf_data.pop(item_id)
    print('\nChosen recipe id', item_id, data[item_id].get('name'))

    count = 0
    for i in range(10):
        queue, indexes = generate_queue(item, tf_data, depth=2)
        print('Q', len(queue))
        result = kNN(k, item, queue, indexes, user_likes)
        print('\nMost preferable recipes are:')
        for recipe_info in result:
            print(recipe_info[0], data[recipe_info[1][1]].get('name'))
            if recipe_info[1][1] in fav_meat_recipe_ids:
                count += 1

    print('\nFinal Accuracy:', count / 10 * k)


if __name__ == '__main__':
    recipes_main()
