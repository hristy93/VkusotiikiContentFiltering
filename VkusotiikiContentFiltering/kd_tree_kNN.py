import numpy

from heapq import heappush, heappop, nlargest
from random import sample, choice
from time import time
from sklearn.model_selection import train_test_split

from kd_tree import kdtree, Tree
from VkusotiikiContentFiltering import prepare_data, test_with_k_fold_cross_validation


LEFT_IDX = 1
RIGHT_IDX = 2


def group_classes(queue, user_likes):
    r = []
    bucket, pref = {}, {}
    for score, recipe_info in queue:
        recipe, index = recipe_info
        if index in user_likes:
            r.append(index)
        like = user_likes[index]
        if like in bucket:
            bucket[like] += 1
            pref[like].append((score, recipe_info))
        else:
            bucket[like] = 1
            pref[like] = [(score, recipe_info)]
    print('Should like', r)
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

    # build k-d tree
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


def kNN(k, item, neighbours, indexes, user_likes, data):
    print('LIKES:', [i for i in neighbours if i in user_likes])
    queue = []
    print('Fill in neighbours:')
    for neighbour, indx in zip(neighbours, indexes):
        dist = numpy.dot(item, neighbour)
        if dist:
            print(dist, data[indx].get('name'))
            heappush(queue, (dist, (neighbour, indx)))

    queue.sort()
    # queue = queue[:k]

    # We calculate cos between vectors, so bigger cos => closer recipes
    queue = nlargest(k, queue)
    # --------------------------------------------------------------

    most_spread_class, pref = group_classes(queue, user_likes)
    print(most_spread_class)

    # return pref.get(1, [])
    # return pref.values()
    return pref

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
    # k = int(input('Enter k = '))
    k = 3

    fetched_data = prepare_data()
    data = fetched_data.get('data')
    tf_data = fetched_data.get('tf_data')
    user_likes = fetched_data.get('user_likes')
    
    train_recipe_ids, test_recipe_ids = train_test_split(user_likes)
    cl = True

    for item_id in test_recipe_ids:
        print('Test with', item_id, data[item_id].get('name'))
        item = tf_data[item_id]
        
        queue, indexes = generate_queue(item, tf_data, depth=2)
        result = kNN(k, item, queue, indexes, user_likes, data)
        
        count = count + 1 if cl in result else count
        print(item_id, cl in result)
        
        
        break
        
    print('Accuracy: {}%'.format(100 * count / len(test_recipe_ids)))


def recipes_main():
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
    
    # for i, r_id in enumerate(user_likes):
    #     if r_id:
    #         print('rec_id', i, data[i].get('name'))


    # test_with_k_fold_cross_validation(fav_recipe_ids, k_fold_count)
    item_id = 56
    k = 3

    item = tf_data.pop(item_id)
    print('\nChosen recipe id', item_id, data[item_id].get('name'))

    count = 0
    for i in range(1):
        queue, indexes = generate_queue(item, tf_data, depth=2)
        print('Q', len(queue))
        result = kNN(k, item, queue, indexes, user_likes, data)

        print('\nMost preferable recipes are:')

        for like, i in result.items():
            print('like', like)
            for recipe_info in i:
                print('is liked', recipe_info[1][1] in user_likes)
                print(recipe_info[0], data[recipe_info[1][1]].get('name'))
                if recipe_info[1][1] in fav_meat_recipe_ids:
                    count += 1

    print('\nFinal Accuracy:', count / (1 * k))


if __name__ == '__main__':
    main()

