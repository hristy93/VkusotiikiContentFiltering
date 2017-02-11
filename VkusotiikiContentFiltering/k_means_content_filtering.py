import random
import math
from heapq import nlargest
import numpy as np
import sys

from sklearn.model_selection import KFold
from VkusotiikiContentFiltering import prepare_data
from VkusotiikiContentFiltering import train_test_split
from VkusotiikiContentFiltering import generate_user_likes_by_recipes_ids
from VkusotiikiContentFiltering import generate_user_profile
from VkusotiikiContentFiltering import generate_user_pref
from VkusotiikiContentFiltering import print_recipes_info
from VkusotiikiContentFiltering import get_accuracy_with_test_data

""" Returns a random element in an array """
def random_element(arr, taken):
    elem = random.choice(arr)
    
    while elem in taken:
        elem = random.choice(arr)

    return elem

""" Selects recipes for initial centroids"""
def get_initial_centroids_indeces(filtered_tf_data, k):
    recipe_indeces = list(filtered_tf_data.keys())
    taken = []
    centroids = {}

    for x in range(0, k):
        elem = random_element(recipe_indeces, taken)
        taken.append(elem)
        centroids[x] = filtered_tf_data[elem]

    return centroids

""" Returns the indeces of the recipes that are not selected for centroids """
def filter_centroids(centroids_indeces, data):
    rest_indeces = list(filter((lambda x: x not in centroids_indeces), data))
    rest = map((lambda x: data[x]), rest_indeces)

    return dict(zip(rest_indeces, rest))

""" Finds the jaccard distance between two recipes """
def jaccard_distance(recipe, centroid):
    recipe_ingredients = [i for i, e in enumerate(recipe) if e != 0]
    centroid_ingredients = [i for i, e in enumerate(centroid) if e != 0]
    intersect = set(recipe_ingredients).intersection(set(centroid_ingredients))
    union = set(recipe_ingredients).union(set(centroid_ingredients))

    return len(intersect) / len(union)

""" Builds the clusters around the centroids """
def build_clusters(filtered_tf_data, centroids, k):
    clusters = {}
    all_indeces = list(filtered_tf_data.keys())

    """ Initialize clusters """
    for num in list(centroids.keys()):
        clusters[num] = []
    
    for index in filtered_tf_data:
        recipe = filtered_tf_data[index]
        clusters[choose_centroid(recipe, centroids)].append(index)

    return clusters

""" Finds the k closest recipes to a given recipe between a specific cluster """
def find_closest(tf_data, recipes, k, recipe, train_likes, searched_index):
    proximities = {}

    for recipe_index in recipes:
        if (recipe_index not in train_likes and recipe_index != searched_index):
            proximities[recipe_index] = proximity(tf_data[recipe_index], recipe)

    return nlargest(k, proximities, key=proximities.get)

""" Finds the proximity between two recipes """
def proximity(recipe, alternate_recipe):
    return np.dot(recipe, alternate_recipe)

def max_by_prop(arr, prop):
    return max(arr, key=(lambda x: x["proximity"]))

""" Рeturns the centroid closest centroid """
def choose_centroid(recipe, centroids):
    centroids_indeces = list(centroids.keys())
    proximities = list(map((lambda c: {
        "centroid": c,
        "proximity": proximity(recipe, centroids[c])
    }), centroids_indeces))

    return max_by_prop(proximities, "proximity")["centroid"]

""" Checks if the there was any transitions of recipes between clusters """
def static_clusters(old, new):
    clusters = list(old.keys())
    static_clusters = list(filter(lambda x: set(old[x]) == set(new[x]), clusters))

    return len(static_clusters) == len(clusters)

""" Finds the centroids for the next algorithm iteration """
def next_centroids(clusters, filtered_tf_data):
    centroids = {}

    for cluster_id in clusters:
        cluster_recipes_indeces = list(filter((lambda x: x in clusters[cluster_id]),
            list(filtered_tf_data.keys())))
        cluster_recipes = list(map((lambda x: filtered_tf_data[x]), cluster_recipes_indeces))
        
        if len(cluster_recipes) == 0:
            centroids[cluster_id] = [0]*430
        else:
            centroids[cluster_id] = list(np.mean(cluster_recipes, axis=0))

    return centroids

""" Finds the cluster in which a given recipe belongs """
def predict_cluster(recipe, centroids, clusters):
    centroid = choose_centroid(recipe, centroids)

    return clusters[centroid]

def kmeans(tf_data, filtered_tf_data, k, best_recipe_pref_index, train_likes):
    centroids = get_initial_centroids_indeces(filtered_tf_data, k)
    clusters = build_clusters(filtered_tf_data, centroids, k)
    iteration = 1

    while iteration < 40:
        centroids = next_centroids(clusters, filtered_tf_data)
        old_clusters = clusters
        clusters = build_clusters(filtered_tf_data, centroids, k)
        iteration += 1

        if static_clusters(old_clusters, clusters):
            print("Clusters are static after {} iterations".format(iteration))
            break

    predicted_indeces = predict_cluster(tf_data[best_recipe_pref_index], centroids, clusters)

    predicted_tf_data = list(map((lambda x: tf_data[x]), predicted_indeces))
    
    k_for_closest = int(math.sqrt(len(predicted_indeces)))
    
    print("Searching for {} closest recipes".format(k_for_closest))
    
    k_closest = find_closest(tf_data, predicted_indeces, k_for_closest, tf_data[best_recipe_pref_index], train_likes, best_recipe_pref_index)
    
    return k_closest

def run_tests(fav_recipe_ids, k_fold_count, user_likes,
    tf_data, idf_data, ingredients_count, data_count, use_random_likes, data):
    kf = KFold(n_splits=k_fold_count)
    k_fold_index = 0

    print("Running tests")

    for fav_ids_train, fav_ids_test in kf.split(fav_recipe_ids):
        recipe_ids_train, recipe_ids_test = train_test_split(fav_recipe_ids)

        print("\n Test {}: ".format(k_fold_index))

        k_fold_index += 1
        user_likes = generate_user_likes_by_recipes_ids(data_count, recipe_ids_train)
        user_profile = generate_user_profile(tf_data, ingredients_count, user_likes)
        user_pref = generate_user_pref(data_count, tf_data, idf_data, user_profile)
        best_user_pref = max(user_pref)
        best_recipe_pref_index = user_pref.index(best_user_pref)

        train_likes = [item for index, item in enumerate(user_likes) if index not in recipe_ids_test]
        test_likes = [item for index, item in enumerate(user_likes) if index in recipe_ids_test]

        formated_tf_data = {index : item for index, item in enumerate(tf_data)}
        best_recipe_pref_index = random.choice(range(500))
        k_means_predictions = kmeans(tf_data, formated_tf_data, 16, best_recipe_pref_index, recipe_ids_train)
        
        print("\n Searched:")
        print_recipes_info(data, [best_recipe_pref_index])
        
        print("\n Suggested:")
        print_recipes_info(data, k_means_predictions)

        # if not use_random_likes: 
            # accuracy = get_accuracy_with_test_data(k_means_predictions, recipe_ids_test, user_likes)

def main():
    fetched_data = prepare_data()
    data = fetched_data.get('data')
    ingredients = fetched_data.get('ingredients')
    ingredients_count = fetched_data.get('ingredients_count')
    data_count = fetched_data.get('data_count')
    tf_data = fetched_data.get('tf_data')
    idf_data = fetched_data.get('idf_data')
    user_likes = fetched_data.get('user_likes')
    best_user_pref_count = fetched_data.get('best_user_pref_count')
    use_random_likes = fetched_data.get('use_random_likes')
    recipe_ids_test = fetched_data.get('recipe_ids_test')
    best_recipe_count = fetched_data.get('best_recipe_count')
    recipe_ids_train = fetched_data.get('recipe_ids_train')
    fav_recipe_ids = fetched_data.get('fav_recipe_ids')

    run_tests(fav_recipe_ids, 10, user_likes,
    tf_data, idf_data, ingredients_count, data_count, use_random_likes, data)

    user_profile = generate_user_profile(tf_data, ingredients_count, user_likes)
    user_pref = generate_user_pref(data_count, tf_data, idf_data, user_profile)
    best_user_pref = max(user_pref)
    best_recipe_pref_index = user_pref.index(best_user_pref)


    formated_tf_data = {index : item for index, item in enumerate(tf_data)}


if __name__ == "__main__":
        # enables the unicode console encoding on Windows
    if sys.platform == "win32":
        enable_win_unicode_console()
    main()
