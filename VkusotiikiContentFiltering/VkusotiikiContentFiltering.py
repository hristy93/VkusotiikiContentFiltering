# -*- coding: utf-8 -*-
import json
import sys
import math
import difflib
import time
from heapq import nlargest
import numpy as np
from itertools import compress
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
#from sklearn import datasets, preprocessing
#from k_nearest_neighbours import *
import random


"""
Instructions:
 - for first use - install (in this order) numpy, scipy, sklearn

Algorithm:
 - Tf-idf with binary data + VSM (Vector Space Model)

Testing:
 - kNN
 - kMeans
 - Naive Bayes

Functionalities:
 - finding the closest recipes to a given recipe by index
 - finding the best prefered recipes by user likes
"""

""" Enables the unicode console for windows """
def enable_win_unicode_console():
    try:
        # Fix UTF8 output issues on Windows console.
        # Does nothing if package is not installed
        from win_unicode_console import enable
        enable()
    except ImportError:
        pass


""" Reads the json data and saves it in data """
def read_json(json_file_name):
    data = []
    with open(json_file_name, 'r', encoding="utf-8") as json_data:
        data = json.load(json_data)
    return data


""" Generates the users' favourite recipes with random data - likes for some recipes """
def generate_user_likes(data_count, binary_propabilities):
    #user_likes = []
    #for _ in range(data_count):
    #    user_likes.append(random.randint(0, 1))
    #return user_likes
    user_likes = np.random.choice([0, 1], size=(data_count,), p=binary_propabilities)
    print("user_likes count:", len(user_likes))
    print("{} recipes liked from {}".format(sum(user_likes), data_count))
    return user_likes


""" Generates the users' favourite recipes with particular recipe ids """
def generate_user_likes_by_recipes_ids(data_count, fav_recipe_ids):
    user_likes = [0] * data_count

    for recipe_id in fav_recipe_ids:
        user_likes[recipe_id] = 1

    print("user_likes count:", len(user_likes))
    print("{} recipes liked from {}".format(sum(user_likes), data_count))
    return user_likes


""" Gets the tf data from the json data """
def get_tf_data(ingredient_data, data):
    tf_data = []
    for ingredient_inner_data in ingredient_data.values():
        tf_inner_data = []
        count = sum(ingredient_inner_data)
        for item in ingredient_inner_data:
            if item == 0:
                tf_inner_data.append(0)
            else:
                tf_value = item / math.sqrt(count)
                #tf_inner_data.append(math.log10(tf_value) + 1) # old way
                #tf_inner_data.append(math.log(tf_value) + 1) # new way
                tf_inner_data.append(tf_value)
        tf_data.append(tf_inner_data)
    print("tf_data count: ", str(len(tf_data)))
    return tf_data


""" Gets the idf data from the json data """
def get_idf_data(ingredients_count_info, data_count):
    idf_data = []
    max_count = max(ingredients_count_info.values())
    for item in ingredients_count_info.values():
        if item == 0:
            idf_data.append(item)
        else:
            #idf_data.append(math.log((data_count) / (item + 1))) # old way
            #idf_data.append(math.log(max_count) / (item + 1)) # new way
            idf_data.append(math.log((data_count) / (item + 1)))
    print("idf_data count: ", str(len(idf_data)))
    return idf_data


""" Gets the tf-idf data from the tf_data and idf_data """
def get_tfidf_data(tf_data, idf_data, data_count):
    tfidf_data = list()
    for i in range(data_count):
        tf_list = tf_data[i]
        tfidf_data.append(np.dot(tf_list, idf_data))
    #print(tfidf_data)
    return tfidf_data


""" Gets the ingreients from the json data """
def get_ingredients(data):
    ingredients = set()
    for item in data:
        for ingredient in item['ingredients']:
            if ingredient['name'] != '':
                ingredients.add(ingredient['name']) 
    print("ingredients count: ", str(len(ingredients)))
    return ingredients


""" Processes the json data and gets the ingredients_count_info (the ingredinets count in all recipes)
    and ingredient data
"""
def process_data(data, ingredients, ingredient_data, ingredients_count_info, meaty_ingredients):
    index = 0
    ingredient_inner_data_counter = 0
    ingredients_count_info_counter = 0
    print("Processing recipes data ...")
    #start_time = time.perf_counter()
    for recipe in data:
        #if index == 30:
        #    print("ok")
        ingredient_inner_data_counter = 0
        ingredients_count_info_counter = 0
        error_indexes = []
        #print(recipe['name'])
        ingredient_inner_data = dict()
        for ingredient in ingredients:
            #if ingredient_inner_data_counter in error_indexes:
            #    print("lets see")
            is_found = False
            #print( "<-----")
            for item in recipe['ingredients']:
                if ingredient == item['name']:
                    is_found = True
                    break
            if is_found:
                to_add = True
                #meaty_ingredients_index = [key for key, value in meaty_ingredients.items() if ingredient in value]
                #if len(meaty_ingredients_index) != 0:
                #    meaty_key = meaty_ingredients_index[0]
                #    meaty_value = ingredient
                for meaty_key, meaty_value in meaty_ingredients.items():
                    if ingredient in meaty_value:
                        meaty_key_index = ingredients.index(meaty_key)
                        error_indexes.append(meaty_key_index)
                        ingredient_inner_data[meaty_key_index] = 1
                        #ingredient_inner_data_counter += 1
                        #print("ingredient_inner_data_counter: ", ingredient_inner_data_counter)
                        modify_ingredient_count_info(ingredients_count_info, ingredients, meaty_key)
                        ingredients_count_info_counter += 1
                        #print("ingredients_count_info_counter: ", ingredients_count_info_counter)
                        #if meaty_key in meaty_value:
                        if meaty_key == ingredient:
                            to_add = False
                            ingredient_inner_data_counter += 1
                            break


                    #if ingredient == meaty_key:
                    #    meaty_index = ingredients.index(meaty_key)
                
                if to_add:
                    ingredient_inner_data[ingredient_inner_data_counter] = 1
                    ingredient_inner_data_counter += 1
                    #print("ingredient_inner_data_counter: ", ingredient_inner_data_counter)
                    modify_ingredient_count_info(ingredients_count_info, ingredients, ingredient)
                    ingredients_count_info_counter += 1
                    #print("ingredients_count_info_counter: ", ingredients_count_info_counter)
            else:
                #if ingredient not in meaty_ingredients.keys():
                if ingredient_inner_data_counter not in error_indexes:
                    ingredient_inner_data[ingredient_inner_data_counter] = 0
                ingredient_inner_data_counter += 1
                #print("ingredient_inner_data_counter: ", ingredient_inner_data_counter)
            #print( "----->")
            #print(ingredients.index(ingredient))
        #print(str(len(ingredient_inner_data)))
        ingredient_data[index] = ingredient_inner_data.values()
        ingredient_inner_data_count = len(ingredient_inner_data.values())
        ingredients_count = len(ingredients)
        if ingredient_inner_data_count != ingredients_count:
            print("error")
    
        index += 1
        #print(data.index(recipe))
        #ingredient_id[data.index(recipe)] = ingredient_data

    #end_time = time.perf_counter()
    #print("Recipes data processed for {} seconds".format(time.strftime("%H.%M.%S", time.gmtime(time.time() - start_time))))
    print("ingredient_data count: ", str(len(ingredient_data.values())))
    print("ingredients_count_info count: ", str(len(ingredients_count_info)))


    # the old implementation of the process_data function
    #def process_data(data, ingredients, ingredient_data, ingredients_count_info):
    #index = 0
    #for recipe in data:
    #    #print(recipe['name'])
    #    ingredient_inner_data = []
    #    for ingredient in ingredients:
    #        is_found = False
    #        for item in recipe['ingredients']:
    #            if ingredient == item['name']:
    #                is_found = True
    #                break
    #        if is_found:
    #            ingredient_inner_data.append(1)
    #            if ingredients.index(ingredient) in ingredients_count_info.keys():
    #                ingredients_count_info[ingredients.index(ingredient)] += 1
    #            else:
    #                ingredients_count_info[ingredients.index(ingredient)] = 0
    #        else:
    #            ingredient_inner_data.append(0)
    #        #print(ingredients.index(ingredient))
    #    #print(str(len(ingredient_inner_data)))
    #    ingredient_data[index] = ingredient_inner_data
    #    index += 1
    #    #print(data.index(recipe))
    #    #ingredient_id[data.index(recipe)] = ingredient_data

    #print("ingredient_data count: " + str(len(ingredient_data.values())))


""" Modifies the ingredient_count_info and counts how many times there 
    is an ingredient in the recipes
"""
def modify_ingredient_count_info(ingredients_count_info, ingredients, ingredient):
    ingredient_index = ingredients.index(ingredient)
    if ingredient_index in ingredients_count_info.keys():
        ingredients_count_info[ingredient_index] += 1
    else:
        ingredients_count_info[ingredient_index] = 0
    

""" Generates the user profile using a dot product of tf value for
    each ingredient and the user's likes (favourite recipes)
"""
def generate_user_profile(tf_data, ingredients_count, user_likes):
    user_profile = []
    for i in range(ingredients_count):
        tf_list = []
        for item in tf_data:
            #print(str(i))
            tf_list.append(item[i])
        #print(str(len(tf_list)))
        user_profile_value = np.dot(user_likes, tf_list)
        user_profile.append(user_profile_value)
    print("user_profile count: ", str(len(user_profile)))
    return user_profile


""" Generates the user preferences for the recipe using a dot product
    of user profile data, idf data and tf data
"""
def generate_user_pref(data_count, tf_data, idf_data, user_profile):
    user_pref = []
    for i in range(data_count):
        tf_list = tf_data[i]
        user_pref_value = round(sum(p * q * t for p, q, t in zip(user_profile, idf_data, tf_list)), 2)
        user_pref.append(user_pref_value)
    #print("user_pref: " + str(len(user_pref)))
    #print("user_pref_data count: " + str(len(user_pref_data)))
    return user_pref


""" Gets the n largest user preferences """
def get_n_largest_user_pref(best_user_pref_count, best_recipe_pref_index, user_pref, user_likes):
    # how to add them in the user_pref sorted - heap ?
    max_user_pref = user_pref[best_recipe_pref_index]
    modified_user_pref = {index : float("{0:.2f}".format(round(item / max_user_pref, 2))) for index, item in enumerate(user_pref) if user_likes[user_pref.index(item)] == 0}
    n_largest_user_pref_indexes = nlargest(best_user_pref_count, modified_user_pref, key=modified_user_pref.get)
    n_largest_user_pref = {index : modified_user_pref[index] for index in n_largest_user_pref_indexes}
    print("The {} largest user pref for the unliked recipes for a user:".format(best_user_pref_count))
    print("n_largest_user_pref_indexes: ", str(n_largest_user_pref_indexes))
    print("n_largest_user_pref: ", n_largest_user_pref)
    return (n_largest_user_pref_indexes, n_largest_user_pref)


""" Filters the data by a recipe_category_id """
def filter_data_by_category(recipe_category_id, data):
    filtered_data = []
    for recipe in data:
        if recipe_category_id == recipe['category']:
            filtered_data.append(recipe)
    return filtered_data


""" Gets the n closest recipes to the best prefered recipe with a given best_recipe_pref_index """
def n_closest_recipes_to_best_recipe_pref(best_recipe_count, tf_data, user_likes, best_recipe_pref_index):
    recipes_diff = dict()
    n_closest_recipe_indexes = []
    best_recipe_pref_tf_values = tf_data[best_recipe_pref_index]
    index = 0
    for tf_values in tf_data:
        if tf_values != best_recipe_pref_tf_values:
            recipes_diff[index] = float("{0:.2f}".format(round(np.dot(tf_values, best_recipe_pref_tf_values), 2)))
            index += 1
    recipe_diff_modified = {index : item for index, item in recipes_diff.items() if user_likes[index] == 0}
    n_closest_recipe_indexes = nlargest(best_recipe_count, recipe_diff_modified, key=recipe_diff_modified.get)
    n_closest_recipes = {index : recipes_diff[index] for index in n_closest_recipe_indexes}
    #print("recipes_diff: " + str(recipes_diff))
    print("The closest recipes to the recipe with the largest user pref using VSM (Vector Space Model):")
    print("n_closest_recipes: ", str(n_closest_recipes))
    print("n_closest_recipe_indexes: ", str(n_closest_recipe_indexes))
    return n_closest_recipes, n_closest_recipe_indexes


""" Gets the k closest recipes to the best prefered recipe using kNN algorithm """
def k_closest_recipes_with_knn(tf_data, filtered_tf_data, user_likes, k, best_recipe_pref_index):
    print("\nData from k closest recipes to the best prefered recipe using kNN with {} neighbours".format(k))

    knn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')
    #TODO - find if pop is the best way to get all items but one with a specific index
    best_recipe_pref = tf_data.pop(best_recipe_pref_index)
    filtered_tf_data_keys = list(filtered_tf_data.keys())
    if best_recipe_pref_index in filtered_tf_data_keys:
        filtered_tf_data.pop(best_recipe_pref_index)

    arr = np.array(list(filtered_tf_data.values()))
    nbrs = knn.fit(arr)

    if best_recipe_pref_index in filtered_tf_data_keys:
        filtered_tf_data[best_recipe_pref_index] = best_recipe_pref
    tf_data.append(best_recipe_pref)

    distances, indices = nbrs.kneighbors((best_recipe_pref,), k)

    modified_indices = [filtered_tf_data_keys[item] for item in indices[0]]

    print("distances\n", str(distances[0]))
    print("indices\n", str(modified_indices))
    return modified_indices


""" Gets the closest recipes to the best prefered recipe using kMeans algorithm """
def closest_recipes_with_kmeans(tf_data, filtered_tf_data, k, best_recipe_pref_index):
    print("\nData from closest recipes to the best prefered recipe using kMeans with {} clusteres".format(k))

    arr = np.array(list(filtered_tf_data.values()))
    filtered_tf_data_keys = list(filtered_tf_data.keys())

    kmeans = KMeans(n_clusters=k)
    clustered_data = kmeans.fit(arr)
    distance_data = kmeans.transform(arr)
    predicted_cluster = kmeans.predict((tf_data[best_recipe_pref_index],))[0]

    print("clustered_data count: ", str(len(clustered_data.labels_)))
    print("distance_data count: ", str(len(distance_data)))
    print("predicted_cluster: ", str(predicted_cluster))

    modified_indices = [filtered_tf_data_keys[index] for index, item in enumerate(clustered_data.labels_)
                        if item == predicted_cluster]

    #index = 0
    #closest_recipes = {}
    #for item in clustered_data.labels_:
    #    if item == predicted_cluster:
    #        modified_index = filtered_tf_data_keys[index]
    #        closest_recipes[modified_index] = distance_data[index]
    #    index += 1

    print("{} recipes in the same cluster as the best prefered recipe: ".format(len(modified_indices)))
    return modified_indices


""" Gets presumably liked recipes by the user using Naive Bayes algorithm """
def presumably_liked_recipes_with_naive_bayes(tf_data, recipe_ids_train, recipe_ids_test, user_likes):
    print("\nData from the most presumably liked recipes by the user using Naive Bayes")
    gnb = GaussianNB()
    #y_pred = gnb.fit(tf_data, user_likes).predict(list(filtered_tf_data.values()))
    y_pred = gnb.fit([tf_data[item] for item in recipe_ids_train], [user_likes[item] for item in recipe_ids_train]).predict([tf_data[item] for item in recipe_ids_test])
    #y_pred = gnb.fit(tf_data, user_likes).predict([tf_data[item] for item in recipe_ids_test])
    print("y_pred: ", y_pred)

    #filtered_tf_data_keys = list(filtered_tf_data.keys())
    #modified_indices = [filtered_tf_data_keys[index] for index, item in enumerate(y_pred) if item == 1]

    #modified_indices = [recipe_ids_test[index] for index, item in enumerate(y_pred) if user_likes[item] == 1]

    modified_indices = [recipe_ids_test[index] for index, item in enumerate(y_pred)]

    return modified_indices

""" Gets the n closest users to an user with a given user_pref """
def n_closest_users_to_user_pref(best_user_pref_count, user_pref_data, user_pref):
    users_diff = dict()
    n_closest_user_indexes = []
    index = 0
    for user_pref_values in user_pref_data:
        if user_pref_values != user_pref:
            users_diff[index] = np.dot(user_pref_values, user_pref)
            index += 1
    n_closest_user_indexes = nlargest(best_user_pref_count, users_diff, key=users_diff.get)
    n_closest_users = {index : users_diff[index] for index in n_closest_user_indexes}
    print("The closest users to the user with the largest user pref:")
    print("n_closest_users: ", str(n_closest_users))
    print("n_closest_user_indexes: ", str(n_closest_user_indexes))
    return n_closest_user_indexes
 
   
""" Tests the kNN methods of sklearn module """
def test_knn(tf_data, user_likes, recipe_ids_train, recipe_ids_test, max_k, best_recipe_pref_index):
    knn_average_accuracy_scores = {}
    k_fold_count = 10
    for k in range(1, max_k):
        #print("\ndata from kNN with {} neighbours".format(k))
        kf = KFold(n_splits=k_fold_count)
        knn_accuracy_scores = []
        for train_index, test_index in kf.split(tf_data, user_likes):
            X_train = [tf_data[index] for index in train_index]
            X_test = [tf_data[index] for index in test_index]
            y_train = [user_likes[index] for index in train_index]
            y_test = [user_likes[index] for index in test_index]

            knn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')
            nbrs = knn.fit(X_train, y_train)
            distances, results = nbrs.kneighbors(X_test, k)
            modified_distances = [item for item in distances]
            #print("kNN results: ", results)
            #print("kNN results count: ", len(results))

            y_pred = []
            index = 0
            for result in results:
                modified_result = [user_likes[item] for item in result]
                ones = sum(modified_result)
                zeros = k - ones
                if ones > zeros:
                    y_pred.append(1)
                elif ones < zeros:
                    y_pred.append(0)
                else:
                    dist = distances[index]
                    index = np.argmin(dist)
                    y_pred.append(user_likes[index])
                index += 1
            score = accuracy_score(y_test, y_pred)
            knn_accuracy_scores.append(score)
            #print("y_pred: ", str(y_pred))
            #print("y_pred count: ", str(len(y_pred)))
            #print("accuracy_score: ", score)
            #print("classification_report: ", classification_report(y_test, y_pred))
        knn_average_accuracy_score = sum(knn_accuracy_scores) / k_fold_count
        knn_average_accuracy_scores[k] = knn_average_accuracy_score
        #print("average accuracy score for k = {} is : {}".format(k,  knn_average_accuracy_score))
    #print(knn_average_accuracy_scores)
    best_k = [key for key, value in knn_average_accuracy_scores.items() 
                                    if value == max(knn_average_accuracy_scores.values())][0]
    print("The best value for k is: ", best_k)
    return best_k


""" Tests the k-Means methods of sklearn module with k-fold crossvalidation """
def test_kmeans_with_kfold_crossvalidation(tf_data, filtered_tf_data, user_likes, max_k, best_recipe_pref_index):
    print("\nData from k-Means test with max {} clusters".format(max_k))
    k_fold_count = 10
    kf = KFold(n_splits=k_fold_count)
    kmeans_average_accuracy_scores = []
    for k in range(1, max_k):
        #print("\ndata from kMeans with {} neighbours".format(k))
        kmeans_accuracy_scores = []
        kmeans = KMeans(n_clusters=k)
        for train_index, test_index in kf.split(tf_data, user_likes):
            X_train = [tf_data[index] for index in train_index]
            X_test = [tf_data[index] for index in test_index]
            y_train = [user_likes[index] for index in train_index]
            y_test = [user_likes[index] for index in test_index]

            clustered_data = kmeans.fit(X_train, y_train)
            distance_data = kmeans.transform(X_train)
            predicted_clusters = kmeans.predict(X_test)

            #print("clustered_data count: ", str(len(clustered_data.labels_)))
            #print("distance_data count: ", str(len(distance_data)))
            #print("predicted_cluster: ", str(predicted_clusters))
            #print("{} recipes in the same cluster as the best prefered recipe: ".format(k))

            #filtered_tf_data_keys = list(filtered_tf_data.keys())
            #modified_indices = [filtered_tf_data_keys[item] for item in indices[0]]

            closest_recipes = {}
            for cluster_id in predicted_clusters:
                index = 0
                for item in clustered_data.labels_:
                    if item == cluster_id:
                        closest_recipes[index] = distance_data[index]
                    index += 1

            #filtered_tf_data_keys = list(filtered_tf_data.keys())
            #y_pred = [filtered_tf_data_keys[index] for index, item in enumerate(clustered_data.labels_)
            #                    if item == predicted_cluster[index]]

            score = accuracy_score(y_test, y_pred)
            kmeans_accuracy_scores.append(score)
            #print("y_pred: ", str(y_pred))
            #print("y_pred count: ", str(len(y_pred)))
            #print("accuracy_score: ", score)
            #print("classification_report: ", classification_report(y_test, y_pred))
        kmeans_average_accuracy_score = sum(kmeans_accuracy_scores) / k_fold_count
        kmeans_average_accuracy_scores[k] = kmeans_average_accuracy_scores

    print("The best value for k is: ", [key for key, value in kmeans_average_accuracy_scores.items() 
                                    if value == max(kmeans_average_accuracy_scores.values())][0])    

    #closest_recipes = [item for item in clustered_data if item == predicted_cluster]
    #print("kmeans\n" + str(kmeans))
    #print("cluster_centers_\n" + str(kmeans.cluster_centers_))


""" Tests the k-Means methods of sklearn module with variance of the total 
    between-cluster sum of squares
"""
def test_kmeans_with_variance(tf_data, filtered_tf_data, user_likes, max_k, best_recipe_pref_index):
    k_range = range(1, max_k * 2)
    k_means_data = list(filtered_tf_data.values())
    k_means_var = [KMeans(k).fit(k_means_data) for k in k_range]
    centroids = [var.cluster_centers_ for var in k_means_var]
    k_euclid = [cdist(k_means_data, centroid, 'euclidean') for centroid in centroids]
    distances = [np.min(item, axis=1) for item in k_euclid]

    # total within-cluster sum of squares
    wcss = [sum(distance ** 2) for distance in distances]
    # total sum of squares
    tss = sum(pdist(k_means_data) ** 2) / len(k_means_data)
    # total between-cluster sum of squares
    bss = tss - wcss
    print(bss)

    best_k = max_k - 1
    variance = dict()
    for index in range(0, len(bss) - 1):
        variance[index] = bss[index + 1] - bss[index]
        if variance[index] <= 0.5:
            best_k = index + 1
            break
    print(variance)

    plt.xlabel("k")
    plt.ylabel("variance")
    plt.title("K - Variance Graph")
    plt.scatter(range(1, len(variance) + 1), list(variance.values()))
    plt.show()

    print("The best value for k is: ", best_k)
    return best_k


""" Tests the Naive Bayes method of sklearn module """
def test_naive_bayes(tf_data, user_likes, best_user_pref_count, best_recipe_pref_index):
    print("\nData from Naive Bayes")
    k_fold_count = 10
    gnb = GaussianNB()
    X_train, X_test, y_train, y_test = train_test_split(tf_data, user_likes)
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    print("y_pred: ", y_pred)
    print("accuracy_score: ", accuracy_score(y_test, y_pred))
    print("classification_report:\n", classification_report(y_test, y_pred))

    scores = cross_val_score(gnb, tf_data, user_likes, cv=k_fold_count)
    print("{}-fold cross validation scores: {}".format(k_fold_count, scores))


""" Finds the closest ingredients to be refactored and trimmed """
def find_closest_ingredients(ingredients):
    pass
    #closest_ingredients = dict()
    #for item in ingredients:
    #    closest_ingredients[item] = difflib.get_close_matches(item, ingredients)
    #    print(ingredients.index(item), item, closest_ingredients[item])
    #print(closest_ingredients)

""" Finds ingredients that contain some words """
def find_meaty_ingredients(ingredients, meat_food, fish_food):
    meaty_ingredients = dict()
    print("meat_food count: " + str(len(meat_food)))
    print("fish_food count: " + str(len(fish_food)))

    for item in ingredients:
        for meat in meat_food:
            if item.find(meat) != -1:
                if meat in meaty_ingredients.keys():
                    meaty_ingredients[meat].append(item)
                else:
                    meaty_ingredients[meat] = [item]

        for fish in fish_food:
            if item.find(fish) != -1:
                if fish in meaty_ingredients.keys():
                    meaty_ingredients["риба"].append(item)
                else:
                    meaty_ingredients["риба"] = [item]
                #print(meaty_ingredients.index(meat), item, meaty_ingredients[meat])'
    #for item in meaty_ingredients:
    #    print(item, meaty_ingredients[item])

    return meaty_ingredients


""" Adds the meaty food as ingredients """
def add_food_groups_as_ingredients(meaty_ingredients, ingredients):
    ingredients.extend([key for key in meaty_ingredients.keys() if key not in ingredients])
    print("new ingredients count: ", str(len(ingredients)))
    print("meaty_ingredients count: ", str(len(meaty_ingredients)))


""" Tests the tf-idf transformer onto the ingredients data """     
def tfidf_transform(ingredient_data):
    transformer = TfidfTransformer(smooth_idf=True)
    ingredient_data = [ v for v in ingredient_data.values() ]
    tfidf = transformer.fit_transform(ingredient_data)
    tfidf_result = tfidf.toarray()
    #print(str(tfidf_result))
    #print(len(tfidf_result)) 


""" Gets the recipe ids filtered by some kind/group of food  """
def get_favourite_recipe_ids(data, group_food):
    fav_recipe_ids = dict()
    for recipe in data:
        is_found = False
        for item in recipe["ingredients"]:
            for dish in group_food:
                if item['name'].find(dish) != -1:
                    fav_recipe_ids[data.index(recipe)] = recipe["name"]
                    is_found = True
                    break
            if is_found:
                break

    return fav_recipe_ids


""" Prints the names of the recipes by given indexes """
def print_recipes_info(data, indexes, percents = None):
    if hasattr(indexes, "__iter__"):
        if percents is None:
            to_print = [(index, data[index]["name"]) for index in indexes]
        else:
            to_print = [(index, percents[index], data[index]["name"]) for index in indexes]
        for item in to_print:
            print(item)
    else:
        print((indexes, data[indexes]["name"]))


""" Prints the accuracy of some predicted values and test values """
def print_accuracy_with_test_data(predicted_values, test_values, user_likes):
    predicted_values_count = len(predicted_values)
    result = list(map(lambda x: x in test_values, predicted_values))
    correct = [value for value, is_correct in zip(predicted_values, result) if is_correct]
    incorrect = [value for value, is_correct in zip(predicted_values, result) if not is_correct]
    #str(sum([1 for index in predicted_values if index in test_values]) /
          #predicted_values_count))
    print("accuracy with test data: ", float("{0:.2f}".format(round(len(correct) / predicted_values_count, 2))))
    print("correct: ", correct)
    print("correct_likes: ", [user_likes[index] for index in correct])
    print("incorrect: ", incorrect)
    print("incorrect_likes: ", [user_likes[index] for index in incorrect])

""" Gets the user input for the type of likes (random wih some propability or some group of food) """
def get_user_input():
    use_radom_likes = input("Use random likes with some propability: (y) or (n) \n")
    if use_radom_likes == "y":
        propability_of_one = input("The propability of like for a recipe is (in 0.## format): \n")
        return float(propability_of_one)
    elif use_radom_likes == "n":
        food_group = input("Like a group of recipes with: meat (1), fish (2) or vegetarian products (3) \n")
        return int(food_group)
    else:
        raise ValueError("Not correct input!")


def prepare_data():
    # defines some variables and constants
    json_file_name = "recipes_500_refined_edited.json"
    propability_of_one = 0.8
    users_count = 20
    best_user_pref_count = 5
    best_recipe_count = 5
    use_random_likes = False
    use_user_input = True

    meat_food = { "кайма", "телешк", "овч", "агнешк", "свинск", "суджук", "филе", "заеш", "месо", "кайма", "кренвирш", "кюфте", "говежд", "скарид", "овнешк", "пиле", "пуйка", "патешк", "надениц", "пушен", "роле", "шпек", "колбас", "еленск", "шунка", "гъши", "гъск", "бекон", "агнешк", "кървавиц", "салам", "пастърма" };
    fish_food = { "риба", "скумрия", "шаран", "рибн" , "рибен", "сьомга", "пъстърв", "ципур", "щука", "риба тон", "треска", "барабун", "султанк", "атерина", "сребърка", "зарган", "костур", "калкан", "карагьоз", "каракуда", "лаврак", "лефер", "моруна", "попче", "паламуд", "писия", "сафрид", "сардин", "цаца", "хамсия" }
    fav_fish_recipe_ids = [448, 385, 5, 454, 455, 8, 330, 141, 462, 208, 211, 212, 446, 24, 476, 222, 223, 164, 39, 40, 169, 426, 487, 174, 368, 498, 190, 184, 468, 378, 382]
    fav_meat_recipe_ids = [0, 6, 10, 12, 14, 15, 18, 21, 22, 24, 27, 28, 29, 30, 31, 32, 34, 36, 37, 38, 43, 51, 54, 58, 62, 66, 69, 72, 73, 75, 76, 77, 78, 79, 82, 83, 84, 87, 91, 95, 98, 99, 101, 102, 103, 104, 106, 118, 119, 121, 123, 124, 127, 128, 130, 131, 132, 133, 134, 139, 142, 143, 146, 150, 151, 152, 157, 158, 160, 162, 163, 165, 166, 167, 172, 175, 180, 181, 184, 188, 189, 190, 192, 193, 195, 199, 201, 204, 207, 208, 209, 213, 214, 216, 219, 223, 224, 231, 237, 239, 250, 251, 252, 253, 254, 255, 256, 258, 261, 262, 263, 264, 266, 269, 274, 275, 288, 291, 295, 296, 297, 298, 302, 307, 308, 309, 312, 313, 314, 315, 316, 318, 319, 321, 324, 325, 326, 328, 329, 331, 332, 337, 339, 347, 348, 349, 350, 351, 352, 354, 364, 372, 376, 378, 379, 380, 381, 384, 387, 388, 389, 395, 399, 406, 410, 411, 412, 414, 415, 419, 422, 432, 438, 439, 440, 441, 442, 444, 449, 453, 455, 456, 458, 459, 460, 462, 464, 465, 466, 467, 470, 471, 472, 473, 474, 475, 476, 477, 479, 480, 483, 487, 494, 495, 497, 498]
    fav_vegetarian_recipe_ids = [1, 2, 3, 4, 7, 9, 11, 13, 16, 17, 19, 20, 23, 25, 26, 33, 35, 41, 42, 44, 45, 46, 47, 48, 49, 50, 52, 53, 55, 56, 57, 59, 60, 61, 63, 64, 65, 67, 68, 70, 71, 74, 80, 81, 85, 86, 88, 89, 90, 92, 93, 94, 96, 97, 100, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 120, 122, 125, 126, 129, 135, 136, 137, 138, 140, 144, 145, 147, 148, 149, 153, 154, 155, 156, 159, 161, 168, 170, 171, 173, 176, 177, 178, 179, 182, 183, 185, 186, 187, 191, 194, 196, 197, 198, 200, 202, 203, 205, 206, 210, 215, 217, 218, 220, 221, 225, 226, 227, 228, 229, 230, 232, 233, 234, 235, 236, 238, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 257, 259, 260, 265, 267, 268, 270, 271, 272, 273, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 289, 290, 292, 293, 294, 299, 300, 301, 303, 304, 305, 306, 310, 311, 317, 320, 322, 323, 327, 333, 334, 335, 336, 338, 340, 341, 342, 343, 344, 345, 346, 353, 355, 356, 357, 358, 359, 360, 361, 362, 363, 365, 366, 367, 369, 370, 371, 373, 374, 375, 377, 383, 386, 390, 391, 392, 393, 394, 396, 397, 398, 400, 401, 402, 403, 404, 405, 407, 408, 409, 413, 416, 417, 418, 420, 421, 423, 424, 425, 427, 428, 429, 430, 431, 433, 434, 435, 436, 437, 443, 445, 447, 450, 451, 452, 457, 461, 463, 469, 478, 481, 482, 484, 485, 486, 488, 489, 490, 491, 492, 493, 496, 499]

    # reads the recipes data from the json file
    data = read_json(json_file_name)

    # for recipe filtering by category id
    # recipe_category_id = 1
    # pe_category = ""
    # filter_data_by_category(recipe_category_id, data)

    # defines more variables
    data_count = len(data)
    print("data count: " + str(data_count))
    ingredient_data = dict()
    ingredients_count_info = dict()
    ingredients = set()
    ingredients = get_ingredients(data)
    ingredients = list(ingredients)
    ingredients_count = len(ingredients)
    n_closest_recipes = []
    n_closest_users = []
    fav_recipe_ids = fav_meat_recipe_ids
    recipe_ids_train = []
    recipe_ids_test = []

    # processes the user input
    if use_user_input:
        user_input = get_user_input()
        if isinstance(user_input, int):
            if user_input == 1:
                fav_recipe_ids = fav_meat_recipe_ids
                print("You chose to like meat food\n")
            elif user_input == 2:
                fav_recipe_ids = fav_fish_recipe_ids
                print("You chose to like fish food\n")
            elif user_input == 3:
                fav_recipe_ids = fav_vegetarian_recipe_ids
                print("You chose to like vegetarian food\n")
            else:
                raise ValueError("The input is not correct!")
        else:
            propability_of_one = user_input
            use_random_likes = True

    else:
        if use_random_likes:
            print("The user will use random likes with {} propability of like for a recipe".format(propability_of_one))
        else:
            print("The user will only like a group of recipes with meat")

    binary_propabilities = [1 - propability_of_one, propability_of_one]
    
    # finds the ingredients with some kind of meat or fish, groups them into 
    # new ingredients that are added to the other ones
    meaty_ingredients = find_meaty_ingredients(ingredients, meat_food, fish_food)
    add_food_groups_as_ingredients(meaty_ingredients, ingredients)
    ingredients_count = len(ingredients)

    # processes the whole initial data from the json and gets the necessary data
    process_data(data, ingredients, ingredient_data, ingredients_count_info, meaty_ingredients)

    # finds the closest ingredients to eachother
    #find_closest_ingredients(ingredients)
    
    # generates the tf, idf and tf-idf data
    idf_data = get_idf_data(ingredients_count_info, data_count)
    tf_data = get_tf_data(ingredient_data, data)
    tfidf_data = get_tfidf_data(tf_data, idf_data, data_count)
    #tfidf_transform(ingredient_data)

    # gerenates the indexes of the recipes that contain some ingredients
    #fav_meat_recipe_ids = get_favourite_recipe_ids(data, meat_food)
    #fav_fish_recipe_ids = get_favourite_recipe_ids(data, fish_food)
    #print("fav_meat_recipe_ids: ", fav_meat_recipe_ids.keys())
    #print(len(fav_meat_recipe_ids))
    #print("fav_fish_recipe_ids: ", fav_fish_recipe_ids.keys())
    #print(len(fav_fish_recipe_ids))

    #fav_vegetarian_recipe_ids = [id for id in range(0, data_count) if id not in fav_meat_recipe_ids.keys()
    #                             and id not in fav_fish_recipe_ids.keys()]
    #print("fav_vegetarian_recipe_ids: ", fav_vegetarian_recipe_ids)
    #print(len(fav_vegetarian_recipe_ids))

    # genrerates user likes on random or by some group of foods
    recipe_ids_train, recipe_ids_test = [], []
    if use_random_likes:
        user_likes = generate_user_likes(data_count, binary_propabilities)
    else:
        # old implementation
        recipe_ids_train, recipe_ids_test = train_test_split(fav_recipe_ids)
        user_likes = generate_user_likes_by_recipes_ids(data_count, recipe_ids_train)

        #new implementation
        #user_likes = generate_user_likes_by_recipes_ids(data_count, fav_recipe_ids)

        # split the tf_data and user_likes into training and testing data
        #tf_data_train, tf_data_test, likes_train, likes_test = train_test_split(tf_data, user_likes)

    return {
        'data': data,
        'ingredients': ingredients,
        'ingredients_count': ingredients_count,
        'data_count': data_count,
        'tf_data': tf_data,
        'idf_data': idf_data,
        'user_likes': user_likes,
        'best_user_pref_count': best_user_pref_count,
        'use_random_likes': use_random_likes,
        'recipe_ids_test': recipe_ids_test,
        'best_recipe_count': best_recipe_count,
        'recipe_ids_train': recipe_ids_train,
    }


def get_recipes_names(data):
    return [recipe.get('name') for recipe in data]


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

    # creates the user profile and his/her recipe preference and gets the most suitable recipe
    user_profile = generate_user_profile(tf_data, ingredients_count, user_likes)
    user_pref = generate_user_pref(data_count, tf_data, idf_data, user_profile)
    best_user_pref = max(user_pref)
    best_recipe_pref_index = user_pref.index(best_user_pref)

    # prints the best recipe for the user
    print("\n")
    print("best recipe for user: ")
    print_recipes_info(data, best_recipe_pref_index)
    print("\n")

    # gets the n best recipes for the user in descending order
    n_largest_user_pref_indexes, n_largest_user_pref = get_n_largest_user_pref(best_user_pref_count, best_recipe_pref_index, user_pref, user_likes)
    print("{} closest recipes to the best recipe for user: ".format(best_user_pref_count))
    print_recipes_info(data, n_largest_user_pref_indexes, n_largest_user_pref)
    if not use_random_likes:
        print_accuracy_with_test_data(n_largest_user_pref_indexes, recipe_ids_test, user_likes)
    print("\n")

    # gets the n closest recipes to the best uesr's reicpe using VSM (Vector Space Model)
    n_closest_recipes, n_closest_recipe_indexes = n_closest_recipes_to_best_recipe_pref(best_recipe_count, tf_data, user_likes, best_recipe_pref_index)
    print_recipes_info(data, n_closest_recipe_indexes, n_closest_recipes)
    if not use_random_likes:
        print_accuracy_with_test_data(n_closest_recipe_indexes, recipe_ids_test, user_likes)

    # gets the n closest users to some user with user preference
    #n_closest_users = n_closest_users_to_user_pref(best_user_pref_count, user_pref_data, user_pref)

    # filters tf_data that is not liked by the user
    filtered_tf_data = {tf_data.index(item) : item for item in tf_data if user_likes[tf_data.index(item)] == 0}

    # get the k closest recipes to the best uesr's reicpe using kNN
    knn_data_indexes = k_closest_recipes_with_knn(tf_data, filtered_tf_data, user_likes, best_user_pref_count, best_recipe_pref_index)
    print_recipes_info(data, knn_data_indexes)
    if not use_random_likes:
        print_accuracy_with_test_data(knn_data_indexes, recipe_ids_test, user_likes)

    # gets some presumably likes recipes by the user using the Naive Bayes
    if not use_random_likes:
        naive_bayes_recipe_indexes = presumably_liked_recipes_with_naive_bayes(tf_data, recipe_ids_train, recipe_ids_test, user_likes)
        print_accuracy_with_test_data(naive_bayes_recipe_indexes, recipe_ids_test, user_likes)

    k_means_data_indexes = closest_recipes_with_kmeans(tf_data, filtered_tf_data, int(math.sqrt(len(filtered_tf_data))), best_recipe_pref_index)
    print_recipes_info(data, k_means_data_indexes)
    if not use_random_likes: 
        print_accuracy_with_test_data(k_means_data_indexes, recipe_ids_test, user_likes)

    #TODO - add closest recipes by category if everything is ok

    # sklearn tests - kMeans, kNN, Naive Bayes

    # not working
    #test_kmeans_with_kfold_crossvalidation(tf_data, filtered_tf_data, user_likes, int(math.sqrt(len(filtered_tf_data))), best_recipe_pref_index)

    #best_k_for_kmeans = test_kmeans_with_variance(tf_data, filtered_tf_data, user_likes, int(math.sqrt(len(filtered_tf_data))) + 1, best_recipe_pref_index)

    #best_k_for_knn = test_knn(tf_data, user_likes, recipe_ids_train, recipe_ids_test, data_count // 2, best_recipe_pref_index)

    #naive_bayes_data = test_naive_bayes(tf_data, user_likes, best_user_pref_count, best_recipe_pref_index)


if __name__ == "__main__":
        # enables the unicode console encoding on Windows
    if sys.platform == "win32":
        enable_win_unicode_console()
    main()
