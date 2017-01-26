# -*- coding: utf-8 -*-
import json
import sys
import math
import difflib
from heapq import nlargest
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfTransformer
import random


# for first use - install (in this order) numpy, scipy, sklearn !!!
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
def generate_user_likes(data_count):
    #user_likes = []
    #for _ in range(data_count):
    #    user_likes.append(random.randint(0, 1))
    #return user_likes
    user_likes = np.random.choice([0, 1], size=(data_count,), p=[4./5, 1./5])
    return user_likes


""" Gets the tf data from the json data """
def get_tf_data(ingredient_data, ingredients_count, data):
    tf_data = []
    for ingredient_inner_data in ingredient_data.values():
        tf_inner_data = []
        count = sum(ingredient_inner_data)
        for item in ingredient_inner_data:
            if item == 0:
                tf_inner_data.append(0)
            else:
                tf_inner_data.append(math.log10(item / math.sqrt(count)) + 1)
        tf_data.append(tf_inner_data)
    print("tf_data count: " + str(len(tf_data)))
    return tf_data


""" Gets the idf data from the json data """
def get_idf_data(ingredients_count_info, data_count):
    idf_data = []
    for item in ingredients_count_info.values():
        #if item == 0:
        #    idf_data.append(item)
        #else:
            idf_data.append(math.log((data_count) / (item + 1)))
    print("idf_data count: " + str(len(idf_data)))
    return idf_data

""" Gets the tf-idf data from the tf_data and idf_data """
def get_tfidf_data(tf_data, idf_data):
    tfidf_data = list()
    for i in range(data_count):
        tf_list = tf_data[i]
        tfidf_data.append(np.dot(tf_list, idf_data))
    print(tfidf_data)
    return tfidf_data

""" Gets the ingreients from the json data """
def get_ingredients(data):
    for item in data:
        for ingredient in item['ingredients']:
            if ingredient['name'] != '':
                ingredients.add(ingredient['name']) 
    print("ingredients count: " + str(len(ingredients)))
    return ingredients


""" Processes the json data and gets the ingredients_count_info (the ingredinets count in all recipes)
    and ingredient data
"""
def process_data(data, ingredients, ingredient_data, ingredients_count_info):
    index = 0
    for recipe in data:
        #print(recipe['name'])
        ingredient_inner_data = []
        for ingredient in ingredients:
            is_found = False
            for item in recipe['ingredients']:
                if ingredient == item['name']:
                    is_found = True
                    break
            if is_found:
                ingredient_inner_data.append(1)
                if ingredients.index(ingredient) in ingredients_count_info.keys():
                    ingredients_count_info[ingredients.index(ingredient)] += 1
                else:
                    ingredients_count_info[ingredients.index(ingredient)] = 0
            else:
                ingredient_inner_data.append(0)
            #print(ingredients.index(ingredient))
        #print(str(len(ingredient_inner_data)))
        ingredient_data[index] = ingredient_inner_data
        index += 1
        #print(data.index(recipe))
        #ingredient_id[data.index(recipe)] = ingredient_data

    print("ingredient_data count: " + str(len(ingredient_data.values())))


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
    print("user_profile count: " + str(len(user_profile)))
    return user_profile


""" Generates the user preferences for the recipe using a dot product
    of user_profile, idf_data and tf_data
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
def get_n_largest_user_pref(best_user_pref_count, user_pref, user_likes):
    # how to add them in the user_pref sorted - heap ?
    modified_user_pref = {index : item for index, item in enumerate(user_pref) if user_likes[user_pref.index(item)] == 0}
    n_largest_user_pref_indexes = nlargest(best_user_pref_count, modified_user_pref, key=modified_user_pref.get)
    n_largest_user_pref = {index : modified_user_pref[index] for index in n_largest_user_pref_indexes}
    print("The n largest user pref for the unliked recipes for a user:")
    print("n_largest_user_pref_indexes: " + str(n_largest_user_pref_indexes))
    print("n_largest_user_pref: " + str(n_largest_user_pref))
    return (n_largest_user_pref_indexes, n_largest_user_pref)

    ## not working when there is more that one percent (item) with
    ## the same value and it gets the same index
    #n_largest_user_pref = dict()
    ##modified_user_pref = {item : user_pref[index] for index, item in
    ##enumerate(user_pref) if user_likes[user_pref.index(item)] == 0}
    ##sorted_user_pref = sorted(sorted_user_pref.items(), key=lambda x: x[1],
    ##reverse=True)
    ##user_pref.sort(key = None, reverse = True)
    ##print(user_pref)
    #n_largest_count = 0
    #for key, value in sorted_user_pref.items():
    #    user_pref_index = key
    #    if user_likes[user_pref_index] == 0:
    #        n_largest_user_pref[user_pref_index] = item
    #        n_largest_count += 1
    #        if n_largest_count == best_user_pref_count:
    #            break
    #return n_largest_user_pref


""" Filters the data by a recipe_category_id """
def filter_data_by_category(recipe_category_id, data):
    filtered_data = []
    for recipe in data:
        if recipe_category_id == recipe['category']:
            filtered_data.append(recipe)
    return filtered_data


""" Gets the n (best_recipe_count) closest recipes to the best prefered recipe with a given best_recipe_pref_index """
def n_closest_recipes_to_best_recipe_pref(best_recipe_count, tf_data, best_recipe_pref_index):
    recipes_diff = dict()
    n_closest_recipe_indexes = []
    best_recipe_pref_tf_values = tf_data[best_recipe_pref_index]
    index = 0
    for tf_values in tf_data:
        if tf_values != best_recipe_pref_tf_values:
            recipes_diff[index] = np.dot(tf_values, best_recipe_pref_tf_values)
            index += 1
    n_closest_recipe_indexes = nlargest(best_recipe_count, recipes_diff, key=recipes_diff.get)
    n_closest_recipes = {index : recipes_diff[index] for index in n_closest_recipe_indexes}
    #print("recipes_diff: " + str(recipes_diff))
    print("The closest recipes to the recipe with the largest user pref:")
    print("n_closest_recipes: " + str(n_closest_recipes))
    print("n_closest_recipe_indexes: " + str(n_closest_recipe_indexes))
    return n_closest_recipe_indexes


""" Gets the n (best_user_pref_count) closest users to an user with a given user_pref """
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
    print("n_closest_users: " + str(n_closest_users))
    print("n_closest_user_indexes: " + str(n_closest_user_indexes))
    return n_closest_user_indexes
    

""" Tests the kNN methods of sklearn module """
def test_knn(tf_data, k):
    arr = np.array(tf_data)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(arr)
    distances, indices = nbrs.kneighbors(arr)
    print("distances\n" + str(distances))
    print("indices\n" + str(indices))


""" Tests the k Means methods of sklearn module """
def test_kmeans(tf_data, k):
    arr = np.array(tf_data)
    kmeans = KMeans(n_clusters=k).fit(arr)
    print("kmeans\n" + str(kmeans))
    print("labels_\n" + str(kmeans.labels_))
    print("cluster_centers_\n" + str(kmeans.cluster_centers_))


""" Finds the closest ingredients to be refactored and trimmed """
def find_closest_ingredients(ingredients):
    closest_ingredients = dict()
    for item in ingredients:
        closest_ingredients[item] = difflib.get_close_matches(item, ingredients)
        print(item, closest_ingredients[item])
    #print(closest_ingredients)
   
""" Tests the tf-idf transformer onto the ingredients data """     
def tfidf_transform(ingredient_data):
    transformer = TfidfTransformer(smooth_idf=True)
    ingredient_data = [ v for v in ingredient_data.values() ]
    tfidf = transformer.fit_transform(ingredient_data)
    tfidf_result = tfidf.toarray()
    #print(str(tfidf_result))
    #print(len(tfidf_result))

if __name__ == "__main__":
    if sys.platform == "win32":
        enable_win_unicode_console()
    json_file_name = "recipes_500_refined_edited.json"
    data = read_json(json_file_name)
    # for recipe filtering by category id
    #recipe_category_id = 1
    #filter_data_by_category(recipe_category_id, data)
    data_count = len(data)
    users_count = 20
    best_user_pref_count = 5
    best_recipe_count = 5
    recipe_category = ""
    print("data count: " + str(data_count))
    ingredient_data = dict()
    ingredients_count_info = dict()
    ingredients = set()
    ingredients = get_ingredients(data)
    ingredients = list(ingredients)
    ingredients_count = len(ingredients)
    n_closest_recipes = []
    n_closest_users = []
    
    process_data(data, ingredients, ingredient_data, ingredients_count_info)
    #find_closest_ingredients(ingredients)
    
    idf_data = get_idf_data(ingredients_count_info, data_count)
    tf_data = get_tf_data(ingredient_data, ingredients_count, data)
    #tfidf_data = get_tfidf_data(tf_data, idf_data)
    #tfidf_transform(ingredient_data)

    user_likes = generate_user_likes(data_count)
    user_profile = generate_user_profile(tf_data, ingredients_count, user_likes)
    user_pref = generate_user_pref(data_count, tf_data, idf_data, user_profile)
    best_user_pref = max(user_pref)
    best_recipe_pref_index = user_pref.index(best_user_pref)
    print("\n")
    print("best recipe for user: " + str(data[best_recipe_pref_index]["name"]))
    n_largest_user_pref_indexes, n_largest_user_pref = get_n_largest_user_pref(best_user_pref_count, user_pref, user_likes)
    print("n closest recipes to the best recipe for user: ")
    print(str([(index, data[index]["name"]) for index in n_largest_user_pref_indexes]))
    print("\n")
    n_closest_recipe_indexes = n_closest_recipes_to_best_recipe_pref(best_recipe_count, tf_data, best_recipe_pref_index)
    print(str([(index, data[index]["name"]) for index in n_closest_recipe_indexes]))
    #n_closest_users = n_closest_users_to_user_pref(best_user_pref_count, user_pref_data, user_pref)

    # sklearn_tests
    #test_kmeans(tf_data, best_user_pref_count)
    #test_knn(tf_data, best_user_pref_count)

    #best_user_pref = dict()
    #for item in user_pref:
    #    user_pref_index = user_pref.index(item)
    #    best_user_pref[user_pref_index] = item
    #    print("user_likes" + str(user_pref.index(item)) + " : " +
    #          str(user_likes[user_pref_index]))
    #print(best_user_pref)
