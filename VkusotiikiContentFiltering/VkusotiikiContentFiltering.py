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
from collections import defaultdict
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
def generate_user_likes(data_count, binary_propabilities):
    #user_likes = []
    #for _ in range(data_count):
    #    user_likes.append(random.randint(0, 1))
    #return user_likes
    user_likes = np.random.choice([0, 1], size=(data_count,), p=binary_propabilities)
    print("{} recipes liked from {}".format(sum(user_likes), data_count))
    return user_likes

""" Generates the users' favourite recipes with particular recipe ids """
def generate_user_likes_by_recipes_ids(data_count, fav_recipe_ids):
    user_likes = [0] * data_count
    for recipe_id in fav_recipe_ids:
        user_likes[recipe_id] = 1
    
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
def get_tfidf_data(tf_data, idf_data):
    tfidf_data = list()
    for i in range(data_count):
        tf_list = tf_data[i]
        tfidf_data.append(np.dot(tf_list, idf_data))
    #print(tfidf_data)
    return tfidf_data

""" Gets the ingreients from the json data """
def get_ingredients(data):
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
                for meaty_key, meaty_value in meaty_ingredients.items():
                    #if meaty_key in meaty_value and ingredient in meaty_value:
                    #    print("tes")
                    if ingredient in meaty_value:
                        #ingredient_inner_data.append(0)
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
    print("n_largest_user_pref_indexes: ", str(n_largest_user_pref_indexes))
    print("n_largest_user_pref: ", str(n_largest_user_pref))
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
    print("n_closest_recipes: ", str(n_closest_recipes))
    print("n_closest_recipe_indexes: ", str(n_closest_recipe_indexes))
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
    print("n_closest_users: ", str(n_closest_users))
    print("n_closest_user_indexes: ", str(n_closest_user_indexes))
    return n_closest_user_indexes
    

""" Tests the kNN methods of sklearn module """
def test_knn(tf_data, k, best_recipe_pref_index):
    print("\ndata from kNN with {} neighbours".format(k))
    best_recipe_pref = tf_data.pop(best_recipe_pref_index)
    arr = np.array(tf_data)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(arr)
    distances, indices = nbrs.kneighbors((best_recipe_pref,), k)
    print("distances\n", str(distances))
    print("indices\n", str(indices))
    return indices


""" Tests the k-Means methods of sklearn module """
def test_kmeans(tf_data, k,  best_recipe_pref_index):
    print("\ndata from k-Means with {} clusters".format(k))
    arr = np.array(tf_data)
    kmeans = KMeans(n_clusters=k)
    clustered_data = kmeans.fit(arr)
    distance_data = kmeans.transform(arr)
    predicted_cluster = kmeans.predict((arr[best_recipe_pref_index],))[0]
    print("clustered_data count: ", str(len(clustered_data.labels_)))
    print("distance_data count: ", str(len(distance_data)))
    print("predicted_cluster count: ", str(predicted_cluster))
    index = 0
    closest_recipes = {}
    for item in clustered_data.labels_:
        if item == predicted_cluster:
            closest_recipes[index] = distance_data[index]
        index += 1
    return closest_recipes
    #closest_recipes = [clustered_dataitem for item in clustered_data if item == predicted_cluster]
    #print("kmeans\n" + str(kmeans))
    #print("cluster_centers_\n" + str(kmeans.cluster_centers_))


""" Finds the closest ingredients to be refactored and trimmed """
def find_closest_ingredients(ingredients):
    closest_ingredients = dict()
    #for item in ingredients:
    #    closest_ingredients[item] = difflib.get_close_matches(item, ingredients)
    #    print(ingredients.index(item), item, closest_ingredients[item])
    #print(closest_ingredients)

def find_meaty_ingredients(ingredients, meat_food, fish_food):
    meaty_ingredients = dict()
    print("meat_food count: " + str(len(meat_food)))
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

""" Adds the meaty food as ingredients"""
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

""" Prints the names of the recipes by given indexes """
def print_recipe_names_by_indexes(data, indexes):
     print(str([(index, data[index]["name"]) for index in indexes]))

if __name__ == "__main__":
    if sys.platform == "win32":
        enable_win_unicode_console()
    json_file_name = "recipes_500_refined_edited.json"
    propability_of_one = 0.1
    binary_propabilities = [1 - propability_of_one, propability_of_one]
    meat_food = { "кайма", "телешк", "овч", "агнешк", "свинск", "суджук", "филе", "заеш", "месо", "кайма", "кренвирш", "кюфте", "говежд", "скарид", "овнешк", "пиле", "пуйка", "патешк", "надениц", "пушен", "колбас", "еленск",
        "шунка",  "гъши", "гъск", "бекон", "агнешк", "кървавиц"};
    fish_food = { "риба", "скумрия", "шаран", "рибн" , "рибен", "сьомг", "пъстърва", "ципура", "щука", "риба тон", "треска" }
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
    
    meaty_ingredients = find_meaty_ingredients(ingredients, meat_food, fish_food)
    #br = 0
    #for key, value in meaty_ingredients.items():
    #    if key in value:
    #        br += 1
    #print("same: ", br)
    add_food_groups_as_ingredients(meaty_ingredients, ingredients)
    process_data(data, ingredients, ingredient_data, ingredients_count_info, meaty_ingredients)
    #find_closest_ingredients(ingredients)
    
    idf_data = get_idf_data(ingredients_count_info, data_count)
    tf_data = get_tf_data(ingredient_data, data)
    tfidf_data = get_tfidf_data(tf_data, idf_data)
    #tfidf_transform(ingredient_data)

    user_likes = generate_user_likes(data_count, binary_propabilities)
    user_profile = generate_user_profile(tf_data, ingredients_count, user_likes)
    user_pref = generate_user_pref(data_count, tf_data, idf_data, user_profile)
    best_user_pref = max(user_pref)
    best_recipe_pref_index = user_pref.index(best_user_pref)
    print("\n")
    print("best recipe for user: " + str((best_recipe_pref_index, data[best_recipe_pref_index]["name"])))
    n_largest_user_pref_indexes, n_largest_user_pref = get_n_largest_user_pref(best_user_pref_count, user_pref, user_likes)
    print("n closest recipes to the best recipe for user: ")
    print_recipe_names_by_indexes(data, n_largest_user_pref_indexes)
    print("\n")
    n_closest_recipe_indexes = n_closest_recipes_to_best_recipe_pref(best_recipe_count, tf_data, best_recipe_pref_index)
    print_recipe_names_by_indexes(data, n_closest_recipe_indexes)
    #n_closest_users = n_closest_users_to_user_pref(best_user_pref_count, user_pref_data, user_pref)

    #TODO - add closest recipes by category

    # sklearn_tests
    #test_kmeans(tf_data, best_user_pref_count, best_recipe_pref_index)
    knn_data_indexes = test_knn(tf_data, best_user_pref_count, best_recipe_pref_index)[0]
    print_recipe_names_by_indexes(data, knn_data_indexes)


    #best_user_pref = dict()
    #for item in user_pref:
    #    user_pref_index = user_pref.index(item)
    #    best_user_pref[user_pref_index] = item
    #    print("user_likes" + str(user_pref.index(item)) + " : " +
    #          str(user_likes[user_pref_index]))
    #print(best_user_pref)
