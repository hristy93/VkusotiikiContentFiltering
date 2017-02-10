import sys
import random
import math
from VkusotiikiContentFiltering import prepare_data, enable_win_unicode_console
from sklearn.model_selection import train_test_split
from numpy import mean


def main():
    fetched_data = prepare_data()
    tf_data = fetched_data.get('tf_data')
    user_likes = fetched_data.get('user_likes')
    recipe_ids_test = fetched_data.get('recipe_ids_test')
    recipe_ids_train = fetched_data.get('recipe_ids_train')

    train_tf_data, test_tf_data, train_likes, test_likes = train_test_split(tf_data, user_likes)
    #test_tf_data = [tf_data[item] for item in recipe_ids_test]
    #train_tf_data = [tf_data[item] for item in recipe_ids_train]

    class_statistical_data = get_statistical_data_by_class(train_tf_data, train_likes)

    #print("train_tf_data: ", train_tf_data)
    print("train_likes: ", train_likes)
    #print("test_tf_data: ", test_tf_data)
    print("test_likes: ", test_likes)

    predictions = get_predictions(class_statistical_data, test_tf_data)
    accuracy = get_accuracy(test_likes, predictions)
    print("predictions: ", predictions, len(predictions))
    print('accuracy: {}'.format(accuracy))


def group_tf_data_by_class(tf_data, user_likes):
    grouped_tf_data = {}
    grouped_tf_data[0] = list()
    grouped_tf_data[1] = list()

    for tf_index, tf_value in enumerate(tf_data):
        tf_value_like = user_likes[tf_index]
        grouped_tf_data[tf_value_like].append(tf_value)

    return grouped_tf_data


def standard_deviation(tf_values):
    avg = mean(tf_values)
    variance = sum([pow(x - avg,2) for x in tf_values])/float(len(tf_values)-1)
    deviation = math.sqrt(variance)
    return deviation
 

def get_attribute_probability(input_value, mean, standard_deviation):
    if standard_deviation != 0:
        input_value_diff = input_value - mean
        exponent = math.exp(-((input_value_diff * input_value_diff) / (2  * standard_deviation * standard_deviation)))
        probability = (1 / (math.sqrt(2*math.pi) * standard_deviation)) * exponent
    else:
        probability = 1

    return probability


def get_class_probabilities(class_statistical_data, input_data):
    probabilities = dict()

    for class_id, class_statistical_value in class_statistical_data.items():
        probabilities[class_id] = 1
        for index in range(len(class_statistical_value)):
            input_value = input_data[index]
            mean, standard_deviation = class_statistical_value[index]
            probabilities[class_id] *= get_attribute_probability(input_value, mean, standard_deviation)

    return probabilities


def get_statistical_data(dataset):
    statistical_data = [(mean(attribute), standard_deviation(attribute)) for attribute in zip(*dataset)]
    return statistical_data
 

def get_statistical_data_by_class(tf_data, user_likes):
    statistical_data = dict()
    group_tf_data = group_tf_data_by_class(tf_data, user_likes)
    #print("group_tf_data: ", group_tf_data)

    for class_value, tf_value in group_tf_data.items():
        statistical_data[class_value] = get_statistical_data(tf_value)

    return statistical_data


def predict_likes(class_statistical_data, input_data):
    probabilities = get_class_probabilities(class_statistical_data, input_data)
    print("probabilities :", probabilities) 

    if probabilities[0] > probabilities[1]:
        class_type = 0
    else:
        class_type = 1

    return class_type

 
def get_predictions(class_statistical_data, test_data):
    predictions = []

    for test_likes in range(len(test_data)):
        result = predict_likes(class_statistical_data, test_data[test_likes])
        predictions.append(result)

    return predictions
 

def get_accuracy(test_likes, predictions):
    correct_predictions_count = 0

    for index in range(len(test_likes)):
        if test_likes[index] == predictions[index]:
            correct_predictions_count += 1

    accuracy = correct_predictions_count / float(len(test_likes))

    return accuracy * 100.0


if __name__ == "__main__":
        # enables the unicode console encoding on Windows
    if sys.platform == "win32":
        enable_win_unicode_console()

    main()