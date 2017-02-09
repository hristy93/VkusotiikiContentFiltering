from VkusotiikiContentFiltering import prepare_data, enable_win_unicode_console
import sys
import random
import math
from sklearn.model_selection import train_test_split

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
    naive_bayes(tf_data, user_likes)

    splitRatio = 0.67
    #trainingSet, testSet = splitDataset(dataset, splitRatio)
    #print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
    # prepare model
    train_set, test_set = train_test_split(tf_data)
    print("train_set likes", [item[-1] for item in train_set])
    print("test_set likes", [item[-1] for item in test_set])
    #test_set = [tf_data[item] for item in recipe_ids_test]
    #train_set = [tf_data[item] for item in recipe_ids_train]
    summaries = summarizeByClass(train_set)
    # test model
    # print("recipe_ids_train", recipe_ids_train)
    # print("recipe_ids_train likes", [user_likes[item] for item in recipe_ids_train])
    # print("recipe_ids_test", recipe_ids_test)
    # print("recipe_ids_test likes", [user_likes[item] for item in recipe_ids_test])
    predictions = getPredictions(summaries, test_set)
    accuracy = getAccuracy(test_set, predictions)
    print("predictions: ", predictions, len(predictions))
    print('Accuracy: {}'.format(accuracy))


def naive_bayes(tf_data, user_likes):
    for index, tf_item in enumerate(tf_data):
        tf_item.append(user_likes[index])


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]
 
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated
 
def mean(numbers):
    return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)
 
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    #print("summaries", summaries, len(summaries))
    # print("----------------")
    # print(list(zip(*dataset)))
    del summaries[-1]
    return summaries
 
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries
 
def calculateProbability(x, mean, stdev):
    if stdev != 0:
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        probability = (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
    else:
        probability = 1
    return probability
 
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities
            
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    print("probabilities :", probabilities) 
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel
 
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions
 
def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


if __name__ == "__main__":
        # enables the unicode console encoding on Windows
    if sys.platform == "win32":
        enable_win_unicode_console()

    main()