from func import get_word_embeddings2, num_of_text, data, removeUnnededWords, getFeatureNamesFromTFIDFVectorizer
from som import som, plotSom
from blacklist import tag_list
from ff import keras, keras_classify
from svm import createSvmClassifier, classify


def prepareDataWithSkippedTags(skipped_tags):
    tags = tag_list + skipped_tags
    arr = []  # array to save texts
    labels = []  # to save info about if text is isAntiVaccine or not
    for i in range(0, num_of_text):
        text = data[i]["text"]
        labels.append(data[i]["isAntiVaccine"])
        clearText = removeUnnededWords(text, tags)
        arr.append(clearText)

    featureNames, vectorizer_dt = getFeatureNamesFromTFIDFVectorizer(arr)
    return vectorizer_dt, labels, featureNames


if __name__ == "__main__":

    dataAll, labels, names = prepareDataWithSkippedTags([])
    model,  X_train, y_train, X_test, y_test = keras(dataAll)
    keras_classify(model, X_train, y_train, X_test, y_test)
    svc = createSvmClassifier(X_train, y_train)
    classify(svc, X_train, y_train, X_test, y_test)

    dataWithoutAdj, labelsWithoutAdj, NamesWithoutAdj = prepareDataWithSkippedTags(["JJ", "JJR", "JJS"])
    modelWithoutAdj, X_train, y_train, X_test, y_test = keras(dataWithoutAdj)
    keras_classify(modelWithoutAdj, X_train, y_train, X_test, y_test)
    svc = createSvmClassifier(X_train, y_train)
    classify(svc, X_train, y_train, X_test, y_test)

    dataWithoutAdv, labelsWithoutAdv, NamesWithoutAdv = prepareDataWithSkippedTags(["RB", "RBR", "RBS"])
    modelWithoutAdv, X_train, y_train, X_test, y_test = keras(dataWithoutAdv)
    keras_classify(modelWithoutAdv, X_train, y_train, X_test, y_test)
    svc = createSvmClassifier(X_train, y_train)
    classify(svc, X_train, y_train, X_test, y_test)

    dataWithoutAdjAdv, labelsWithoutAdjAdv, NamesWithoutAdjAdv = prepareDataWithSkippedTags(["JJ", "JJR", "JJS", "RB", "RBR", "RBS"])
    modelWithoutAdjAdv, X_train, y_train, X_test, y_test = keras(dataWithoutAdjAdv)
    keras_classify(modelWithoutAdjAdv, X_train, y_train, X_test, y_test)
    svc = createSvmClassifier(X_train, y_train)
    classify(svc, X_train, y_train, X_test, y_test)

