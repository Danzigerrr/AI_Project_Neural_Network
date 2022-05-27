import json
from pandas._libs.tslibs.period import validate_end_alias
import gensim
from gensim.models import Word2Vec
import nltk
import nltk.data
import pandas
from blacklist import blacklist, ignored_tokens, tag_list
from sklearn.feature_extraction.text import TfidfVectorizer
from mappings import word_map

with open("s.json", encoding="utf-8") as file:
    data = json.load(file)
num_of_text = len(data)  # liczba analizowanych tekstów


def get_data():
    """
    This functions imports the data basing on json file.

    Returns
        :vectorizer_dt: TF-IDF is a numerical statistic that is intended to reflect how important a word is to a document
        :labels save: info about isAntiVaccine
        :featureNames: feature names from vectorizer_dt got from texts basing on TF-IDF method
    """

    arr = [] # array to save texts
    labels = [] # to save info about if text is isAntiVaccine or not
    for i in range(0, num_of_text):
        text = data[i]["text"]
        labels.append(data[i]["isAntiVaccine"])
        clearText = removeUnnededWords(text)
        arr.append(clearText)

    featureNames, vectorizer_dt = getFeatureNamesFromTFIDFVectorizer(arr)
    return vectorizer_dt, labels, featureNames

def get_word_embeddings():
    t_for = []
    t_against = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for i in range(num_of_text):
        text = data[i]["text"]
        text = tokenizer.tokenize(text)
        for j in text:
            d = []
            for k in nltk.word_tokenize(j):
                d.append(k.lower())
            if data[i]["isAntiVaccine"] == 0:
                t_for.append(d)
            else:
                t_against.append(d)
    model_for = gensim.models.Word2Vec(t_for, vector_size = 500)
    model_against = gensim.models.Word2Vec(t_against, vector_size = 500)
    v_for = model_for.wv.key_to_index
    v_against = model_against.wv.key_to_index
    v_for = {x: v_for[x] for x in v_for.keys() if x in v_against.keys()}
    v_against = {x: v_against[x] for x in v_against.keys() if x in v_for.keys()}

    output = []
    labels = []
    featureNames = []
    for key, index in v_for.items():
        output.append(model_for.wv[index])
        labels.append(0)
        featureNames.append(key)

    for key, index in v_for.items():
        output.append(model_against.wv[index])
        labels.append(1)
        featureNames.append(key)
    print(featureNames)
    return output, labels, featureNames

def removeUnnededWords(text):
    """
    This functions removes useless words from the text.

    Parameters:
        :text: a text to analyze

    Returns:
        :converted_text: converted text
    """
    text = text.lower()
    # put a blank space in place of ignored symbols
    for j in ignored_tokens:
        text = text.replace(j, " ")
    # tokenize text, remove unneeded data basing on tokens
    tokenized_text = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokenized_text)
    for (word, tag) in tags:
        if tag in tag_list:
            tokenized_text.remove(word)
    # handle pairs of words ending with "s" and without it
    for (w, m) in word_map.items():
        tokenized_text = list(map(lambda it: it.replace(w, m), tokenized_text))
    # remove words from text included in blacklist
    converted_text = ' '.join(filter(lambda x: blacklist.count(x) == 0, tokenized_text))
    return converted_text


def getFeatureNamesFromTFIDFVectorizer(arr):
    """
    This functions removes useless words from the text.

    Convert a collection of raw documents to a matrix of TF-IDF features.
    TF-IDF --> TF – term frequency, IDF – inverse document frequency.
    It is a numerical statistic that is intended to reflect how important a word is to a document.

    Parameters:
        :arr: An iterable which generates either str, unicode or file objects.

    Returns:
        :featureNames: feature names from vectorizer_dt got from texts basing on TF-IDF method
        :vectorizer_dt: text array transformed to document-term matrix

    """
    # min_df --> Terms that were ignored because they occurred in too few documents
    vectorizer = TfidfVectorizer(stop_words='english', min_df=0.035)
    vectorizer.fit(arr)  # Learn vocabulary and idf from training set.
    vectorizer_dt = vectorizer.transform(arr)  # Transform documents to document-term matrix
    # print(vectorizer.vocabulary_)
    featureNames = vectorizer.get_feature_names()
    return featureNames, vectorizer_dt


def get_important(vocab):
    """
    This function gets only important words from the texts.
    The important words(vocab) are searched in the texts.
    The words are saved to the 'temp' array only if they exist in the provided data(texts).

    Parameters:
        :vocab: A list of important words (returned from permutationImportance() function)

    Returns:
        :vectorizer_dt: text array transformed to document-term matrix
        :featureNames: feature names from vectorizer_dt got from texts basing on TF-IDF method
    """

    #genreownaie nowej prestrezni wekt. ksladajacej sie tlyko z istonych slow
    # iteruje przez jesli pasuja to wrzuca je, a jesli nie ma ich w "vocav" to je pomija
    arr = []  # tabela z tekstami
    for i in range(0, num_of_text):
        temp = []
        st = data[i]["text"]
        st = st.lower()
        tokens = st.split()
        for word in tokens:
            if word in vocab:
                temp.append(word)
        st = ' '.join(temp)
        arr.append(st)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(arr)  # Learn vocabulary and idf from training set.
    vectorizer_dt = vectorizer.transform(arr)  # Transform documents to document-term matrix
    featureNames = vectorizer.get_feature_names()
    # print(vectorizer.vocabulary_)
    # print(len(vocab))
    # print(len(vectorizer.vocabulary_))

    # TODO --> df mozna usunąć, bo nie jest używane
    df = pandas.DataFrame(data=vectorizer_dt.toarray(), columns=featureNames)
    # print(df)
    return vectorizer_dt, featureNames
