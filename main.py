from som import som, plotSom
# from ff import keras
# from svm import getImportantData
from func import get_word_embeddings2, get_word_embeddings
import numpy as np

"""
    SOM map is generated. Data to model is provided by the getImportantData() function. 
"""


def saveDistanceInfoToExcelFile(distances):
    import xlsxwriter
    workbook = xlsxwriter.Workbook('SOMNeuronDistances.xlsx')
    worksheet = workbook.add_worksheet()

    # write column names
    columnNames = ["DataType", "LearningRate", "sumFor", "avgFor", "sumAgainst", "avgAgainst"]
    for iter in range(len(columnNames)):
        worksheet.write(0, iter, columnNames[iter])

    # calcualte differences between before and after values
    additionalSize = int(len(distances) / 2)
    allSize = len(distances) + additionalSize
    distances.append([0, 0, 0, 0, 0])
    for rowIter in range(2, allSize, 3):
        newRow = []
        size = len(distances[rowIter])
        for colIter in range(len(distances[rowIter]) - 1, 0, -1):
            valueBefore = distances[rowIter - 2][colIter]
            valueAfter = distances[rowIter - 1][colIter]
            newRow.insert(0, round(valueAfter - valueBefore, 2))
        newRow.insert(0, distances[rowIter - 2][0])  # Learning rate value

        distances.insert(rowIter, newRow)

    # save the distance Data into file
    dataTypes = ["Before", "After", "Difference"]
    row = 1
    cols = 5
    typeIter = 0
    for data in distances[:-1]:  # last row is filled with zeros
        worksheet.write(row, 0, dataTypes[typeIter % len(dataTypes)])
        for i in range(1,cols):
            worksheet.write(row, i, data[i-1])
        row += 1
        typeIter += 1

    workbook.close()


if __name__ == "__main__":
    _, data, labels, names = get_word_embeddings(.0, .8)
    _, data2, labels, names = get_word_embeddings(.8, 1.0)
    _, data3, labels, names = get_word_embeddings()

    distances = []

    sommodel = som(data, .15)
    distances.append(plotSom(sommodel, data, labels, names, 0.15, "before"))
    sommodel.train_random(data2, 50)
    distances.append(plotSom(sommodel, data3, labels, names, 0.15, "after"))

    sommodel = som(data, .1)
    distances.append(plotSom(sommodel, data, labels, names, 0.1, "before"))
    sommodel.train_random(data2, 50)
    distances.append(plotSom(sommodel, data3, labels, names, 0.1, "after"))

    sommodel = som(data, .01)
    distances.append(plotSom(sommodel, data, labels, names, 0.01, "before"))
    sommodel.train_random(data2, 50)
    distances.append(plotSom(sommodel, data3, labels, names, 0.01, "after"))

    sommodel = som(data, .001)
    distances.append(plotSom(sommodel, data, labels, names, 0.001, "before"))
    sommodel.train_random(data2, 50)
    distances.append(plotSom(sommodel, data3, labels, names, 0.001, "after"))

    sommodel = som(data, .0001)
    distances.append(plotSom(sommodel, data, labels, names, 0.0001, "before"))
    sommodel.train_random(data2, 50)
    distances.append(plotSom(sommodel, data3, labels, names, 0.0001, "after"))

    saveDistanceInfoToExcelFile(distances)
