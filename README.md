# Neural Network modelling deep beliefs about vaccination

Authors: Bogna Lew, Bartosz Strzelecki, Krzysztof Nazar

The report of this project is included in this repository as *Report final version.pdf*.

## Description
**The goal of this project was to analyze the conceptual framework of two social groups: for and against vaccinations.**

The major steps in this project were:
 1. Collect texts for and against vaccinations available online. The database in this project contains approximately 150 texts. 
 2. Classify each text to the "for" or "against" group and save it as a JSON file.
 3. Improve the quality of data by filtering unnecessary words and signs.
 4. Analyse the data by various techniques. Examine how the results change after changing particular hyperparameters, for example learning rate. 

## Analysis methods and tools
 - Keras Classificator
 - SVM Classificator
 - Permutation feature importance
 - SHAP (SHapley Additive exPlanations)
 - SOM Kohonen maps

## Used libraries
 - Numpy
 - Pandas
 - sklearn
 - TensorFlow
 - Keras
 - SHAP (SHapley Additive exPlanations)
 - NLTK
 - SOM
 - matplotlib
 - math

## Future improvements
 - In the further analysis new techniques can be used. 
 - The database can be expanded.
