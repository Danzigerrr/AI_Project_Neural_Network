import json
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
vv = TfidfVectorizer() #obiekt do liczenia slow
with open("s.json",encoding="utf-8") as file:
    data = json.load(file)
arr = [] #tabela z tekstami
numOfText = 25 #liczba analizowanych tekstów
for i in range(0,numOfText):
    arr.append(data[i]["text"]) #zapelnienie tablicy z tekatmi elementami "text" z Jsona
vv.fit(arr) #wrzuecnie do obiektu zeby zaczla liczyc
print(vv.vocabulary_) #drukowanie listy czestosci slow
v = vv.transform(arr) #zamiana na macierz

print(v.toarray()[0]) #drukowanie pierwszego wiersza tej macierzy
df = pandas.DataFrame(data=v.toarray(), columns=vv.get_feature_names()) #konwersja do koncowej tabelki
print(df)#drukowanie tabeli
