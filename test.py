import json
import pandas
from blacklist import blacklist
from sklearn.feature_extraction.text import TfidfVectorizer
vv = TfidfVectorizer(stop_words='english') #obiekt do liczenia slow
with open("s.json",encoding="utf-8") as file:
    data = json.load(file)
arr = [] #tabela z tekstami
numOfText = 25 #liczba analizowanych tekstów
labels = []
for i in range(0,60):
    labels.append(data[i]["isAntiVaccine"])
    st = data[i]["text"]
    st = st.lower()
    for j in blacklist:
        st = st.replace(j, " ")
    arr.append(st)
vv.fit(arr) #wrzuecnie do obiektu zeby zaczla liczyc
print(vv.vocabulary_) #drukowanie listy czestosci slow
v = vv.transform(arr) #zamiana na macierz

print(v.toarray()[0]) #drukowanie pierwszego wiersza tej macierzy
df = pandas.DataFrame(data=v.toarray(), columns=vv.get_feature_names()) #konwersja do koncowej tabelki
print(df)#drukowanie tabeli
print(labels)
