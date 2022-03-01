import json
import pandas
from sklearn.feature_extraction.text import CountVectorizer
vv = CountVectorizer()
with open("s.json") as file:
    data = json.load(file)
arr = []
for i in range(0,25):
    arr.append(data[i]["text"])
vv.fit(arr)
print(vv.vocabulary_)
v = vv.transform(arr)
print(v.toarray()[0])
df = pandas.DataFrame(data=v.toarray(), columns=vv.get_feature_names())
print(df)
