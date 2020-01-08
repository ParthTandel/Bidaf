import numpy as numpy
import json
import nltk
from pprint import pprint
from nltk.tokenize import word_tokenize


with open("data/train-v2.0.json") as fl:
    data = json.load(fl)
    data = data["data"]

# count = 0
# para_count = 0
for elem in data:
    for para in elem["paragraphs"]:
        context =  para["context"].lower()
        print(word_tokenize(context))
        for qas in para["qas"]:
            s = 0
            e = 0
            if len(qas["answers"]) > 0:
                s = qas["answers"][0]["answer_start"]
                e = s + len(qas["answers"][0]['text'])
            print(qas['question'],"     =>      ", para["context"][s:e])
            break
        break
    break

# print(count)
# print(para_count)