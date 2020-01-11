import numpy as numpy
import json
import nltk
from pprint import pprint
import re
import spacy
import unicodedata

nlp = spacy.load('en_core_web_sm', parser=False)
spctokenizer = nlp.Defaults.create_tokenizer(nlp)



def spacyTokenizer(s):
    s = s.replace("-", " ")
    s = re.sub(r'[!@$#^&*()\[\]{};:,.\/<>?\|`~=_+–]', ' \g<0>', s)
    s = " ".join(s.split())
    tokens = spctokenizer(s.lower())
    normalize_text = lambda text : unicodedata.normalize('NFD', text)
    tokens = [normalize_text(w.text) for w in tokens]
    return tokens

def getWordIndexs(tokens, answer):
    s = []
    answer_index = 0
    for index, word in enumerate(tokens):
        if str(word) == str(answer[answer_index]):
            s.append(index)
            answer_index = answer_index + 1
        else:
            s = []
            answer_index = 0
        if answer_index >= len(answer):
            break
    return s

with open("data/train-v2.0.json") as fl:
    data = json.load(fl)
    data = data["data"]

vocab = {
    "<start>" : 0
}
vocab_index = 1
context_index = 1
question_index = 1
count = 0
total = 0

max_context = 0
max_question = 0

question_processed = {}
context_processed = {}

for elem in data:
    for para in elem["paragraphs"]:
        context =  para["context"].lower()
        tokens = spacyTokenizer(context)
        tokens = ["<start>"] + tokens
        if len(tokens) > max_context:
            max_context = len(tokens)

        for word in tokens:
            if word not in vocab:
                vocab[word] = vocab_index
                vocab_index = vocab_index + 1

        inverse_token = [vocab[w] for w in tokens]
        context_processed[context_index] = inverse_token
        if context_index % 100 == 0:
            with open("processedData/context/" + str(context_index) + ".json", "w+") as fl:
                json.dump(context_processed, fl)
            context_processed = {}

        for qas in para["qas"]:
            q_tokens = spacyTokenizer(qas["question"].lower())
            if len(q_tokens) > max_question:
                max_question = len(tokens)

            for word in q_tokens:
                if word not in vocab:
                    vocab[word] = vocab_index
                    vocab_index = vocab_index + 1

            total = total + 1               
            if len(qas["answers"]) > 0:
                answer_text = qas["answers"][0]['text'].lower()
                answer_text_token = spacyTokenizer(answer_text)
                ans_span = getWordIndexs(tokens, answer_text_token)
                if qas["is_impossible"] != True and " ".join([tokens[i] for i in ans_span]) != " ".join(answer_text_token):
                    count = count + 1

            if qas["is_impossible"] == True or len(ans_span) == 0:
                ans_span = [0]

            question_processed[question_index] = {
                "context" : context_index,
                "question" : [vocab[w] for w in q_tokens],
                "start" : ans_span[0],
                "end" : ans_span[-1]
            }

            if question_index % 1000 == 0:
                with open("processedData/question/" + str(question_index) + ".json", "w+") as fl:
                    json.dump({"data" : question_processed}, fl)
                question_processed = {}
            question_index = question_index + 1
        context_index = context_index + 1

with open("processedData/context/" + str(context_index) + ".json", "w+") as fl:
    json.dump(context_processed, fl)

with open("processedData/question/" + str(question_index) + ".json", "w+") as fl:
    json.dump({"data" : question_processed}, fl)