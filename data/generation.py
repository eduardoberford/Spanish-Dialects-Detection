import os 
import json
import pandas as pd
import re
from tqdm import tqdm
import numpy as np

print("Generating the dataset...")

# list of dialects in the training set
dialects = [d for d in os.listdir(".") if not os.path.isfile(d) and d != '.ipynb_checkpoints']

# define dictionaries for string->int and int->label conversion
fold_label = {
    'ast_texts' : 0,
    'eu_texts' : 1,
    'gl_texts' : 2,
    'an_texts' : 3,
    'lad_texts' : 4,
}
dial_label = {
    0 : 'AST',
    1 : 'EU',
    2 : 'GL',
    3 : 'AN',
    4 : 'LAD',
}

# create training dataset
data = []

for d in tqdm(dialects):
    for name in os.listdir(d + "/AA/"):
        f = open(d + "/AA/" + name, "r")
        lines = f.readlines()
        for l in lines:
            jline = json.loads(l)
            if not jline['text']:
                continue
            data.append([int(jline['id']), jline['url'], jline['title'], jline['text'], fold_label[d]])


columns = {'id':int(), 'url':str, 'title':str, 'text':str, 'label':int()}
df = pd.DataFrame(data, columns = columns)

df = df.drop(columns=["id", "url", "title"])

# clean text
def clean(text):
    text = re.sub(r'==.*?==+', '', text)

    text = text.replace("\n", " ")

    text = text.replace('"', " ")

    regex = re.compile('&[^;]+;') 
    text = re.sub(regex, '', text)


    regex = re.compile('(graph.*/graph|\(.*\)|\[.*\]|parentid>.*/parentid>|BR[^>]+>|bR[^>]+>|Br[^>]+>|br[^>]+>|ns>.*/ns>|timestamp>.*/timestamp>|revision>.*/revision>|contributor>.*/contributor>|model>.*/model>|format>.*/format>|comment>.*/comment>)') 
    text = re.sub(regex, '', text)
    regex = re.compile('(parentid.*/parentid|ns.*/ns|timestamp.*/timestamp|revision.*/revision|contributor.*/contributor|model.*/model|format.*/format|comment.*/comment)') 
    text = re.sub(regex, '', text)

    text = text.replace("revision>", "")
    text = text.replace("br>", "")
    text = text.replace("Br>", "")
    text = text.replace("bR>", "")
    text = text.replace("BR>", "")
    text = text.replace("/br>", "")
    text = text.replace("/Br>", "")
    text = text.replace("/bR>", "")
    text = text.replace("/BR>", "")

    text = text.replace("&quot;","")

    text = text.replace("br clear=all>", "")

    if(len(text) < 50):
        text = np.nan

    return text

print("Saving uncleaned dataset...")
df.to_csv("uncleaned.csv", index=None)

print("Cleaning text...")

df['text'] = df['text'].apply(clean)

# drop rows with nan values
df.dropna(inplace=True)

# drop duplicate entries in the samples
df.drop_duplicates(subset ='text',keep = False, inplace = True) 

# create sentences
print("Splitting sentences...")

import spacy

nlp = spacy.load("es_core_news_sm", disable=['ner', 'lemmatizer', "textcat", "custom", "tagger"])

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, verbose=0)

df['text'] = df['text'].parallel_apply(nlp)

X = df["text"].to_numpy()           
y = df["label"].to_numpy()

print("Creating new data...")
X_train = []
y_train = []
for i, article in tqdm(enumerate(X), total=X.shape[0]):
  for sentence in article.sents:
    X_train.append(sentence)
    y_train.append(y[i])
X_train = np.array(X_train, dtype=object)
y_train = np.array(y_train, dtype=object)

print("Cleaning sentences...")
df = pd.DataFrame({'text': X_train, 'label': y_train}, index=None)

df["text"] = df['text'].apply(lambda x: ''.join(x.text))

df["text"] = df['text'].apply(lambda x: np.nan if len(x)<=20 else x)
df.dropna(inplace=True)

df.dropna(inplace=True)
df.drop_duplicates(subset ='text', keep = False, inplace = True) 

df.to_csv("train.csv", index=None)

print("Dataset created.")