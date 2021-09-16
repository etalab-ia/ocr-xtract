import spacy
import numpy as np
nlp = spacy.load('fr_core_news_sm')

text="Michel est allée au 20 rue des martyrs Madrid le 24 Juin 2020"
words=text.split() # this is to simulate the words that would come from Doctr
doc = spacy.tokens.doc.Doc(
    nlp.vocab, words=words)
for name, proc in nlp.pipeline:
    doc = proc(doc)

for d in doc.ents:
    print(d.label_)


