import spacy
import numpy as np
nlp = spacy.load('fr_core_news_sm')


def return_word_vector(text: str) -> np.array:
    doc = nlp(text)
    return doc
