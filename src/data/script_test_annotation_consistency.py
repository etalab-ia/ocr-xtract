import pandas as pd
import os
import json
import numpy as np
import re
from dateutil import parser


# Test function for code postal

def find_con(n, s):
    result = re.search('\d{%s}'%n, s)
    return result.group(0) if result else result

def is_code_postal(x):
    if len(str(x))>4:
        substring = find_con(5,x)
        if (len(str(substring))==5 and substring.isdecimal()):
            return True
    else:
        return False

# Test function for dates

class FrenchParserInfo(parser.parserinfo):
    MONTHS = [("Jan", "Janvier"),
              ("Fev", "Février"),
              ("Mar", "Mars"),
              ("Avr", "Avril"),
              ("Mai", "Mai"),
              ("Juin", "Juin"),
              ("Juil", "Juillet"),
              ("Aou", "Août"),
              ("Sep", "Septembre"),
              ("Oct", "Octobre"),
              ("Nov", "Novembre"),
              ("Dec", "Décembre")]
    WEEKDAYS = [('Lun', 'Lundi'),
                ('Mar', 'Mardi'),
                ('Mer', 'Mercredi'),
                ('Jeu', 'Jeudi'),
                ('Ven', 'Vendredi'),
                ('Sam', 'Samedi'),
                ('Dim', 'Dimanche')]

def is_date(string, fuzzy=True):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        parser.parse(string, fuzzy=fuzzy, parserinfo=FrenchParserInfo())
        return True
    except:
        return False


# Apply tests to dataframe extracted form the annotation json
path_df = "data/quittances/annotation/sample_1.csv"
df = pd.read_csv(path_df, sep = "\t")

df["test_date"] = df.apply(lambda row: is_date(str(row["word"])) if row["label"]=="periode" else np.nan, axis = 1)
df["test_cp"] = df.apply(lambda row: is_code_postal(row["word"]) if row["label"]=="code_postal" else np.nan, axis = 1)