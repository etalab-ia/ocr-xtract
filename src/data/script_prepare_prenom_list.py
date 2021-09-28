import pandas as pd

df_prenoms = pd.read_csv("data/nat2020.csv", sep=";")
df_prenoms = df_prenoms[df_prenoms.annais != 'XXXX']
df_prenoms = df_prenoms[df_prenoms.preusuel != '_PRENOMS_RARES']

grouped = df_prenoms[df_prenoms['annais'].astype(int) > 1950].groupby('preusuel').sum()
grouped = grouped['nombre']
grouped.to_csv('data/prenom_data.csv',header=None)
