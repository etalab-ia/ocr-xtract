import pandas as pd

df_noms = pd.read_csv("data/noms2008nat_txt.txt", sep="\t")
df_noms = df_noms[df_noms.NOM != 'AUTRES NOMS']
df_noms['nombre'] = df_noms[['_1891_1900', '_1901_1910', '_1911_1920', '_1921_1930',
       '_1931_1940', '_1941_1950', '_1951_1960', '_1961_1970', '_1971_1980',
       '_1981_1990', '_1991_2000']].sum(axis=1)
df_noms.set_index(df_noms.NOM, inplace=True)
df_noms = df_noms['nombre']
df_noms.to_csv('data/nom_data.csv',header=None)
