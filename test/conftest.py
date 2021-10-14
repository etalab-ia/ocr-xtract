import pytest

@pytest.fixture()
def prenoms():
    return ['Pierre', 'pierre', 'pierre-henry', 'Pierre henry',]

@pytest.fixture()
def noms():
    return ['Macron', 'Auzière-Jourdan', 'Le Normand',]

@pytest.fixture()
def dates():
    return ['Avril', 'JANVIER', 'hier', '27.03.1989', '05/01/2021', '13 Juin 1993', '09/21', '09/2021', ':31/12/2020',
            '09.08/2021', 'au31/03/2021', '/09/2021']

# @pytest.fixture()
# def mock_data_words(data):
#     return pd.DataFrame()