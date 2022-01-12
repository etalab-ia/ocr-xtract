import random
import json

# reuse from https://github.com/alxcart/SampleByFeatures/blob/master/main_sample_plan.py
dicSampleLength = {2: [2, 2, 3],
                   9: [2, 3, 5],
                   16: [3, 5, 8],
                   26: [5, 8, 13],
                   51: [5, 13, 20],
                   91: [8, 20, 32],
                   151: [13, 32, 50],
                   281: [20, 50, 80],
                   501: [32, 80, 125],
                   1201: [50, 125, 200],
                   3201: [80, 200, 315],
                   10001: [125, 315, 500],
                   35001: [200, 500, 800],
                   150001: [315, 800, 1250],
                   500001: [500, 1250, 2000]}

List_LQA = ["0,40%", "0,65%", "1,00%", "1,50%", "2,50%", "4,00%", "6,50%", "10,00%", "15,00%", "25,00"]  # len : 10
List_nivel_inspecao = ["I", "II", "III"]
List_tipo_inspecao = ["simples", "dupla", "múltipla"]

TAB_LQA = {2: ["A", 2, 0, 0, "down", "down", "down", "down", "down", "down", 0, "down", "down", 1],
           3: ["B", 3, 2, 0, "down", "down", "down", "down", "down", 0, "up", "down", 1, 2],
           5: ["C", 5, 3, 0, "down", "down", "down", "down", 0, "up", "down", 1, 2, 3],
           8: ["D", 8, 5, 2, "down", "down", "down", 0, "up", "down", 1, 2, 3, 5],
           13: ["E", 13, 8, 3, "down", "down", 0, "up", "down", 1, 2, 3, 5, 7],
           20: ["F", 20, 13, 5, "down", 0, "up", "down", 1, 2, 3, 5, 7, 10],
           32: ["G", 32, 20, 8, 0, "up", "down", 1, 2, 3, 5, 7, 10, 14],
           50: ["H", 50, 32, 13, "up", "down", 1, 2, 3, 5, 7, 10, 14, 21],
           80: ["J", 80, 50, 20, "down", 1, 2, 3, 5, 7, 10, 14, 21, "up"],
           125: ["K", 125, 80, 32, 1, 2, 3, 5, 7, 10, 14, 21, "up", "up"],
           200: ["L", 200, 125, 50, 2, 3, 5, 7, 10, 14, 21, "up", "up", "up"],
           315: ["M", 315, 200, 80, 3, 5, 7, 10, 14, 21, "up", "up", "up", "up"],
           500: ["N", 500, 315, 125, 5, 7, 10, 14, 21, "up", "up", "up", "up", "up"],
           800: ["P", 800, 500, 200, 7, 10, 14, 21, "up", "up", "up", "up", "up", "up"],
           1250: ["Q", 1250, 800, 315, 10, 14, 21, "up", "up", "up", "up", "up", "up", "up"],
           2000: ["R", 2000, 1250, 500, 14, 21, "up", "up", "up", "up", "up", "up", "up", "up"]}


# ### Função tamanho da amostra (n)
# Funcao para encontrar a letra codigo a partir do N e do nivel de inspecao
def n0(N, nivel_inspecao):  # função sample size inicial
    """ Tamanho da amostra (n): esta função retorna o tamanho da amostra (n), a partir do tamanho da população (N) e do
    nivel de inspeção: 0 - brando; 1 - normal; 2 - severo.
    Se N = 1 será inspeção completa.
    Se N < 1 não haverá inspeção.
    Se N >=2 inspecao amostral
    O resultado contempla: o tamanho da amostra inicial"""
    # Identificando a chave do tamanho da amostra
    if N == 1:
        msg = "inspeção completa"
        letra_codigo = "inspeção completa"
        sample_size = 1
    if N < 1:
        msg = "camada sem registro"
        letra_codigo = "camada sem registro"
        sample_size = 0
    if N >= 2:
        msg = "inspeção amostral"
        for i in sorted(dicSampleLength.keys(), reverse=True):
            if N >= i:
                index1 = i
                break
        # tamanho da amostra inicial n0, sem considerar o LQA desejado.
        sample_size = dicSampleLength[index1][nivel_inspecao]
        letra_codigo = TAB_LQA.get(sample_size)[0]

    return sample_size, msg, letra_codigo
    # tamanho da amostra, mensagem, letra codigo


# ### Função: número de aceitação (Ac) a partir de n e do LQA
def Ac(n, lqa):
    """
    Esta função retorno o número de aceitação (Ac)
    com base no tamanho da amostra (n) e
    no limite de qualidade aceitável  (LQA)
    """
    if n >= 2 and n <= 2000:
        num_aceitacao = TAB_LQA.get(n)[lqa]
        letra_codigo = TAB_LQA.get(n)[0]
    if n <= 0:
        num_aceitacao = 0
        letra_codigo = "sem letra código"
    if n == 1:
        num_aceitacao = 1
        letra_codigo = "sem letra código"
    if n > 2000:
        n = 2000
        num_aceitacao = TAB_LQA.get(n)[lqa]
        letra_codigo = TAB_LQA.get(n)[0]

    if num_aceitacao == "down":
        tab_index = {}
        letra_codigo = TAB_LQA.get(n)[0]
        for i in enumerate(TAB_LQA):
            # print (i[0], i[1], TAB_LQA.get(i[1])[lqa])
            tab_index[i[0]] = i[1], TAB_LQA.get(i[1])[lqa], TAB_LQA.get(i[1])[0]
            for j in tab_index:
                if tab_index[j][0] == n and tab_index[j][1] == "down":
                    x = j + 1
                    if tab_index.get(x) is not None:
                        n, num_aceitacao, letra_codigo = tab_index.get(x)
                        # print (n)
                        # print (Ac)
    if num_aceitacao == "up":
        tab_index = {}
        letra_codigo = TAB_LQA.get(n)[0]
        # Ac = Ac(n, lqa)
        for i in enumerate(sorted((TAB_LQA), reverse=True)):
            # print (i[0], i[1],  TAB_LQA.get(i[1])[lqa])
            tab_index[i[0]] = i[1], TAB_LQA.get(i[1])[lqa], TAB_LQA.get(i[1])[0]
            for j in tab_index:
                if tab_index[j][0] == n and tab_index[j][1] == "up":
                    x = j + 1
                    if tab_index.get(x) is not None:
                        n, num_aceitacao, letra_codigo = tab_index.get(x)
                        # print (n)
                        # print (Ac)
    return n, num_aceitacao, letra_codigo


# Wrapper function (not from original code cited earlier


def get_num_sample_number_acceptation(N, inspection_level, aql):
    """inputs :
            - N: the total size of the sample
            - inspection level : "I", "II", "III"
            - aql: acceptance quality limit : must be integer in [0.4, 0.65, 1, 1.5, 2.5, 4, 6.5, 10, 15, 25]
        outputs:
            - letter of in the AQL table
            - size of sample to draw
            - maximum number of defective pieces for acceptance
    """
    dict_inscpetion_level = {"I": 0, "II": 1, "III": 2}
    inspection_number = dict_inscpetion_level[inspection_level]
    n = n0(N, inspection_number)[0]
    letter = n0(N, inspection_number)[2]
    dict_aql = {0.4: 4, 0.65: 5, 1: 6, 1.5: 7, 2.5: 8, 4: 9, 6.5: 10, 10: 11, 15: 12, 25: 13}
    number_acceptation = Ac(n, dict_aql[aql])[1]

    return (letter, n, number_acceptation)


# import json and draw subet of 80 sample and write a new json
path_json = "/Users/kimmontalibet/PycharmProjects/ocr-xtract/" + "data/quittances/annotation/sample1.json"
f = open(path_json)
data = json.load(f)


_, size_subsample, acceptance_threshold = get_num_sample_number_acceptation(len(data), "II", 2.5)

sub_data = random.sample(data, size_subsample)

with open('data/quittances/annotation/sub_sample1_to_check.json', 'w') as f:
    json.dump(sub_data, f)
