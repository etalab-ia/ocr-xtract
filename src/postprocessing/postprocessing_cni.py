import re

def clean_date(x):
    if type(x) == list:
        output = '.'.join(x)

        # replace : par .  , remove letters
        output = re.sub("[^0123456789\.]", '', output.replace(":", ".").replace(" ", "").replace("-", "."))
        # remove leading .
        output = re.sub(r'^[\D]*', '', output)
        output = output.replace("..", ".")

        liste = output.split(".")
        if len(liste) > 3:
            liste = [x for x in liste if len(x) > 1]
            liste = liste[-3:]
        if len(liste) == 3:
            if len(liste[-1]) > 4:
                liste[-1] = liste[-1][-4:]
            if len(liste[-2]) > 2:
                liste[-2] = liste[-2][-2:]
            if len(liste[-3]) > 2:
                liste[-3] = liste[-3][-2:]
            if len(liste[-2]) == 1:
                liste[-2] = "0" + liste[-2]
            if len(liste[-3]) == 1:
                liste[-3] = "0" + liste[-3]
            if int(liste[-3][0]) > 3:
                liste[-3] = "0" + liste[-3][1]
            if int(liste[-2][0]) > 1:
                liste[-2] = "0" + liste[-2][1]

        output = [".".join(liste)]

    else:
        output = x
    return output


def clean_names(x):
    output = []
    temp = ""
    for i in x:
        if i.lower() in ["de", "le", "du", "d'", "l'"]:
            if i in ["d'", "l'"]:
                temp = i
            else:
                temp = i + " "
        else:
            i = i.replace("Nom:", "").strip()
            output.append(temp + i)
            temp = ""
    return output