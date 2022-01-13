from jenkspy import JenksNaturalBreaks
import numpy as np

def goodness_of_variance_fit(array, classes):
    # get the break points
    jen = JenksNaturalBreaks(classes)
    jen.fit(array)

    # sum of squared deviations from array mean
    sdam = np.sum((array - array.mean()) ** 2)

    # do the actual classification
    groups = jen.group(array)

    # sum of squared deviations of class means
    sdcm = sum([np.sum((group - group.mean()) ** 2) for group in groups])

    # goodness of variance fit
    gvf = (sdam - sdcm) / sdam

    return gvf

def get_optimal_nb_classes(y):
    gvf = 0.0
    nclasses = 2
    while gvf < .9999:
        gvf = goodness_of_variance_fit(y, nclasses)
        if gvf < .9999:
            nclasses += 1
        if gvf == 1.0:
            nclasses -= 1
        if nclasses > int(len(y) / 2):
            break
    return nclasses
