# Model card - Pseudonymisation 

Source: model card paper :  https://arxiv.org/abs/1810.03993

## Model Details. 
*Basic information about the model.*

* Person or organization developing model: Lab IA, Etalab 
* Model date : last updated 3rd february 2020 
* Model type : Named Entity Recognition based on Flair embeddings 

* Information about training algorithms, parameters, fairness

![](9d09f94c17f240efe7b989f4713b04ee.png)

* constraints or other applied approaches, and features
* Paper or other resource for more information : [this presentation](https://speakerdeck.com/etalabia/psuedo-ce-20201128) and [this more general presentation]( https://speakerdeck.com/etalabia/psuedo-ce-20201128-general)
* License : MIT License
* Where to send questions or comments about the model : lab-ia@data.gouv.fr

## Intended Use. 
*Use cases that were envisioned during development.*

* Primary intended uses : Model used to pseudonymize administrative justice decisions (from the Conseil d'Etat), by replacing first names, family names and addresses by aliases
* Secondary intended users: other types of justice decisions, but note that the performance may be less good 
* Out-of-scope use cases: other types of texts 


## Factors. 
*Factors could include demographic or phenotypic groups, environmental conditions, technical attributes, or others listed in Section 4.3.*

* Relevant factors : mispelling, misuse of capital letters for first names, family names, location names. Uncomon names might have a smaller chance of detection 
* Evaluation factors : not relevant

## Metrics. 
*Metrics should be chosen to reflect potential realworld impacts of the model.*

* Model performance measures

![](59a4d1fccd4ac7bc05dd18976d61a831.png)

On the evaluation set : F-score de 92.98%

![](b4c71908f4da22987ac65567b2f65276.png)


* Decision thresholds
* Variation approaches: not relevant

## Evaluation Data. Details on the dataset(s) used for the quantitative analyses in the card.

* Datasets
* Motivation
* Preprocessing : tokenization 


## Training Data. 

*May not be possible to provide in practice.*

When possible, this section should mirror Evaluation Data. If such detail is not possible, minimal allowable information should be provided here, such as details of the distribution over various factors in the training datasets.

## Quantitative Analyses

* Unitary results
* Intersectional results

## Ethical Considerations

## Caveats and Recommendations
