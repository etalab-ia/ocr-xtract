# Model card - LirIA - CNI

Source: model card paper :  https://arxiv.org/abs/1810.03993

## Model Details. 
*Basic information about the model.*

* Model developed by the Lab IA of Etalab 
* T1 2022
* Model type : Doctr (https://github.com/mindee/doctr) library for OCR and bounding box detection, XGBOOST for token classification (information extraction)
* Information about training algorithms, parameters, fairness: 
* constraints or other applied approaches, and features: for each word in document extracted with Doctr library, the features computed are the positions on 
  * the x and y axes (from doctr library)
  * the height and the width of the box (from doctr library )
  * dummy variable for each of a set predefined words, with value 1 if the word is present in the line (lines are computed with Jenks algorithm), 0 otherwise  
  * for each of a set predefined words, the relative positions of the example to the word in the given set characterized by angle and distance 
  * if the word is a date
  * if the word is composed of digits
  * if the word is the INSEE list of first names
  * if the word is the INSEE list of family names 
* Paper or other resource for more information
* License : not relevant as model not published, source code for training the model under MIT license
* Where to send questions or comments about the model: lab-ia@data.gouv.fr

## Intended Use. 
*Use cases that were envisioned during development.*

* Primary intended uses : extract key information (first name, family name and date of birth from pictures of scans of National French ID cards of the pre-2021 version )
* Secondary intended users: no other possible used 



## Factors. 
*Factors could include demographic or phenotypic groups, environmental conditions, technical attributes, or others listed in Section 4.3.*

* Relevant factors: relevant factors could be 
  * Quality of the picture of the scan 
  * if the picture is straight of pivoted (pivoted images may result in poor quality OCR and therefore poor quality information extraction)
* Evaluation factors

## Metrics. 
*Metrics should be chosen to reflect potential realworld impacts of the model.*

* Model performance measures: for each of the key information extracted, precision, recall, F1-Score. For the overall performance, average of these 3 indicators over the different labels to be extracted. 
* F1-score for the 3 extracted categories: 
  *"prenom":0.99,
  * "nom": 0.96, 
  * "date_naissance"1.0

* Decision thresholds
* Variation approaches

## Evaluation Data. 
*Details on the dataset(s) used for the quantitative analyses in the card.*

* Datasets : 20% random draw from the annotated datasets
A set of 400 documents were annotated using Label Studio. 
* Preprocessing: The images were loaded in Label Studio with the Doctr bounding boxes, and the annotators had to attribute the corresponding label to the boxes containing the pieces of information to be extracted (first names, family names, date of birth)


## Training Data. 

*May not be possible to provide in practice.*

When possible, this section should mirror Evaluation Data. If such detail is not possible, minimal allowable information should be provided here, such as details of the distribution over various factors in the training datasets.

* Datasets : 80% random draw from the annotated datasets
A set of 400 documents were annotated using Label Studio. The images were loaded in Label Studio with the Doctr bounding boxes, and the annotators had to attribute the corresponding label to the boxes containing the pieces of information to be extracted (first names, family names, date of birth)

## Quantitative Analyses

* Unitary results
* Intersectional results

## Ethical Considerations

## Caveats and Recommendations