# NLP_PMQ_classification
Natural Language Processing project that will categorize PMQ student responses automatically.

# PMQ_classify

This script will classify a set of uncategorized student response data to the RD, SMDS, and DMSS probes of the Physics Measurement Questionnaire (PMQ).  

A user can either use our pretrained model to classify their unlabeled data, or train a new classifier on labeled training data that they may have. This can be specified by use of different flags at the command line. These flags include:

-t Training Flag: specify a path to a .xlsx file that contains labeled training data from the PMQ
-c Classify Flag: specify a path to a .xlsx file that contains unlabeled PMQ data that needs to be classified
-f Figure Flag: use this flag followed by a file name for a bar graph figure that shows the proportion of each class for each probe in your newly classified data
-a Aggregate Results Flag: use this flag followed by a .xlsx file name for a spreadsheet containing the raw counts of classes for each probe, as well as their proportion to the entire dataset for that probe.

Follow any of these flags with the appropriate file name to execute the script. The only required flag is "-c" so the program has something to classify. If no "-t" flag is provided, then the system assumes you want to use our pretrained model to classify your data. 

An example of executing this script using training data, but no aggregate results or figure output desired, would be:

`python3 PMQ_classify.py -t training_data.xlsx -c unlabeled_data.xlsx`

The script will automatically output a raw results spreadsheet that contains the original student response, their multiple choice answer, and the corresponding predicted class for each response in each probe.

## Note on data file format  
This sytem expects your PMQ data to be in a .xlsx file with the following columns for each probe in order to classify it:
[probe]_choice (The student's multiple choice selection)
[probe]_explanation (The student's original written free response)

For training data, your file needs the two columns listed above as well as:
[probe]_Code (The predetermined classification of the student response)

## Note on pretrained model  
Both the pretrained logistic regression model and the pre-fit tf-idf vectorizer come in the form of .pickle files. These files serialize the weight vector coefficients and vectroizer features so they can be used without needing to download the training data. Pickle files are notoriously clunky. There may be problems when loading them onto a different platform from the one they were created on.  

At this time, though, in order to classify your own data without providing a training file, you must have the following files (found in this repository):  
* RD_vector.pickle  
* RD_logit.pickle  
* SMDS_vector.pickle  
* SMDS_logit.pickle  
* DMSS_vector.pickle  
* DMSS_logit.pickle  

## Notes from Ben's initial test on June 15:
* I needed to run the following (in python) to download the required nltk info:
```
import nltk
nltk.download('averaged_perceptron_tagger') 
nltk.download('wordnet')
```
* Has issues with blank entries...
