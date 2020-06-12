"""
PMQ_classify.py

This script will classify a set of uncategorized student response data
to the RD, SMDS, and DMSS probes of the PMQ. These responses will be 
classified based on a pre-trained machine learning classifier using data
from multiple semesters of CU Boulder student PMQ data.

The data is passed to the script using a command line flag '-d' and then the
name of the data file.

The script will output an excel spreadsheet named "pmq_predictions.xlsx" that
will contain the student's multiple choice answer, their written free response,
and the system's prediction of set-like, point-like, or undefined. 

Also, the user has the option of outputting a bar graph figure showing the proportion
of set-like, point-like, and undefined responses for each probe. The name for this figure 
can be passed to the script using the '-f' flag and then the desired name of the
figure. If no '-f' flag is given, no figure will be produced.

The user can also output and aggregated results spreadsheet. This contains the 
raw number of set-like, point-like, and undefined responses for each probe, as well
as their proportion to the overall data set for that probe, in a spreadsheet format.
The name of this spreadsheet can be passed to the script using the '-a' flag and
then the desired name of the file. If no '-a' flag is given, no spreadsheet will
be produced.

~Author: Joe Wilson~
Last Updated: 06/04/2020
"""

import numpy as np
import pandas as pd
import sys 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import matplotlib.pyplot as plt
import argparse
import time
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

def main():
    #Create parser object to use command line flags
    parser = argparse.ArgumentParser()
    
    #Define flags for the unlabeled data file, the training data file, bar graph figure name, and aggregated
    #results optional output.
    parser.add_argument("-t", "--training", help="Path to Data File to Train On")
    parser.add_argument("-c", "--classify", help="Path to Data File to Classify")
    parser.add_argument("-f", "--figname", help="Figure Filename")
    parser.add_argument("-a", "--aggresults", help="Aggregate Results Filename") #Must be .xlsx extension

    args = parser.parse_args()
    
    classify_file = args.classify
    train_file = args.training
    
    #Check to make sure a data file was provided.
    if classify_file == None:
        print("No unclassified data file entered. Use -d flag and data file path at command line.")
    else:
        #Check to make sure data is in correct file format
        if classify_file[-5:] != '.xlsx':
            print("Data file must be in .xlsx format")
        else:
            #Checks for training data file
            if train_file != None:
                if train_file[-5:] != '.xlsx':
                    print("Training data file must be in .xlsx format")
                else:
                    #Read this training data as separate dataframe
                    pmq_train = pd.read_excel(train_file)
                    
                    #Clean the text and grab the labels of the training data
                    train_text = clean_text(pmq_train)
                    train_label = get_labels(pmq_train)
                    
                    #Sometimes there are missing labels, so we scrub those out
                    train_label, train_text = clean_nans(train_label, train_text)
            
            print(f'Classifying data from {classify_file}...')
            
            #Read in the excel file, clean the data, vectorize the text responses,
            #and classify them.
            pmq = pd.read_excel(classify_file)
            text_data = clean_text(pmq)
            
            #If there is training data provided, we'll create a vectorizer based
            #on that data, train on it, then classify the unlabeled data.
            if train_file != None:
                train_vector, classify_vector = train_vectorize(train_text, text_data)
                clf_results = train_classify(classify_vector, train_vector, train_label)
            #If no training data is provided, just use the pretrained model
            else:
                vector_data = vectorize(text_data)
                clf_results = classify(vector_data)

            #Create new dataframe containing classifications
            #print(vector_data)
            pmq['RD_pred'] = clf_results['RD']
            pmq['SMDS_pred'] = clf_results['SMDS']
            pmq['DMSS_pred'] = clf_results['DMSS']

            pmq_pred = pmq[['RD_choice', 'RD_explanation', 'RD_pred', 
                            'SMDS_choice', 'SMDS_explanation', 'SMDS_pred', 
                            'DMSS_choice', 'DMSS_explanation', 'DMSS_pred']]

            
            #Create ExcelWriter object so we can customize the spreadsheet output
            print('Exporting raw results to pmq_predictions.xlsx...')
            writer = pd.ExcelWriter("pmq_predictions.xlsx", engine='xlsxwriter')
            pmq_pred.to_excel(writer, sheet_name='Sheet1')
            workbook  = writer.book
            worksheet = writer.sheets['Sheet1']

            #Make the column width of the spreadsheet about equal to the length
            #of the title of each column
            for col in range(0,9):
                width = len(pmq_pred.columns[col])
                worksheet.set_column(col+1,col+1,width+2)
            writer.save()

            #If the user wants a bargraph, pass the given name to the 'pred_bar_chart()' function.
            if args.figname != None:
                print(f'Exporting results figure to {args.figname}...')
                pred_bar_chart(clf_results, args.figname)

            #If the user wants an aggregated results spreadsheet, pass the given name
            #to the 'results_spreadsheet()' function.
            if args.aggresults != None:
                #Make sure they supplied the right file format
                if args.aggresults[-5:] != '.xlsx':
                    print('Aggregate Results file must be .xlsx')
                    print('No aggregate results file created.')
                else:
                    print(f'Exporting aggregate results to {args.aggresults}...')
                    results_spreadsheet(clf_results, args.aggresults)


def nltk2wn_tag(nltk_tag):
    """
    Taken from https://simonhessner.de/lemmatize-whole-sentences-with-python-and-nltks-wordnetlemmatizer/
    
    Keyword Arguments
    nltk_tag: This is a part of speech tag from NLTK's 'pos_tag()' function.
    
    Output
    This function takes the NLTK part of speech tag and converts it to the necessary
    part of speech tag that the lemmatizer function requries.
    """
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def clean_text(pmq):
    """
    Keyword Arguments
    pmq: Pandas DataFrame containing data collected from PMQ responses
        to RD, SMDS, and DMSS probes
        Data comes in form of a '[probe]_choice' column for the students'
        multiple choice answers and a '[probe]_explanation' for the students'
        written free response. 
        
    Output
    probe_clean: A dictionary named probe_clean that contains a list of 
                cleaned written responses. This means the text has been scrubbed
                of punctuation, converted to lower case, and the words lemmatized.
                The student's multiple choice answer is also appended to the end of
                their response.
    """
    probe_names = ['RD', 'SMDS', 'DMSS']
    probe_clean = {'RD': [], 'SMDS': [], 'DMSS': []}
    
    for probe in probe_names:
        text = '{}_explanation'.format(probe)
        mc = '{}_choice'.format(probe)
        probe_text = list(pmq[text])
        probe_mc = list(pmq[mc])

        #Create lemmatizer object using NLTK's WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        
        for i,response in enumerate(probe_text):
            #Scrub punctuation using regex command
            response = re.sub(r'[^\w\s]','',response)
            
            #Make lowercase
            response = response.lower()
            words = response.split()
            
            #Tag the part of speech of each word in order for them
            #to be lemmatized properly.
            tagged = nltk.pos_tag(words)
            for k,tag in enumerate(tagged):
                #Grab part of speech tag (sometimes NLTK cannot find a 
                #part of speech, and returns None)
                new_tag = nltk2wn_tag(tag[1])
                
                #Lemmatize word based on part of speech
                if new_tag != None:
                    word = lemmatizer.lemmatize(words[k], new_tag)
                    words[k] = word
                else:
                    word = lemmatizer.lemmatize(words[k])
                    words[k] = word
            
            #Join the words back into one string by spaces
            probe_text[i] = ' '.join(words)
            
            #Append multiple choice answer to response using 'mc_[choice]' format
            if probe_mc[i] == 'A':
                probe_text[i] += ' mc_a'
            elif probe_mc[i] == 'B':
                probe_text[i] += ' mc_b'
            elif probe_mc[i] == 'C':
                probe_text[i] += ' mc_c'

        probe_clean[probe] = probe_text
        
    return probe_clean

def get_labels(pmq):
    """
    Keyword Arguments
    pmq: Pandas DataFrame containing data collected from PMQ responses
        to RD, SMDS, and DMSS probes
        This function is designed to grab the predetermined labels from
        the student data. It should be stored in a column called '[probe]_Code'
        
    Output
    label_dict: A dictionary containing a list for each probe that contains the
                true labels (P, S, or U) for each student response in the training
                data.
    """
    probe_names = ['RD', 'SMDS', 'DMSS']
    label_dict = {'RD': [], 'SMDS': [], 'DMSS': []}
    
    for probe in probe_names:
        #Grab the column of labels (also called 'Codes')
        codes = f'{probe}_Code'
        probe_labels = list(pmq[codes])
        
        for i in range(len(probe_labels)):
            #Check to see if a label exists
            if str(probe_labels[i]) != 'nan':
                #True labels are typically broken down into more nuanced
                #categories. But we only care about the prefix 'S', 'P', or 'U'
                label = probe_labels[i][0]
                
                #Sometimes data can be classified as 'Mixed', but we will
                #count this as 'Undefined' for our purposes
                if label == 'M':
                    probe_labels[i] = 'U'
                else:
                    probe_labels[i] = label
                    
        label_dict[probe] = probe_labels

    return label_dict

def clean_nans(labels, text):
    """
    Keyword Arguments
    labels: A label dictionary output from 'get_labels()'
    text: Text dictionary output from 'clean_text()'
    
    Output
    labels: The dictionary of true labels scrubbed of any missing labels.
    text: The dictionary of cleaned student text that has been scrubbed of any
          response that's corresponding label was missing
    """
    probe_names = ['RD', 'SMDS', 'DMSS']
    
    for probe in probe_names:
        clean_label = []
        indices = []
        for i,label in enumerate(labels[probe]):
            if (label == 'S') or (label == 'P') or (label == 'U'):
                clean_label.append(label)
            else:
                indices.append(i)

        for i in range(len(indices)-1,-1, -1):
            text[probe].remove(text[probe][indices[i]])
        
        labels[probe] = clean_label
        
    return labels, text

def vectorize(text_dict):
    """
    Keyword Arguments
    text_dict: Dictionary of cleaned text responses output from 'clean_text()'
    
    Output
    vector_dict: Dictionary containing a 2D array for each probe.
                 Each array contains the student responses for that
                 probe converted into feature vectors using tf-idf
                 vectorization. This output is a numerical representation
                 of the student response.
    """
    probe_names = ['RD', 'SMDS', 'DMSS']
    vector_dict = {'RD': [], 'SMDS': [], 'DMSS': []}
    for probe in probe_names:
        #This loads a pre-fitted vectorizer that converts student responses
        #to feature vectors based on the features from our training data
        filename = '{}_vector.pickle'.format(probe)
        vec_file = open(filename, 'rb')
        vectorizer = pickle.load(vec_file)
        
        #This line actually transforms the text responses to feature vectors
        vector_dict[probe] = vectorizer.transform(text_dict[probe])
        
    return vector_dict

def train_vectorize(fit_text, text_dict):
    """
    Keyword Arguments
    fit_text: Text training data. Used to fit the vectorizer
    text_dict: Unlabeled text data from 'clean_text()'
    
    Output
    training_vector: Dictionary of training data that has been vectorized
    classify_vector: Dictionary of unlabeled data that has been vectorized
    
    This function mirrors 'vectorize()' but it fits a brand new vectorizer based
    on the training data.
    """
    probe_names = ['RD', 'SMDS', 'DMSS']
    
    #Dictionary of vectorized training data
    training_vector = {'RD': [], 'SMDS': [], 'DMSS': []}
    
    #Dictionary of vectorized unlabeled data to classify
    classify_vector = {'RD': [], 'SMDS': [], 'DMSS': []}
    
    #Initialize vectorizer object
    vectorizer = TfidfVectorizer(lowercase=True, stop_words=stopwords.words('english'))
    
    for probe in probe_names:
        #'fit_transform' fits the vectorizer and then transforms the training text data to
        #feature vectors using the fitted vectorizer.
        training_vector[probe] = vectorizer.fit_transform(fit_text[probe])
        
        #'transform' uses the fitted vectorizer from above to transform the unlabeled text data
        #to feature vectors.
        classify_vector[probe] = vectorizer.transform(text_dict[probe])
    
    return training_vector, classify_vector

def prob_score(probs):
    '''
    Keyword Arguments
    probs: 2D array or list containing probabilities for point, set, and
           undefined for each test response output from our classifier.
    
    Output
    scores: A 2D array containing a score value for set, point, and undefined
            classifications for each response. This score is different from the
            raw probability as it measures how different the point and set 
            probabilities are. Only if a response's set and point probabilities
            are hardly distinguishable will it be classified as undefined.
    '''
    
    scores = []
    
    for proba in probs:
        p = proba[0]
        s = proba[1]
        
        #The p_score is just the difference between point and set probabilities
        p_score = p-s
        #The s_score is the opposite of p_score.
        s_score = s-p
        u_score = abs(p_score)
        new_score = [p_score, s_score, u_score]
        scores.append(new_score)
        
    return scores

def classify(vector_dict):
    """
    Keyword Arguments
    vector_dict: Dictionary containing feature vectors for each probe. This is
                 the output of 'vectorize()'
                 
    Output
    classify_dict: Dictionary containing a list of classifications for each probe.
                   This is the point, set, or undefined categorization for each student
                   response.
    """
    probe_names = ['RD', 'SMDS', 'DMSS']
    
    #These thresholds constitute what prob_score is needed for 
    #a student response to be classified as point or set. The
    #specific values were found from the training data.
    thresholds = {'RD': 0.2, 'SMDS': 0.63, 'DMSS': -0.33}
    classify_dict = {'RD': [], 'SMDS': [], 'DMSS': []}
    
    for probe in probe_names:
        #This loads the pre-trained classifier
        filename = '{}_logit.pickle'.format(probe)
        model_file = open(filename, 'rb')
        model = pickle.load(model_file)
        
        #This uses the pre-trained classifier to output classification
        #probabilities for the new data
        predicted = model.predict_proba(vector_dict[probe])
        
        #This assigns a prob_score for set,point, and undefined for each 
        #of the student responses based on their probabilities
        ncu_scores = prob_score(predicted)
        
        for score_arr in ncu_scores:
            p_score = score_arr[0]
            s_score = score_arr[1]
            
            #If the score for 'point-like' is over the appropriate threshold and 
            #the score for 'set-like' is not, then it is sufficiently point-like and
            #is classified as such.
            if (p_score > thresholds[probe]) and (s_score < thresholds[probe]):
                classify_dict[probe].append('P')
            #This is the same as above but for 'set-like'
            elif (p_score < thresholds[probe]) and (s_score > thresholds[probe]):
                classify_dict[probe].append('S')
            #If the response is not sufficiently point-like or set-like, it is undefined.
            else:
                classify_dict[probe].append('U')
                
    return classify_dict

def train_classify(classify_vector, train_vector, train_label):
    """
    Keyword Arguments
    classify_vector: Dictionary of vectorized, unlabeled data to be classified. Output from 'train_vectorize()'
    train_vector: Dictionary of vectorized training data. Used to train a new classifier. Output from 'train_vectorize()'
    train_label: Dictionary of corresponding training labels. Output from 'get_labels()' and 'clean_nans()'
    
    Output
    classify_dict: Dictionary of predictions of unlabeled data. Dictionary contains a list for each probe that has
                   corresponding 'S', 'P', or 'U' classifications for each unlabeled student response.
    """
    probe_names = ['RD', 'SMDS', 'DMSS']
    #Dictionary of final predictions for each response for each probe
    classify_dict = {'RD': [], 'SMDS': [], 'DMSS': []}
    
    #Create a logistic regression model. This is passed to OneVsRestClassifier since we are predicting
    #more than 2 classes
    log_clf = OneVsRestClassifier(LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000, C=1.0))
    
    for probe in probe_names:
        #Fit logistic regression model on vectorized training data
        log_clf.fit(train_vector[probe], train_label[probe])
        
        #Predict classes using trained model
        preds = list(log_clf.predict(classify_vector[probe]))
        classify_dict[probe] = preds
                
    return classify_dict

def pred_bar_chart(pred_dict, filename):
    """
    Keyword Arguments
    pred_dict: Dictionary of predictions for each student response in
               each probe. This is the output of classify().
    filename: The name of the file output given as a command line argument with
              '-f' flag.
              
    Output
    Bar graph figure showing the proportion of each class predicted for each probe.
    """
    fig,ax = plt.subplots(1,3,figsize=(15,5), sharey=True)
    classes = ['S', 'P', 'U']
    for i,probe in enumerate(pred_dict):
        x_pos = range(3)
        y_pos = [(pred_dict[probe].count(c)) / len(pred_dict[probe]) for c in classes]
        ax[i].set_ylim(0,1)
        ax[i].bar(x_pos, y_pos, tick_label=['Set-Like', 'Point-Like', 'Undefined'], edgecolor='white')
        ax[i].set_ylabel('Proportion')
        ax[i].set_title(f'{probe} Probe')
    plt.suptitle('Distribution of Classes')
    plt.savefig(filename, bbox_inches='tight')    
    #plt.show()
    
def results_spreadsheet(pred_dict, filename):
    """
    Keyword Arguments
    pred_dict: Dictionary of predictions for each student response in
               each probe. This is the output of classify().
    filename: The name of the file output given as a command line argument with
              '-a' flag.
              
    Output
    Excel spreadsheet containing the raw number of classes predicted for each probe
    and their proportion to the overall dataset for that probe.
    """
    results_df = pd.DataFrame(index = ['Set-Like', 'Point-Like', 'Undefined'])
    classes = ['S', 'P', 'U']
    for i,probe in enumerate(pred_dict):
        column_count = [pred_dict[probe].count(c) for c in classes]
        column_prop = [(pred_dict[probe].count(c)) / len(pred_dict[probe]) for c in classes]
        results_df[f'{probe}_Count'] = column_count
        results_df[f'{probe}_Proportion'] = column_prop
        
    #Create ExcelWriter object so we can customize the column width of spreadsheet.
    writer_agg = pd.ExcelWriter(filename, engine='xlsxwriter')
    results_df.to_excel(writer_agg, sheet_name='Sheet1')
    workbook  = writer_agg.book
    worksheet = writer_agg.sheets['Sheet1']
    
    #Make width of the columns about the size of the column name.
    for col in range(0,6):
        width = len(results_df.columns[col])
        worksheet.set_column(col+1,col+1,width+2)

    writer_agg.save()
    
    
if __name__ == '__main__':
    main()