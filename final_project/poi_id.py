#!/usr/bin/python

import pickle
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [
                 'poi',
                 'salary',
                 'deferral_payments',
                 'total_payments',
                 'loan_advances',
                 'bonus',
                 'restricted_stock_deferred',
                 'deferred_income',
                 'total_stock_value',
                 'expenses',
                 'exercised_stock_options',
                 'other',
                 'long_term_incentive',
                 'restricted_stock',
                 'director_fees',
                 'to_messages',
                 'from_poi_to_this_person',
                 'from_poi_perc',
                 'from_messages',
                 'from_this_person_to_poi',
                 'to_poi_perc',
                 'shared_receipt_with_poi'
                 ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
data_dict_df = pd.DataFrame(data_dict).T.replace("NaN", np.nan) # Also create dataframe

### Compute some terms on the dataset (change False to True to see the results)
print "--------------------------------"
print "There is a total of", len(data_dict), "employee rows"
print ""
pois = 0
for employee, features in data_dict.iteritems():
    if features['poi'] == 1:
        pois += 1
print "There is a total of", pois, "POIs"
print ""

print "These are the features: ", features_list
print ""

print "This is the list of features with NaN values:"
print pd.DataFrame(data_dict).T.replace("NaN", np.nan).isnull().sum()
print ""
print "--------------------------------"

### Task 2: Remove outliers

### Get histogram of each feature (change False to True to see the results)
if False:
    # Plot features on a 3 x 2 grid
    i = 1
    plt.figure(1)
    num_plot = 0
    while i < (len(features_list)):
        plt.subplot(2, 3, i - (num_plot * 6))
        plt.hist(data_dict_df[features_list[i]].dropna().astype(float))
        plt.title(features_list[i])
        plt.grid(True)

        if abs(i / 6) > num_plot:
            plt.tight_layout()
            plt.savefig('figure' + str(num_plot) + ".png", dpi=600, pad_inches = 'tight')
            #plt.show()
            plt.close()
            plt.figure(1)
            num_plot += 1

        i += 1
    plt.tight_layout()
    plt.savefig('figure' + str(num_plot) + ".png", dpi=600)
    #plt.show()

## Get Top3 values by salary or bonus
Top3Salary = data_dict_df['salary'].sort_values(ascending = False)[:3]
print "Top3 Salary values are:"
print Top3Salary
print ""

Top3Bonus = data_dict_df['bonus'].sort_values(ascending = False)[:3]
print "Top3 Bonus values are:"
print Top3Salary
print ""

## Get all NaN row values
all_NaNs = data_dict_df.drop('poi', 1).isnull().all(1)
print "These are the rows with all values NaNs:", all_NaNs[all_NaNs == True].index
print ""

## Remove all the outliers found
data_dict.pop("TOTAL")
data_dict.pop("LOCKHART EUGENE E")

## Also remove another one found by observation because is near to total in the PDF
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

## Create new features
# to_poi_perc as a percentage of total mails
for employee, features in data_dict.iteritems():
	if features['from_this_person_to_poi'] == "NaN" or features['from_messages'] == "NaN":
		features['to_poi_perc'] = "NaN"
	else:
		features['to_poi_perc'] = float(features['from_this_person_to_poi']) \
                                                         / float(features['from_messages'])

# from_poi_perc as a percentage of total mails
for employee, features in data_dict.iteritems():
	if features['from_poi_to_this_person'] == "NaN" or features['to_messages'] == "NaN":
		features['from_poi_perc'] = "NaN"
	else:
		features['from_poi_perc'] = float(features['from_poi_to_this_person']) \
                                                         / float(features['to_messages'])

### Impute missing email features to mean
email_features = ['to_messages',
	              'from_poi_to_this_person',
                      'from_poi_perc',
	              'from_messages',
	              'from_this_person_to_poi',
	              'to_poi_perc',
	              'shared_receipt_with_poi']
from collections import defaultdict
email_feature_sums = defaultdict(lambda:0)
email_feature_counts = defaultdict(lambda:0)

for employee, features in data_dict.iteritems():
	for ef in email_features:
		if features[ef] != "NaN":
			email_feature_sums[ef] += features[ef]
			email_feature_counts[ef] += 1

email_feature_means = {}
for ef in email_features:
	email_feature_means[ef] = float(email_feature_sums[ef]) / float(email_feature_counts[ef])

for employee, features in data_dict.iteritems():
	for ef in email_features:
		if features[ef] == "NaN":
			features[ef] = email_feature_means[ef]

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

## Get importance of features using SelectKBest and score in DT
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier

KBest = SelectKBest(k='all')
KBest.fit(features, labels)

clf = DecisionTreeClassifier()
clf.fit(features, labels)

KBest_score = pd.Series(KBest.scores_)
clf_score = pd.Series(clf.feature_importances_)
KBest_table = pd.concat([KBest_score, clf_score], axis=1)
KBest_table.columns = ['scoreSKB', 'scoreDT']
KBest_table.index = features_list[1:]
print KBest_table.sort_values(by = 'scoreSKB', ascending = False)
print ""

features_selected = pd.Series(KBest_table.sort_values(by = 'scoreSKB', ascending = False).index[:])
features_selected = pd.Series([features_selected[i] for i in [0, 1, 2, 3, 4]]) # Selected features
features_selected = pd.Series("poi").append(features_selected, ignore_index=True)
print "Then the most important features selected are:", '\n', features_selected[1:]
print ""
print "--------------------------------"

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

## Get only selected features and split train/test
data = featureFormat(my_dataset, features_selected, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

## Create pipeline
scaler = MinMaxScaler()
clf_dt = DecisionTreeClassifier()
clf_svc = SVC()
clf_kn = KNeighborsClassifier()

steps = [
    #Preprocessing
    #('min_max_scaler', scaler),

    # Classifier
    ('dtc', clf_dt)
    #('svc', clf_svc)
    #('knc', clf_kn)
    ]

pipeline = Pipeline(steps)

## Do a Grid Search
GridParams = dict(
                  dtc__criterion=['gini', 'entropy'],
                  dtc__max_depth=[None, 5, 10,15,20, 30, 40],
                  dtc__min_samples_split=[2, 5, 10, 20],
                  dtc__class_weight=[None, 'balanced'],
                  dtc__random_state=[42],
                  dtc__splitter = ['best','random'],
                  dtc__max_leaf_nodes = [5,10,20, 30, 40]
                  #svc__C=[0.1, 1, 10, 100, 1000],
                  #svc__kernel=['linear', 'rbf'],
                  #svc__gamma=[0.001, 0.0001],
                  #svc__class_weight=[None, 'balanced']
                  #knc__n_neighbors=[1, 2, 3, 4, 5],
                  #knc__leaf_size=[1, 10, 30, 60]
                  )

# Cross-validation for parameter tuning in grid search
sss = StratifiedShuffleSplit(
    n_splits = 10,
    test_size = 0.3,
    random_state = 42
    )

clfGS = GridSearchCV(pipeline, param_grid = GridParams, cv = sss, scoring = "f1")
clfGS.fit(features_train, labels_train)

## Finally choose the best model for predict
clf = clfGS.best_estimator_
print ""
print "The best parameters are:"
print clfGS.best_params_
print ""
print "--------------------------------"

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.metrics import classification_report

labels_predictions = clf.predict(features_test)

# Print classification report (focus on precision and recall)
report = classification_report( labels_test, labels_predictions )
print(report)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)