# python 2.x

import numpy as np
import sys

from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
sys.path.append("../tools/")

from tools.feature_format import featureFormat, targetFeatureSplit


def selectFeatures(features, labels, features_list, plot=False):
    """Apply SelectKBest to features and transform to 5 best features."""
    selector = SelectKBest(k=21)

    if plot:
        selector.fit(features, labels)
        h = len(features[0])
        scores = -np.log10(selector.pvalues_)

        plt.bar(range(h), scores)
        plt.xticks(range(h), features_list[1:], rotation='vertical')
        plt.show()

    else:
        selector.fit(features, labels)
        print(selector.scores_)
        print(selector.pvalues_)

    return selector.fit_transform(features, labels)


def selectFeatures2(features, labels, clf):
    selector = SelectKBest(k=21)
    pipe = Pipeline(steps=[('selector', selector), ('clf', clf)])
    grid_search = GridSearchCV(pipe, {'selector__k': range(1, 19), 'clf__n_neighbors': range(1, 11, 2),
                                      'clf__p': [1, 2]}, scoring='recall')

    grid_search.fit(features, labels)

    best_params = grid_search.best_params_

    print(best_params)

    return grid_search.best_estimator_

def computeFraction(all_emails, poi_emails):
    """Divide poi emails by all the emails sent or received by a person, and return quotient as float."""
    fraction = 0.
    if all_emails == 'NaN' or poi_emails == 'NaN':
        return 0
    fraction = float(poi_emails) / float(all_emails)
    return fraction


def computeCompensation(salary, bonus):
    """Add salary and bonus; handle NaN."""
    if salary == 'NaN':
        if bonus == 'NaN':
            return 'NaN'
        else:
            return bonus
    if bonus == 'NaN':
        return salary
    else:
        return salary + bonus


def createFeatures(data_dict):
    for person in data_dict.values():
        all_to = person['to_messages']
        all_from = person['from_messages']
        to_poi = person['from_this_person_to_poi']
        from_poi = person['from_poi_to_this_person']
        salary = person['salary']
        bonus = person['bonus']

        fraction_to_poi = computeFraction(all_to, to_poi)
        fraction_from_poi = computeFraction(all_from, from_poi)
        total_compensation = computeCompensation(salary, bonus)

        person['fraction_to_poi'] = fraction_to_poi
        person['fraction_from_poi'] = fraction_from_poi
        person['total_compensation'] = total_compensation

    return data_dict


def scaleFeatures(features):
    """Scale the features from (0, 1)."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(features)


def createNB(features, labels):
    """Create and return a clf based on sklearn Naive Bayes."""
    clf = GaussianNB()
    clf.fit(features, labels)
    return clf


def createKNN(features, labels):
    """Create and return a clf based on sklearn K-Nearest Neighbors."""
    clf = KNeighborsClassifier(p=1, n_neighbors=5)
    clf.fit(features, labels)
    return clf


def createLinearSVC(features, labels):
    """Create and return a clf based on sklearn Linear SVM classifier."""
    clf = LinearSVC()
    clf.fit(features, labels)
    return clf

