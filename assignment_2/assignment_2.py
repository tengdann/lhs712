import os
import itertools
import numpy as np
import pandas as pd
from sklearn import naive_bayes, svm, neural_network
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
# from muffnn import MLPClassifier, MLPRegressor

# Create dataframe for model output
# curdir = r'C:\Users\dteng\Desktop\lhs712\assignment_2\dataset\unlabeled-test-data\Gastroenterology'
curdir = r'C:\Users\mrasianman3\Desktop\lhs712\assignment_2\dataset\unlabeled-test-data\Gastroenterology'
files = list()
for file in os.listdir(curdir):
    files.append(file)

files.sort()
df = pd.DataFrame(data = files, columns = ['Filename'])

# Preprocessing steps
train_data = load_files('assignment_2/dataset/train-data', load_content = True, shuffle = True, random_state = 42)
test_data = load_files('assignment_2/dataset/unlabeled-test-data', load_content = True, shuffle = False)
# train_labels = train_data.target

# count_vect = CountVectorizer(decode_error = 'ignore')
# train_counts = count_vect.fit_transform(train_data.data)
# tf_transformer = TfidfTransformer(use_idf = False)
# train_tf = tf_transformer.fit_transform(train_counts)

# test_counts = count_vect.transform(test_data.data)
# tf_transformer = TfidfTransformer(use_idf = False)
# test_tf = tf_transformer.fit_transform(test_counts)

cont = str(input('Train Naive Bayes classifier? [y/n]: '))

if cont.lower() == 'y':
    # Naive Bayes
    clfrNB = Pipeline(
        [
            ('vect', CountVectorizer(decode_error = 'ignore', strip_accents = 'unicode', lowercase = True)),
            ('tfidf', TfidfTransformer(use_idf = False)),
            ('clf', naive_bayes.MultinomialNB(fit_prior = False))
        ]
    )
    clfrNB = clfrNB.fit(train_data.data, train_data.target)
    predictedNB = clfrNB.predict(test_data.data)
    df['Labels_NB'] = [train_data.target_names[i] for i in predictedNB]
    print(df.head())
    df.to_csv('./assignment_2/predictions_NB.csv', columns = ['Filename', 'Labels_NB'], header = ['Filename', 'Label'],index = False)

cont = str(input('Train Logistic Regression classifier? [y/n]: '))

if cont.lower() == 'y':
    # Logistic Regressions?
    clfrLR = Pipeline(
        [
            ('vect', CountVectorizer(decode_error = 'ignore', strip_accents = 'unicode', lowercase = True)),
            ('tfidf', TfidfTransformer(use_idf = True)),
            ('clf', LogisticRegression(class_weight = 'balanced', multi_class = 'auto', verbose = 1))
        ]
    )
    clfrLR = clfrLR.fit(train_data.data, train_data.target)
    predictedLR = clfrLR.predict(test_data.data)
    df['Labels_LR'] = [train_data.target_names[i] for i in predictedLR]
    print(df.head())
    df.to_csv('./assignment_2/predictions_LR.csv', columns = ['Filename', 'Labels_LR'], header = ['Filename', 'Label'],index = False)

cont = str(input('Train Linear SVM classifier? [y/n]: '))

if cont.lower() == 'y':
    # Linear SVM
    clfrSVM = Pipeline(
        [
            ('vect', CountVectorizer(decode_error = 'ignore', strip_accents = 'unicode')),
            ('tfidf', TfidfTransformer(use_idf = True)),
            ('clf', svm.LinearSVC())
        ]
    )
    clfrSVM = clfrSVM.fit(train_data.data, train_data.target)
    predictedSVM = clfrSVM.predict(test_data.data)
    df['Labels_SVM'] = [train_data.target_names[i] for i in predictedSVM]
    print(df.head())
    df.to_csv('./assignment_2/predictions_SVM.csv', columns = ['Filename', 'Labels_SVM'], header = ['Filename', 'Label'], index = False)

# cont = str(input('Train Neural Network classifier? [y/n]: '))

# if cont.lower() == 'y':
#     # MLPClassifier
#     clfrNN = Pipeline(
#         [
#             ('vect', CountVectorizer(decode_error = 'ignore', strip_accents = 'unicode')),
#             ('tfidf', TfidfTransformer(use_idf = True)),
#             ('clf', MLPClassifier(hidden_units = (256, )))
#         ]
#     )
#     clfrNN = clfrNN.fit(train_data.data, train_data.target)
#     predictedNN = clfrNN.predict(test_data.data)
#     df['Labels_NN'] = [train_data.target_names[i] for i in predictedNN]
#     print(df.head())
#     df.to_csv('./assignment_2/predictions_NN.csv', columns = ['Filename', 'Labels_NN'], header = ['Filename', 'Label'], index = False)

cont = str(input('Train Random Forest classifier? [y/n]: '))

if cont.lower() == 'y':
    # Random Forest Classifier
    clfrRFC = Pipeline(
        [
            ('vect', CountVectorizer(decode_error = 'ignore', strip_accents = 'unicode')),
            ('tfidf', TfidfTransformer(use_idf = True)),
            ('clf', RandomForestClassifier(n_estimators = 4000, n_jobs = -1, verbose = 1))
        ]
    )
    clfrRFC = clfrRFC.fit(train_data.data, train_data.target)
    predictedRFC = clfrRFC.predict(test_data.data)
    df['Labels_RFC'] = [train_data.target_names[i] for i in predictedRFC]
    print(df.head())
    df.to_csv('./assignment_2/predictions_RFC.csv', columns = ['Filename', 'Labels_RFC'], header = ['Filename', 'Label'], index = False)

cont = str(input("Optimize NB model? [y/n]: "))

if cont.lower() == 'y':
    # Model selection?
    params_gsnb = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10),
    }

    gs_clfnb = GridSearchCV(clfrNB, params_gsnb, cv = 10, n_jobs = -1)
    gs_clfnb = gs_clfnb.fit(train_data.data, train_data.target)
    predictedGSNB = gs_clfnb.predict(test_data.data)
    df['Labels_GSNB'] = [train_data.target_names[i] for i in predictedGSNB]
    print(df.head())
    df.to_csv('./assignment_2/predictions_GSNB.csv', columns = ['Filename', 'Labels_GSNB'], header = ['Filename', 'Label'], index = False)

cont = str(input("Optimize LR model? [y/n]: "))

if cont.lower() == 'y':
    # Model selection?
    params_gslr = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__use_idf': (True, False),
        'clf__solver': ('newton-cg', 'sag', 'saga', 'lbfgs'),
        'clf__C': (1e-6, 1e-4, 1e-2, 1)
    }

    gs_clflr = GridSearchCV(clfrLR, params_gslr, cv = 10, n_jobs = -1)
    gs_clflr = gs_clflr.fit(train_data.data, train_data.target)
    predictedGSLR = gs_clflr.predict(test_data.data)
    df['Labels_GSLR'] = [train_data.target_names[i] for i in predictedGSLR]
    print(df.head())
    df.to_csv('./assignment_2/predictions_GSLR.csv', columns = ['Filename', 'Labels_GSLR'], header = ['Filename', 'Label'], index = False)

cont = str(input("Optimize SVM model? [y/n]: "))

if cont.lower() == 'y':
    # More model selection?
    tuned_parameters = [
        {
            'vect__ngram_range' : [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            'clf__C': [0.001, 0.01, 0.1, 1],
            'clf__loss': ('hinge', 'squared_hinge'),
            'clf__class_weight': ['balanced'],
            'clf__max_iter': (2000, 3000, 4000)
        },
    ]
    gs_clfrSVM = Pipeline(
        [
            ('vect', CountVectorizer(decode_error = 'ignore')),
            ('tfidf', TfidfTransformer(use_idf = False)),
            ('clf', svm.LinearSVC(verbose = 1))
        ]
    )
    gs_clfsvm = GridSearchCV(gs_clfrSVM, tuned_parameters, cv = 10, n_jobs = -1)
    gs_clfsvm = gs_clfsvm.fit(train_data.data, train_data.target)
    predictedGSSVM = gs_clfsvm.predict(test_data.data)
    df['Labels_GSSVM'] = [train_data.target_names[i] for i in predictedGSSVM]
    print(df.head())
    df.to_csv('./assignment_2/predictions_GSSVM.csv', columns = ['Filename', 'Labels_GSSVM'], header = ['Filename', 'Label'], index = False)

# cont = str(input("Optimize NN model? [y/n]: "))

# if cont.lower() == 'y':
#     # More model selection?
#     tuned_parameters = [
#         {
#             'vect__ngram_range' : [(1, 1), (1, 2)],
#             'tfidf__use_idf': (True, False),
#             'clf__learning_rate': ['constant', 'invscaling', 'adaptive'],
#             'clf__activation': ['logistic', 'relu', 'identity']
#         }
#     ]
#     gs_clfrNN = Pipeline(
#         [
#             ('vect', CountVectorizer(decode_error = 'ignore')),
#             ('tfidf', TfidfTransformer(use_idf = False)),
#             ('clf', MLPClassifier())
#         ]
#     )
#     gs_clfnn = GridSearchCV(gs_clfrNN, tuned_parameters, cv = 10, n_jobs = -1)
#     gs_clfnn = gs_clfnn.fit(train_data.data, train_data.target)
#     predictedGSNN = gs_clfnn.predict(test_data.data)
#     df['Labels_GSNN'] = [train_data.target_names[i] for i in predictedGSNN]
#     print(df.head())
#     df.to_csv('./assignment_2/predictions_GSNN.csv', columns = ['Filename', 'Labels_GSNN'], header = ['Filename', 'Label'], index = False)

cont = str(input('Optimize RFC model? [y/n]: '))

if cont.lower() == 'y':
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

    # Create the random grid
    random_grid = {
        'clf__n_estimators': n_estimators,
        'vect__ngram_range' : [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
    }
    gs_clfrrfc = Pipeline(
        [
            ('vect', CountVectorizer(decode_error = 'ignore', strip_accents = 'unicode')),
            ('tfidf', TfidfTransformer(use_idf = False)),
            ('clf', RandomForestClassifier(max_features = 'sqrt', verbose = True))
        ]
    )
    gs_clfrrfc = GridSearchCV(gs_clfrrfc, random_grid, cv = 10, n_jobs = -1)
    gs_clfrrfc = gs_clfrrfc.fit(train_data.data, train_data.target)
    predictedGSRFC = gs_clfrrfc.predict(test_data.data)
    df['Labels_GSRFC'] = [train_data.target_names[i] for i in predictedGSRFC]
    print(df.head())
    df.to_csv('./assignment_2/predictions_GSRFC.csv', columns = ['Filename', 'Labels_GSRFC'], header = ['Filename', 'Label'], index = False)