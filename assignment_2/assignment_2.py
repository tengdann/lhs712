import os
import numpy as np
import pandas as pd
from sklearn import naive_bayes, svm
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# Create dataframe for model output
curdir = r'C:\Users\dteng\Desktop\lhs712\assignment_2\dataset\unlabeled-test-data\Gastroenterology'
# curdir = r'C:\Users\mrasianman3\Desktop\lhs712\assignment_2\dataset\unlabeled-test-data\Gastroenterology'
files = list()
for file in os.listdir(curdir):
    files.append(file)

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
            ('vect', CountVectorizer(decode_error = 'ignore', strip_accents = 'unicode')),
            ('tfidf', TfidfTransformer(use_idf = False)),
            ('clf', naive_bayes.MultinomialNB())
        ]
    )
    clfrNB = clfrNB.fit(train_data.data, train_data.target)
    predictedNB = clfrNB.predict(test_data.data)
    df['Labels_NB'] = [train_data.target_names[i] for i in predictedNB]
    print(df.head())
    df.to_csv('./assignment_2/predictions_NB.csv', columns = ['Filename', 'Labels_NB'], header = ['Filename', 'Label'],index = False)


cont = str(input('Train Linear SVM classifier? [y/n]: '))

if cont.lower() == 'y':
    # Linear SVM
    clfrSVM = Pipeline(
        [
            ('vect', CountVectorizer(decode_error = 'ignore', strip_accents = 'unicode')),
            ('tfidf', TfidfTransformer(use_idf = False)),
            ('clf', svm.SVC(kernel = 'linear', C = 0.1))
        ]
    )
    clfrSVM = clfrSVM.fit(train_data.data, train_data.target)
    predictedSVM = clfrSVM.predict(test_data.data)
    df['Labels_SVM'] = [train_data.target_names[i] for i in predictedSVM]
    print(df.head())
    df.to_csv('./assignment_2/predictions_SVM.csv', columns = ['Filename', 'Labels_SVM'], header = ['Filename', 'Label'], index = False)

cont = str(input('Train Neural Network classifier? [y/n]: '))

if cont.lower() == 'y':
    # MLPClassifier
    clfrNN = Pipeline(
        [
            ('vect', CountVectorizer(decode_error = 'ignore', strip_accents = 'unicode')),
            ('tfidf', TfidfTransformer(use_idf = False)),
            ('clf', MLPClassifier())
        ]
    )
    clfrNN = clfrNN.fit(train_data.data, train_data.target)
    predictedNN = clfrNN.predict(test_data.data)
    df['Labels_NN'] = [train_data.target_names[i] for i in predictedNN]
    print(df.head())
    df.to_csv('./assignment_2/predictions_NN.csv', columns = ['Filename', 'Labels_NN'], header = ['Filename', 'Label'], index = False)

cont = str(input('Train Random Forest classifier? [y/n]: '))

if cont.lower() == 'y':
    # Random Forest Classifier
    clfrRFC = Pipeline(
        [
            ('vect', CountVectorizer(decode_error = 'ignore', strip_accents = 'unicode')),
            ('tfidf', TfidfTransformer(use_idf = False)),
            ('clf', RandomForestClassifier(n_estimators = 1000, n_jobs = -1, random_state = 25, verbose = 1))
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

cont = str(input("Optimize SVM model? [y/n]: "))

if cont.lower() == 'y':
    # More model selection?
    tuned_parameters = [
        {
            'vect__ngram_range' : [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            'clf__kernel': ['rbf'],
            'clf__gamma': [1e-3, 1e-4, 1e-2, 0.1, 1],
            'clf__C': [0.001, 0.01, 0.1, 1]
        },
        {
            'vect__ngram_range' : [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            'clf__kernel': ['linear'],
            'clf__gamma': [1e-3, 1e-4, 1e-2, 0.1, 1],
            'clf__C': [0.001, 0.01, 0.1, 1]
        }
    ]
    gs_clfrSVM = Pipeline(
        [
            ('vect', CountVectorizer(decode_error = 'ignore')),
            ('tfidf', TfidfTransformer(use_idf = False)),
            ('clf', svm.SVC())
        ]
    )
    gs_clfsvm = GridSearchCV(gs_clfrSVM, tuned_parameters, cv = 10, n_jobs = -1)
    gs_clfsvm = gs_clfsvm.fit(train_data.data, train_data.target)
    predictedGSSVM = gs_clfsvm.predict(test_data.data)
    df['Labels_GSSVM'] = [train_data.target_names[i] for i in predictedGSSVM]
    print(df.head())
    df.to_csv('./assignment_2/predictions_GSSVM.csv', columns = ['Filename', 'Labels_GSSVM'], header = ['Filename', 'Label'], index = False)

cont = str(input('Optimize RFC model? [y/n]: '))

if cont.lower == 'y':
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {
        'clf__n_estimators': n_estimators,
        'clf__max_features': max_features,
        'clf__max_depth': max_depth,
        'clf__min_samples_split': min_samples_split,
        'clf__min_samples_leaf': min_samples_leaf,
        'clf__bootstrap': bootstrap,
        'vect__ngram_range' : [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
    }
    gs_clfrrfc = Pipeline(
        [
            ('vect', CountVectorizer(decode_error = 'ignore', strip_accents = 'unicode')),
            ('tfidf', TfidfTransformer(use_idf = False)),
            ('clf', RandomForestClassifier(verbose = True))
        ]
    )
    gs_clfrrfc = GridSearchCV(gs_clfrrfc, random_grid, cv = 10, n_jobs = -1)
    gs_clfrrfc = gs_clfrrfc.fit(train_data.data, train_data.target)
    predictedGSRFC = gs_clfrrfc.predict(test_data.data)
    df['Labels_GSRFC'] = [train_data.target_names[i] for i in predictedGSRFC]
    print(df.head())
    df.to_csv('./assignment_2/predictions_GSRFC.csv', columns = ['Filename', 'Labels_GSRFC'], header = ['Filename', 'Label'], index = False)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {
    'clf__n_estimators': n_estimators,
    'clf__max_features': max_features,
    'clf__max_depth': max_depth,
    'clf__min_samples_split': min_samples_split,
    'clf__min_samples_leaf': min_samples_leaf,
    'clf__bootstrap': bootstrap,
    'vect__ngram_range' : [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
}
gs_clfrrfc = Pipeline(
    [
        ('vect', CountVectorizer(decode_error = 'ignore', strip_accents = 'unicode')),
        ('tfidf', TfidfTransformer(use_idf = False)),
        ('clf', RandomForestClassifier(verbose = True))
    ]
)
gs_clfrrfc = GridSearchCV(gs_clfrrfc, random_grid, cv = 10, n_jobs = -1)
gs_clfrrfc = gs_clfrrfc.fit(train_data.data, train_data.target)
predictedGSRFC = gs_clfrrfc.predict(test_data.data)
df['Labels_GSRFC'] = [train_data.target_names[i] for i in predictedGSRFC]
print(df.head())
df.to_csv('./assignment_2/predictions_GSRFC.csv', columns = ['Filename', 'Labels_GSRFC'], header = ['Filename', 'Label'], index = False)