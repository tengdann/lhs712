import os
import numpy as np
import pandas as pd
from sklearn import naive_bayes, svm
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}

# Create dataframe for model output
curdir = r'C:\Users\dteng\Desktop\lhs712\assignment_2\dataset\unlabeled-test-data\Gastroenterology'
files = list()
for file in os.listdir(curdir):
    files.append(file)

df = pd.DataFrame(data = files, columns = ['Filename'])
df['Labels_NB'], df['Labels_SVM'], df['Labels_Models'] = '', '', ''

# Preprocessing steps
train_data = load_files('assignment_2/dataset/train-data', load_content = True, shuffle = True, random_state = 42)
test_data = load_files('assignment_2/dataset/unlabeled-test-data', load_content = True, shuffle = False)
train_labels = train_data.target

count_vect = CountVectorizer(decode_error = 'ignore')
train_counts = count_vect.fit_transform(train_data.data)
tf_transformer = TfidfTransformer(use_idf = False)
train_tf = tf_transformer.fit_transform(train_counts)

test_counts = count_vect.transform(test_data.data)
tf_transformer = TfidfTransformer(use_idf = False)
test_tf = tf_transformer.fit_transform(test_counts)

# Naive Bayes
clfrNB = Pipeline(
    [
        ('vect', CountVectorizer(decode_error = 'ignore')),
        ('tfidf', TfidfTransformer(use_idf = False)),
        ('clf', naive_bayes.MultinomialNB())
    ]
)
clfrNB = clfrNB.fit(train_data.data, train_data.target)
predictedNB = clfrNB.predict(test_data.data)
df.Labels_NB = [train_data.target_names[i] for i in predictedNB]

# Linear SVM
clfrSVM = svm.SVC(kernel = 'linear', C = 0.1)
clfrSVM.fit(train_tf, train_labels)
predictedSVM = clfrSVM.predict(test_tf)
df.Labels_SVM = [train_data.target_names[i] for i in predictedSVM]
# print(df.head())

# Model selection?
gs_clf = GridSearchCV(clfrNB, parameters, n_jobs = -1)
gs_clf = gs_clf.fit(train_data.data, train_data.target)
print(gs_clf.best_score_)
# df.Labels_Models = [train_data.target_names[i] for i in predicted_labels]
# print(df.head())

df.to_csv('./assignment_2/predictions_NB.csv', columns = ['Filename', 'Labels_NB'], header = ['Filename', 'Label'],index = False)
df.to_csv('./assignment_2/predictions_SVM.csv', columns = ['Filename', 'Labels_SVM'], header = ['Filename', 'Label'],index = False)