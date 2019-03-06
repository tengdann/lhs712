from sklearn import naive_bayes, svm, model_selection
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Preprocessing steps
train_data = load_files('assignment_2/dataset/train-data', load_content = True, shuffle = True, random_state = 42)
count_vect = CountVectorizer(decode_error = 'ignore')
train_counts = count_vect.fit_transform(train_data.data)
tf_transformer = TfidfTransformer(use_idf = False).fit(train_counts)
train_tf = tf_transformer.transform(train_counts)

# Naive Bayes
clfrNB = naive_bayes.MultinomialNB()
