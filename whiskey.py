# %% Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import spacy
import category_encoders as ce
from module3.helper import *
from sklearn.metrics import accuracy_score
from module3.explore_data import *
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# Load Spacy large
nlp = spacy.load("en_core_web_lg")


# %%

def get_unique(text_in, feature):
    whiskey_ratings = ['Excellent', 'Good', 'Poor']
    excellent_out, good_out, poor_out = [], [], []
    key1 = whiskey_ratings[0] + "_" + feature
    key2 = whiskey_ratings[1] + "_" + feature
    key3 = whiskey_ratings[2] + "_" + feature
    excellent = set(text_in[key1])
    good = set(text_in[key2])
    poor = set(text_in[key3])
    for item in excellent:
        if (item not in good) & (item not in poor):
            excellent_out.append(item)
    for item in good:
        if (item not in excellent) & (item not in poor):
            good_out.append(item)
    for item in poor:
        if (item not in excellent) & (item not in good):
            poor_out.append(item)
    return excellent_out, good_out, poor_out


def plot_and_add_hi_freq_feature(X_train_in, X_val_in, X_test_in, y_train, y_val, feature):
    whiskey_dict = {0: 'Excellent', 1: 'Good', 2: 'Poor'}
    raw_description = {}
    for i in range(0, 3):
        idx = y_train == i
        whiskey_rating = whiskey_dict[i]
        X_i = X_train_in.loc[idx]
        title = 'Frequency Distribution of ngrams for ' + whiskey_rating
        ngrams, counts = plot_frequency_distribution_of_ngrams(X_i[feature], title=title)
        key1 = whiskey_rating + "_ngrams"
        key2 = whiskey_rating + "_counts"
        raw_description[key1] = ngrams
        raw_description[key2] = counts

    # Get the unique ngrams with the highest frequency associated with each class and create numerical feature
    # with how often the highest frequency, unique ngram appeared in the description
    excel_uni_ngrams, good_uni_ngrams, poor_uni_ngrams = get_unique(raw_description, "ngrams")
    ngram_range = (1, 2)
    kwargs = {
        'ngram_range': ngram_range,
        'dtype': 'int32',
        'stop_words': 'english',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': 'word',  # Split text into word tokens.
    }

    vect = CountVectorizer(**kwargs)
    unique_ngrams = excel_uni_ngrams + poor_uni_ngrams + good_uni_ngrams

    n1, n2, n3 = len(X_train_in), len(X_test_in), len(X_val_in)
    X_combined = pd.concat([X_train_in, X_test_in, X_val_in])

    X_combined = make_feature(X_combined, vect, unique_ngrams, excel_uni_ngrams,
                              good_uni_ngrams, poor_uni_ngrams, feature)

    X_train = X_combined.iloc[0:n1]
    X_test = X_combined.iloc[n1:n1 + n2]
    X_val = X_combined.iloc[n1 + n2:n1 + n2 + n3]

    return X_train, X_val, X_test


def make_feature(X, vect, unique_ngrams, e, g, p, feature):
    vect.fit(X[feature])

    dtm = vect.transform(X[feature])
    df = pd.DataFrame(dtm.todense(), columns=vect.get_feature_names())

    df = df[df.columns.intersection(unique_ngrams)]

    e_list, g_list, p_list = [], [], []
    for i in range(0, len(df)):
        row = df.iloc[i]
        e_count, g_count, p_count = 0, 0, 0
        for ngram in unique_ngrams:
            count_at_ngram = row.loc[ngram]
            if count_at_ngram > 0:
                if ngram in e:
                    e_count += 1
                elif ngram in g:
                    g_count += 1
                elif ngram in p:
                    p_count += 1
        e_list.append(e_count)
        p_list.append(p_count)
        g_list.append(g_count)
        col1 = 'excellent' + "_" + feature
        col2 = 'good' + "_" + feature
        col3 = 'poor' + "_" + feature

    X[col1] = e_list
    X[col2] = g_list
    X[col3] = p_list
    return X


def get_pipe(pipe_type, tfidf_def=None, SVD_def=None):
    """
    Set-Up Pipe
    :param pipe_type: lsa, svd, combined
    :param tfidf_def: **kwargs to initialize TfidfVectorizer
    :param SVD_def: ** kwargs to initialize TruncatedSVD
    :return:
    """
    pipe_out = []
    if tfidf_def is None:
        tfidf_def = {'stop_words': 'english',
                     'ngram_range': (1, 2),
                     'min_df': 2,
                     'max_df': .5
                     }
    if SVD_def is None:
        SVD_def = {'n_components': 20,
                   'algorithm': 'randomized',
                   'n_iter': 100
                   }

    vect = TfidfVectorizer(**tfidf_def)
    svm = LinearSVC()
    svd = TruncatedSVD(**SVD_def)
    lr = LogisticRegression(solver='lbfgs')
    lsa = Pipeline([('vect', vect), ('svd', svd)])

    if pipe_type == 'lsa':
        pipe_out = Pipeline([
            ('lsa', lsa),
            ('clf', lr)
        ])

    elif pipe_type == 'svd':
        pipe_out = Pipeline([
            ('vect', vect),
            ('clf', svm)]
        )

    elif pipe_type == 'combined':
        pipe_out = lsa

    return pipe_out


def ohe_cols(X_in):
    X_out = X_in.copy()
    columns = X_in.columns
    ohe_cols = []
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    # define a mapping of chars to integers
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    for col in columns:
        if ('good' in col) or ('excellent' in col) or ('poor' in col):
            ohe_cols.append(col)

    ohe_df = X_out[ohe_cols].copy()
    ohe_df_out = pd.DataFrame(columns=ohe_df.columns)
    for col, data in ohe_df.items():
        for index, value in data.items():
            ohe_df_out.at[index, col] = int_to_char[value]

    enc = ce.OneHotEncoder(cols=ohe_cols, use_cat_names=False)
    ohe = enc.fit_transform(ohe_df_out)

    return ohe


def wrangle(X_tr, X_v, X_te, features, tfidf_def=None, SVD_def=None, include_text=True, one_hot=True):
    lsa = get_pipe('combined', tfidf_def, SVD_def)
    n1, n2, n3 = len(X_tr), len(X_v), len(X_te)
    X = pd.concat([X_tr, X_v, X_te])
    if include_text:
        dtm = lsa.fit_transform(X['description'])
        dtm = pd.DataFrame(data=dtm, index=X.index)
    else:
        dtm = pd.DataFrame()
    ratings = ['excellent', 'good', 'poor']
    if not one_hot:
        for rating in ratings:
            for feature in features:
                col_name = rating + "_" + feature
                dtm[col_name] = X[col_name]
                dtm[col_name] = X[col_name]
                dtm[col_name] = X[col_name]
    else:
        ohe = ohe_cols(X)
        dtm = pd.concat([dtm, ohe], axis=1)
    dtm_training = dtm.iloc[0:n1]
    dtm_validation = dtm.iloc[n1:n1 + n2]
    dtm_test = dtm.iloc[n1+n2:n1+n2+n3]

    return dtm_training, dtm_validation, dtm_test


# %% Load Data, create feature with just adjectives, use spaCy to get word vectors in raw descriptions

load = False
if load:
    # Load Data
    X_train_in, X_val_in, X_test_in, y_train, y_val = load_data(split=1)

    # Pull out adjectives from description:
    X_train_in['adj'] = get_adj(X_train_in, nlp)
    X_val_in['adj'] = get_adj(X_val_in, nlp)
    X_test_in['adj'] = get_adj(X_test_in, nlp)

    # pull out entities from description:
    X_train_in['entities'] = get_entities(X_train_in, nlp)
    X_val_in['entities'] = get_entities(X_val_in, nlp)
    X_test_in['entities'] = get_entities(X_test_in, nlp)

    # Save data to pkl
    X_train_in.to_pickle('X_train.pkl')
    X_val_in.to_pickle('X_val.pkl')
    X_test_in.to_pickle('X_test.pkl')
    y_train.to_pickle('y_train.pkl')
    y_val.to_pickle('y_val.pkl')
else:
    X_train_in = pd.read_pickle('X_train.pkl')
    X_val_in = pd.read_pickle('X_val.pkl')
    X_test_in = pd.read_pickle('X_test.pkl')
    y_train = pd.read_pickle('y_train.pkl')
    y_val = pd.read_pickle('y_val.pkl')

# %%
# Plot n_gram frequency distribution for each rating of whiskey and pull out most common words for each
# rating description and most common adjectives
features = ['description', 'adj']

# Check rows in before / after are equal and columns frow
print(X_train_in.shape)
print(X_test_in.shape)
print(X_val_in.shape)

X_train_f, X_val_f, X_test_f = plot_and_add_hi_freq_feature(
    X_train_in, X_val_in, X_test_in, y_train, y_val, features[0])

X_train, X_val, X_test = plot_and_add_hi_freq_feature(
    X_train_f, X_val_f, X_test_f, y_train, y_val, features[1])

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

# %%

# %%


# %%
# tfidf vectorizer param search
max_df_in = [.3]  # np.linspace(.3, .95, 10)
min_df_in = [.1]  # np.linspace(.1, .5, 10)
max_features_in = [100, 300]  # [100, 300, 500]
n1, n2, n3 = len(max_df_in), len(min_df_in), len(max_features_in)

n_components_in = [30]  # [10, 30, 50]
n4 = len(n_components_in)

# clf = RandomForestClassifier()
# min_samples_leaf_in = [0.0024000000000000002] #np.linspace(.0002, .02, num=10)
# min_samples_split_in = [0.0002777777777777778] #np.linspace(.0001, .0005, num=10)
# n5, n6 = len(min_samples_leaf_in), len(min_samples_split_in)

init = RandomForestClassifier(min_samples_leaf=0.0024000000000000002,
                              min_samples_split=0.0002777777777777778)
clf = GradientBoostingClassifier(init=init)
learn_in = [.1, .5]  # np.linspace(.01, .8, 30)
n_est_in = [100, 300, 500]  # np.linspace(100, 1000, 30)
warm_start_in = [True, False]
loss_in = ['deviance']
min_samples_leaf_in = [1]  # np.linspace(.0002, .02, num=10) #.0002
criterion_in = ['friedman_mse']
n5, n6, n7, n8, n9, n10 = len(learn_in), len(n_est_in), len(warm_start_in), len(loss_in), len(min_samples_leaf_in), \
                          len(criterion_in)

score_out = []
total_iter = n1 * n2 * n3 * n4 * n5 * n6 * n7 * n8 * n9 * n10
idx_out = []

for i in range(0, n1):
    for j in range(0, n2):
        for k in range(0, n3):
            tfidf_params = {
                'max_df': max_df_in[i],
                'min_df': min_df_in[j],
                'max_features': max_features_in[k],
            }
            for n in range(0, n4):
                SVD_params = {'n_components': n_components_in[n],
                              'algorithm': 'randomized',
                              'n_iter': 100
                              }
                dtm_train, dtm_val, dtm_test = wrangle(X_train, X_val, X_test, features, tfidf_params, SVD_params, True, True)
                for o in range(0, n5):
                    for p in range(0, n6):
                        for q in range(0, n7):
                            for r in range(0, n8):
                                for s in range(0, n9):
                                    for t in range(0, n10):
                                        # clf = RandomForestClassifier(min_samples_leaf=min_samples_leaf_in[o],
                                        #                              min_samples_split=min_samples_split_in[p])
                                        xgb_params = {'learning_rate': learn_in[o],
                                                      'n_estimators': n_est_in[p],
                                                      'warm_start': warm_start_in[q],
                                                      'loss': loss_in[r],
                                                      'min_samples_leaf': min_samples_leaf_in[s],
                                                      'criterion': criterion_in[t],
                                                      'init': init,
                                                      }
                                        clf = GradientBoostingClassifier(**xgb_params)
                                        clf = clf.fit(dtm_train, y_train)
                                        y_pred = clf.predict(dtm_val)
                                        score = accuracy_score(y_pred, y_val)
                                        print(f'Run Accuracy Score: {score}')
                                        pct_complete = (i + j + k + n + o + p + q + r + s + t) / total_iter
                                        print(f'% Complete: {pct_complete}')
                                        score_out.append(score)
                                        idx_out.append([i, j, k, n, o, p, r, s, t])

index_max = max(range(len(score_out)), key=score_out.__getitem__)
idx = idx_out[index_max]
print(f'Max Score: {round(score_out[index_max] * 100, 2)} %')
print(f'max_df: {max_df_in[idx[0]]}')
print(f'min_df: {min_df_in[idx[1]]}')
print(f'max_features : {max_features_in[idx[2]]}')
print(f'n_components : {n_components_in[idx[3]]}')
print(f'learning_rate : {n_est_in[idx[4]]}')
print(f'warm_start: {warm_start_in[idx[5]]}')
print(f'loss : {loss_in[idx[6]]}')
print(f'min_samples_leaf_in: {min_samples_leaf_in[idx[7]]}')
print(f'criterion : {criterion_in[idx[8]]}')


# %%
tfidf_params = {
    'max_df': max_df_in[idx[0]],
    'min_df': min_df_in[idx[1]],
    'max_features': max_features_in[idx[2]],
}
SVD_params = {'n_components': n_components_in[idx[3]],
              'algorithm': 'randomized',
              'n_iter': 100
              }

dtm_train, dtm_val, dtm_test = wrangle(X_train, X_val, X_test, features, tfidf_params, SVD_params, True, True)

dtm_train_val = pd.concat([dtm_train, dtm_val])
y_train_val = pd.concat([y_train, y_val])

xgb_params = {'learning_rate': n_est_in[idx[4]],
              'n_estimators': n_est_in[p],
              'warm_start': warm_start_in[idx[5]],
              'loss': loss_in[idx[6]],
              'min_samples_leaf': min_samples_leaf_in[idx[7]],
              'criterion': criterion_in[idx[8]],
              'init': init,
              }
clf = GradientBoostingClassifier(**xgb_params)
clf = clf.fit(dtm_train_val, y_train_val)
print_submission(dtm_test, clf, 10)
# print_submission(dtm_test, clf, 8)

# %%

parameters = {
    'vect__max_features': (10000, 20000),
    'vect__ngram_range': [(1, 2), (1, 3)],
    'clf__penalty': ('l1', 'l2'),
    'clf__C': (0.1, 0.5, 1., 2.),
    'svd__n_components': (100, 500),
}

# %%
# Get word vectors
# X_train_vectors = get_word_vectors(X_train['description'], nlp)
# X_val_vectors = get_word_vectors(X_val['description'], nlp)
# X_train = get_word_vectors(X_train['adj'])
# X_val = get_word_vectors(X_val['adj'])

# %%
# %%
# tfidf vectorizer param search
max_df_in = [.3]  # np.linspace(.3, .95, 10)
min_df_in = [.1]  # np.linspace(.1, .5, 10)
max_features_in = [100, 300]  # [100, 300, 500]
n1, n2, n3 = len(max_df_in), len(min_df_in), len(max_features_in)

n_components_in = [30]  # [10, 30, 50]
n4 = len(n_components_in)

# clf = RandomForestClassifier()
# min_samples_leaf_in = [0.0024000000000000002] #np.linspace(.0002, .02, num=10)
# min_samples_split_in = [0.0002777777777777778] #np.linspace(.0001, .0005, num=10)
# n5, n6 = len(min_samples_leaf_in), len(min_samples_split_in)

clfwhisk = RandomForestClassifier()
learn_in = [.1, .5]  # np.linspace(.01, .8, 30)
n_est_in = [100, 300, 500]  # np.linspace(100, 1000, 30)
warm_start_in = [True, False]
loss_in = ['deviance']
min_samples_leaf_in = [1]  # np.linspace(.0002, .02, num=10) #.0002
criterion_in = ['friedman_mse']
n5, n6, n7, n8, n9, n10 = len(learn_in), len(n_est_in), len(warm_start_in), len(loss_in), len(min_samples_leaf_in), \
                          len(criterion_in)

score_out = []
total_iter = n1 * n2 * n3 * n4 * n5 * n6 * n7 * n8 * n9 * n10
idx_out = []

for i in range(0, n1):
    for j in range(0, n2):
        for k in range(0, n3):
            tfidf_params = {
                'max_df': max_df_in[i],
                'min_df': min_df_in[j],
                'max_features': max_features_in[k],
            }
            for n in range(0, n4):
                SVD_params = {'n_components': n_components_in[n],
                              'algorithm': 'randomized',
                              'n_iter': 100
                              }
                dtm_train, dtm_val, dtm_test = wrangle(X_train, X_val, X_test, features, tfidf_params, SVD_params, True, True)
                for o in range(0, n5):
                    for p in range(0, n6):
                        for q in range(0, n7):
                            for r in range(0, n8):
                                for s in range(0, n9):
                                    for t in range(0, n10):
                                        # clf = RandomForestClassifier(min_samples_leaf=min_samples_leaf_in[o],
                                        #                              min_samples_split=min_samples_split_in[p])
                                        xgb_params = {'learning_rate': learn_in[o],
                                                      'n_estimators': n_est_in[p],
                                                      'warm_start': warm_start_in[q],
                                                      'loss': loss_in[r],
                                                      'min_samples_leaf': min_samples_leaf_in[s],
                                                      'criterion': criterion_in[t],
                                                      'init': init,
                                                      }
                                        clf = GradientBoostingClassifier(**xgb_params)
                                        clf = clf.fit(dtm_train, y_train)
                                        y_pred = clf.predict(dtm_val)
                                        score = accuracy_score(y_pred, y_val)
                                        print(f'Run Accuracy Score: {score}')
                                        pct_complete = (i + j + k + n + o + p + q + r + s + t) / total_iter
                                        print(f'% Complete: {pct_complete}')
                                        score_out.append(score)
                                        idx_out.append([i, j, k, n, o, p, r, s, t])

index_max = max(range(len(score_out)), key=score_out.__getitem__)
idx = idx_out[index_max]
print(f'Max Score: {round(score_out[index_max] * 100, 2)} %')
print(f'max_df: {max_df_in[idx[0]]}')
print(f'min_df: {min_df_in[idx[1]]}')
print(f'max_features : {max_features_in[idx[2]]}')
print(f'n_components : {n_components_in[idx[3]]}')
print(f'learning_rate : {n_est_in[idx[4]]}')
print(f'warm_start: {warm_start_in[idx[5]]}')
print(f'loss : {loss_in[idx[6]]}')
print(f'min_samples_leaf_in: {min_samples_leaf_in[idx[7]]}')
print(f'criterion : {criterion_in[idx[8]]}')


# %%
tfidf_params = {
    'max_df': max_df_in[idx[0]],
    'min_df': min_df_in[idx[1]],
    'max_features': max_features_in[idx[2]],
}
SVD_params = {'n_components': n_components_in[idx[3]],
              'algorithm': 'randomized',
              'n_iter': 100
              }

dtm_train, dtm_val, dtm_test = wrangle(X_train, X_val, X_test, features, tfidf_params, SVD_params, True, True)

dtm_train_val = pd.concat([dtm_train, dtm_val])
y_train_val = pd.concat([y_train, y_val])

xgb_params = {'learning_rate': n_est_in[idx[4]],
              'n_estimators': n_est_in[p],
              'warm_start': warm_start_in[idx[5]],
              'loss': loss_in[idx[6]],
              'min_samples_leaf': min_samples_leaf_in[idx[7]],
              'criterion': criterion_in[idx[8]],
              'init': init,
              }
clf = GradientBoostingClassifier(**xgb_params)
clf = clf.fit(dtm_train_val, y_train_val)
print_submission(dtm_test, clf, 10)
# print_submission(dtm_test, clf, 8)
