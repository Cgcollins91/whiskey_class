from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from collections import Counter
import spacy
from sklearn.model_selection import train_test_split
import tqdm
import re
import numpy as np
import matplotlib.pyplot as plt

def count(tokens):
    """
    Calculates some basic statistics about tokens in our corpus (i.e. corpus means collections text data)
    """
    # stores the count of each token

    word_counts = Counter()

    # stores the number of docs that each token appears in
    appears_in = Counter()
    total_docs = len(tokens)

    for token in tokens:
        # stores count of every appearance of a token
        word_counts.update(token)
        # use set() in order to not count duplicates, thereby count the num of docs that each token appears in
        appears_in.update(set(token))

    # build word count dataframe
    temp = zip(word_counts.keys(), word_counts.values())
    wc = pd.DataFrame(temp, columns=['word', 'count'])

    # rank the the word counts
    wc['rank'] = wc['count'].rank(method='first', ascending=False)
    total = wc['count'].sum()

    # calculate the percent total of each token
    wc['pct_total'] = wc['count'].apply(lambda token_count: token_count / total)

    # calculate the cumulative percent total of word counts
    wc = wc.sort_values(by='rank')
    wc['cul_pct_total'] = wc['pct_total'].cumsum()

    # create dataframe for document stats
    t2 = zip(appears_in.keys(), appears_in.values())
    ac = pd.DataFrame(t2, columns=['word', 'appears_in'])

    # merge word count stats with doc stats
    wc = ac.merge(wc, on='word')

    wc['appears_in_pct'] = wc['appears_in'].apply(lambda x: x / total_docs)

    return wc.sort_values(by='rank')


def load_data(split):
    target = 'ratingCategory'
    path = '/Users/chriscollins/Documents/lambda/unit_4/sprint_1_pycharm/module3/data/'
    train = pd.read_csv(path + 'train.csv')
    test = pd.read_csv(path + 'test.csv')

    train.head()
    train = train.set_index('id')
    X_test = test.set_index('id')
    if split == 1:
        train, val = train_test_split(train,
                                      test_size=0.2,
                                      stratify=train['ratingCategory'])
        y_train = train[target]
        y_val = val[target]
        X_val = val.drop(columns=[target])
        print(X_val.shape)
    else:
        X_val = []
        y_val = []

    X_train = train.drop(columns=[target])
    print(X_train.shape)
    print(X_train.head())
    return X_train, X_val, X_test, y_train, y_val


def my_tokenizer(text):
    clean_text = re.sub('[^a-zA-Z ]', '', text)
    tokens = clean_text.lower().split()
    return tokens


def get_word_vectors_avg(docs, nlp):
    out = []
    for doc in docs:
        vect = nlp(doc).vector
        out.append(sum(vect) / len(vect))
    return out


def get_adj(X_in, nlp):
    docs = X_in['description']
    out = []
    for doc in docs:
        adj = ""
        doc = nlp(doc)
        for token in doc:
            if token.pos_ == 'ADJ':
                adj += " " + token.text
        out.append(adj)
    return out


def get_entities(X, nlp):
    descriptions = X['description']
    entity_list = []

    for description in descriptions:
        doc = nlp(description)
        entitys = ""
        for entity in doc.ents:
            entitys += " " + entity.text
        #         for chunk in doc.noun_chunks:
        #             chunks += " " + chunk.text
        entity_list.append(entitys)
    #         entity_list.append(chunks)
    return entity_list


def grid_search(pipe, parameters, X, y):
    grid_search = GridSearchCV(pipe, parameters, cv=5, n_jobs=4, verbose=10)
    grid_search.fit(X, y)
    print(grid_search.best_score_)
    print(grid_search.best_params_)
    return grid_search.best_score_, grid_search.best_params_


def print_submission(X_test, pipe, subNumber):
    pred = pipe.predict(X_test)
    submission = pd.DataFrame({'id': X_test.index, 'ratingCategory': pred})
    submission['ratingCategory'] = submission['ratingCategory'].astype('int64')
    fname = f'/Users/chriscollins/Documents/lambda/unit_4/sprint_1_pycharm/module3/data/submission{subNumber}.csv'
    submission.to_csv(fname, index=False)
    print(submission.head())


def get_num_classes(labels):
    """Gets the total number of classes.

    # Arguments
        labels: list, label values.
            There should be at lease one sample for values in the
            range (0, num_classes -1)

    # Returns
        int, total number of classes.

    # Raises
        ValueError: if any label value in the range(0, num_classes - 1)
            is missing or if number of classes is <= 1.
    """
    num_classes = max(labels) + 1
    missing_classes = [i for i in range(num_classes) if i not in labels]
    if len(missing_classes):
        raise ValueError('Missing samples with label value(s) '
                         '{missing_classes}. Please make sure you have '
                         'at least one sample for every label value '
                         'in the range(0, {max_class})'.format(
            missing_classes=missing_classes,
            max_class=num_classes - 1))

    if num_classes <= 1:
        raise ValueError('Invalid number of labels: {num_classes}.'
                         'Please make sure there are at least two classes '
                         'of samples'.format(num_classes=num_classes))
    return num_classes


def get_num_words_per_sample(sample_texts):
    """Gets the median number of words per sample given corpus.

    # Arguments
        sample_texts: list, sample texts.

    # Returns
        int, median number of words per sample.
    """
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)


def plot_frequency_distribution_of_ngrams(sample_texts,
                                          ngram_range=(1, 2),
                                          num_ngrams=50,
                                          title='Frequency distribution of n-grams'):
    """Plots the frequency distribution of n-grams.

    # Arguments
        samples_texts: list, sample texts.
        ngram_range: tuple (min, mplt), The range of n-gram values to consider.
            Min and mplt are the lower and upper bound values for the range.
        num_ngrams: int, number of n-grams to plot.
            Top `num_ngrams` frequent n-grams will be plotted.
    """
    # Create args required for vectorizing.
    kwargs = {
        'ngram_range': ngram_range,
        'dtype': 'int32',
        'stop_words': 'english',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': 'word',  # Split text into word tokens.
    }
    vectorizer = CountVectorizer(**kwargs)

    # This creates a vocabulary (dict, where keys are n-grams and values are
    # idxices). This also converts every text to an array the length of
    # vocabulary, where every element idxicates the count of the n-gram
    # corresponding at that idxex in vocabulary.
    vectorized_texts = vectorizer.fit_transform(sample_texts)

    # This is the list of all n-grams in the index order from the vocabulary.
    all_ngrams = list(vectorizer.get_feature_names())
    num_ngrams = min(num_ngrams, len(all_ngrams))
    # ngrams = all_ngrams[:num_ngrams]

    # Add up the counts per n-gram ie. column-wise
    all_counts = vectorized_texts.sum(axis=0).tolist()[0]

    # Sort n-grams and counts by frequency and get top `num_ngrams` ngrams.
    all_counts, all_ngrams = zip(*[(c, n) for c, n in sorted(
        zip(all_counts, all_ngrams), reverse=True)])
    ngrams = list(all_ngrams)[:num_ngrams]
    counts = list(all_counts)[:num_ngrams]

    idx = np.arange(num_ngrams)
    plt.figure(figsize=(14, 6))
    plt.bar(idx, counts, width=0.8, color='b')
    plt.xlabel('N-grams')
    plt.ylabel('Frequencies')
    plt.title(title)
    plt.xticks(idx, ngrams, rotation=45)
    plt.show()
    return ngrams, counts


def plot_sample_length_distribution(sample_texts):
    """Plots the sample length distribution.

    # Arguments
        samples_texts: list, sample texts.
    """
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()


def plot_class_distribution(labels):
    """Plots the class distribution.

    # Arguments
        labels: list, label values.
            There should be at lease one sample for values in the
            range (0, num_classes -1)
    """
    num_classes = get_num_classes(labels)
    count_map = Counter(labels)
    counts = [count_map[i] for i in range(num_classes)]
    idx = np.arange(num_classes)
    plt.bar(idx, counts, width=0.8, color='b')
    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.title('Class distribution')
    plt.xticks(idx, idx)
    plt.show()
