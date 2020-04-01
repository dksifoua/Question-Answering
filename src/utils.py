import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

from gensim.models import KeyedVectors

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': './src/notebooks/out/intent_recognizer.pkl',
    'TAG_CLASSIFIER': './src/notebooks/out/tag_classifier.pkl',
    'TFIDF_VECTORIZER': './src/notebooks/out/tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': './src/notebooks/out/thread_embeddings_by_tags',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""

    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()

def load_embeddings():
    """Loads pre-trained word embeddings from tsv file.
    Args:
      embeddings_path - path to the embeddings file.
    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    
    w2v_embeddings = KeyedVectors.load_word2vec_format('./src/notebooks/data/GoogleNews-vectors-negative300.bin',
                                                       binary=True)
    embeddings_dim = w2v_embeddings['word'].shape[0]
    return w2v_embeddings, embeddings_dim

def question_to_vec(question, embeddings, dim=300):
    """
        question: a string
        embeddings: dict where the key is a word and a value is its' embedding
        dim: size of the representation

        result: vector representation for the question
    """
    vec = []
    for token in question.split():
        if token in embeddings:
            vec.append(embeddings[token])
    if len(vec) == 0:
        return np.zeros((dim,))
    return np.stack(vec).mean(axis=0)


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)