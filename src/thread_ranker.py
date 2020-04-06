import os
from utils import *

class ThreadRanker:
    
    def __init__(self, resource_paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings()
        self.thread_embeddings_folder = resource_paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ 
        Returns id of the most similar thread for the question.
        The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)
        
        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim)[np.newaxis, :]
        best_thread = pairwise_distances_argmin(question_vec, thread_embeddings, metric='cosine')[0]
        
        return thread_ids[best_thread]
