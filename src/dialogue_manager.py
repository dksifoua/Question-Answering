from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

from src.utils import *

class SimpleDialogueManager(object):
    """
    This is the simplest dialogue manager to test the telegram bot.
    Your task is to create a more advanced one in dialogue_manager.py."
    """
    
    def generate_answer(self, question): 
        return "Hello, world!"
    

class DialogueManager(object):
    def __init__(self, resource_paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(resource_paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(resource_paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(resource_paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(resource_paths)

        self.create_chitchat_bot()

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        chatbot = ChatBot('Nannan')
        trainer = ChatterBotCorpusTrainer(chatbot)
        trainer.train('chatterbot.corpus.english')

        self.chitchat_bot = chatbot
       
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        prepared_question = text_prepare(question)
        features = self.tfidf_vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)[0]

        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            response = self.chitchat_bot.get_response(question)
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)[0]
            
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(prepared_question, tag)
           
            return self.ANSWER_TEMPLATE % (tag, thread_id)