from collections import defaultdict
import os

import pandas as pd
import spacy
from spacy.tokens import Token, Doc, Span
# # https://pypi.org/project/spacy-arguing-lexicon/
from spacy_arguing_lexicon import ArguingLexiconParser
from textblob import TextBlob

Span.set_extension('argument_type', default=None, force=True)
Doc.set_extension('total_argument_types', default=None, force=True)
Doc.set_extension('metadata', default=None, force=True)
Doc.set_extension('subjectivity', default=None, force=True)


class Emotions:
    name = 'emotions'
    def __init__(self):

        # Download from https://github.com/sebastianruder/emotion_proposition_store/tree/master/NRC-Emotion-Lexicon-v0.92
        # https://github.com/JULIELab/XANEW
        # https://github.com/ArtsEngine/concreteness
        # missing values for the later two are normalized to be the "average" value
        # ^ combined into one table
        mypath = os.path.dirname(__file__)
        filepath = os.path.join(mypath, 'lexicons/combined_emotions.csv')
        self.emolex_df = pd.read_csv(filepath, index_col=0)
        
    def __call__(self, doc):
        
        total_emotions = defaultdict(int)
        
        Token.set_extension("emotions", default=None, force=True)
        Doc.set_extension("emotions", default=None, force=True)
        
        for word in doc:
            
            # Keep track to avoid double counting word vs lemma
            seen_emotions = set()

            text = word.text.lower()
            
            # Check both word and its lemma
            if text in self.emolex_df.index:
                cur_word = text
            elif word.lemma_.lower() in self.emolex_df.index:
                cur_word = word.lemma_.lower()
            else:
                continue
            cur_emotions = self.emolex_df.loc[cur_word].to_dict()
            word._.emotions = cur_emotions

            for key, val in cur_emotions.items():
                total_emotions[key] += val
                
        doc._.emotions = dict(total_emotions)
        return doc


def unroll_arguments(doc):
    Span.set_extension('argument_type', default=None, force=True)
    
    all_args = defaultdict(int)
    
    for argument_span in doc._.arguments.get_argument_spans():
        argument_span._.argument_type = argument_span.label_
        all_args[argument_span.label_] += 1
        
    doc._.total_argument_types = dict(all_args)
    return doc

def do_text_blob_sentiment(doc):
    blob = TextBlob(doc.text)

    doc.sentiment = blob.sentiment.polarity
    doc._.subjectivity = blob.sentiment.subjectivity
    return doc

def load_custom_spacy():
    nlp = spacy.load('en_core_web_md')

    nlp.add_pipe(unroll_arguments)   
    nlp.add_pipe(Emotions())
    nlp.add_pipe(ArguingLexiconParser(lang=nlp.lang))
    nlp.add_pipe(do_text_blob_sentiment)

    return nlp

def get_style_features(text, nlp):
    """ Helper to extract useful analysis """
    doc = nlp(text)
    
    final_data = {f'mpqa_{k}': v for k, v in doc._.total_argument_types.items()}
    final_data['tb_sentiment'] = doc.sentiment
    final_data['tb_subjectivity'] =  doc._.subjectivity
    
    # Return avg for emotions
    emotion_data = doc._.emotions
    emotion_data = {k: v / len(doc) for k, v in emotion_data.items()}
    
    final_data.update(emotion_data)
    
    cur_lemmas = list(set(w.lemma_ for w in doc))
    final_data['lemmas'] = cur_lemmas
    
    return final_data


