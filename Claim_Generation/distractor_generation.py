from sense2vec import Sense2Vec
import string
from collections import OrderedDict
import random

class Distractor_Generation():
    def __init__(self, sense2vec_path, T):
        self.sense2vec = Sense2Vec().from_disk(sense2vec_path)
        self.T = T

    def get_options(self, answer):
        distractors =[]
        try:
            distractors = self.sense2vec_get_words(answer)
            if len(distractors) > 0:
                return distractors
            else:
                return None
        except:
            return None

    # word: "Donald Trump"
    # sense: "Donald_Trump|PERSON"
    def sense2vec_get_words(self, word):
        candidates = []

        word = word.replace(" ", "_")

        sense = self.sense2vec.get_best_sense(word)
        sense_type = sense.split("|")[1].strip()

        most_similar = self.sense2vec.most_similar(sense, n=15)
        # print(most_similar)

        if len(most_similar) == 0:
            return None
        
        # sample from all phrases that similarity > T
        # the entity type should be the same
        for each_word in most_similar:
            append_word = each_word[0].split("|")[0].replace("_", " ").strip()
            word_type = each_word[0].split("|")[1].strip()
            sim_score = each_word[1]
            if sim_score >= self.T and sense_type == word_type:
                candidates.append((append_word, word_type))
        
        # if len(candidates) == 0:
        #     append_word = most_similar[0][0].split("|")[0].replace("_", " ").strip()
        #     word_type = most_similar[0][0].split("|")[1].strip()
        #     candidates.append((append_word, word_type))

        self.candidates = candidates
        output = random.sample(candidates, 1)[0]
        return output

if __name__ == "__main__":
    answer = 'one'
    # answer = 'Donald Trump'
    # answer = 'John Amsterdam'
    sense2vec_path = '../dependencies/s2v_old'

    replacement_generator = Distractor_Generation(sense2vec_path = sense2vec_path, T = 0.7)
    replacement = replacement_generator.get_options(answer)
    print(replacement)
