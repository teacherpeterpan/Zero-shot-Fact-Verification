import json
import argparse
import itertools
import stanza
import uuid
import random
import os

from dataset_loader import FEVER_Dataset
from distractor_generation import Distractor_Generation
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from nltk.tokenize import word_tokenize
from T5_QG import pipeline

from tqdm import tqdm

class ClaimGenerator():
    def __init__(self, config):
        # QA2D model object
        print('Loading QA2D module >>>>>>>>')
        model_args = Seq2SeqArgs()
        model_args.max_length = 64

        self.QA2D_model = Seq2SeqModel(
            encoder_decoder_type="bart", 
            encoder_decoder_name=config.QA2D_model_path,
            cuda_device=config.gpu_index,
            args=model_args
        )

        # Replacement Generator object
        print('Loading Replacement Generator module >>>>>>>>')
        self.replacement_generator = Distractor_Generation(sense2vec_path = config.sense_to_vec_path, T = 0.7)

        self.claim_type = config.claim_type
        self.save_path = config.save_path

        # Load fever data
        print('Loading FEVER dataset >>>>>>>>')
        FEVER = FEVER_Dataset(config)
        self.fever_dataset = FEVER.FEVER_train if config.split == 'train' else FEVER.FEVER_dev

        # load entity dict
        print('Loading entity dict >>>>>>>>')
        with open(config.entity_dict, 'r') as f:
            self.entity_dict = json.load(f)

        # load precomputed QA
        print('Loading precomputed QAs >>>>>>>>')
        with open(config.QA_path, 'r') as f:
            self.QA_dict = json.load(f)

        if self.claim_type == 'NEI':
            print('Loading wikipedia database >>>>>>>>')
            self.wiki_path = config.wiki_path
            self.wiki_dict = self.load_wikipedia()
            print('Loading QG module >>>>>>>>')
            self.qg_nlp = pipeline("question-generation", model='valhalla/t5-base-qg-hl', 
                                    qg_format="highlight", gpu_index = config.gpu_index)
            self.stanza_nlp = stanza.Pipeline('en', use_gpu = True)

    '''
    Load Wikipedia to get extra contexts
    '''
    def load_wikipedia(self):
        print('loading wikipedia articles...')
        g = os.walk(self.wiki_path)
        wiki_dict = {}
        for path,dir_list,file_list in g:  
            for file_name in file_list:  
                with open(os.path.join(path, file_name), 'r') as f:
                    for line in f:
                        record = json.loads(line)
                        if not record['id'].strip() == "" and not record['lines'].strip() == "":
                            index_to_sent = {}
                            for sent in record['lines'].split('\n'):
                                tmp = sent.split('\t')
                                if len(tmp) >= 2 and not tmp[1] == "": 
                                    index_to_sent[tmp[0]] = tmp[1]
                            wiki_dict[record['id']] = index_to_sent
        return wiki_dict

    '''
    Generate support claims
    '''
    def generate_support_claims(self, sample):
        texts, ID = sample['context'], str(sample['id'])

        # Step 1: load entities in text
        passage_entities = set()
        for ent_text, ent_type in self.entity_dict[ID]:
            if ent_text not in ['-LRB-', '-RRB-', 'LRB', 'RRB']:
                passage_entities.add(f'{ent_text}:::{ent_type}')
        if len(passage_entities) == 0:
            return None

        # Step 2: load precomputed QAs for entities
        if ID not in self.QA_dict:
            return None
        QA_for_sample = self.QA_dict[ID]
        QA_pairs = []
        for entity in passage_entities:
            if entity in QA_for_sample:
                question, answer = QA_for_sample[entity]
                QA_pairs.append({'question': question, 'answer': answer})
        if len(QA_pairs) == 0:
            return None
    
        # Step 3: QA2D
        to_predict = [qa['question'] + ' [SEP] ' + qa['answer'] for qa in QA_pairs]
        results = []
        try:
            results = self.QA2D_model.predict(to_predict)
        except:
            return None
        if len(results) == 0:
            return None

        generated_samples = []
        claim_set = set()
        for ind in range(len(results)):
            claim = results[ind]
            evidence = texts
            ID = str(uuid.uuid4())
            # don't want repeated claims
            if claim not in claim_set:
                generated_samples.append({'id': ID, 'context': evidence, 'claim': claim, 'label': 'SUPPORTED'})
                claim_set.add(claim)
        
        return generated_samples

    '''
    Generate REFUTE claim with global replacement
    '''
    def generate_refute_global_claims(self, sample):
        texts, ID = sample['context'], str(sample['id'])

        # Step 1: load entities in text
        passage_entities = set()
        for ent_text, ent_type in self.entity_dict[ID]:
            if ent_text not in ['-LRB-', '-RRB-', 'LRB', 'RRB']:
                passage_entities.add(f'{ent_text}:::{ent_type}')
        if len(passage_entities) == 0:
            return None
        
        # Step 2: get entity replacement
        entity_replacement_dict = {} # get replacement beforehand to save time

        valid_entities = set()
        for ent in passage_entities:
            ent_text, _ = ent.split(':::')
            replacement = self.replacement_generator.get_options(ent_text)
            if replacement is not None:
                entity_replacement_dict[ent_text] = replacement
                valid_entities.add(ent)

        # Step 3: load precomputed QAs for entities
        if ID not in self.QA_dict:
            return None
        QA_for_sample = self.QA_dict[ID]
        QA_pairs = []
        for entity in valid_entities:
            if entity in QA_for_sample:
                ent_text, ent_type = entity.split(':::')
                question, answer = QA_for_sample[entity]
                QA_pairs.append({'question': question, 'answer': answer, 'answer_type': ent_type})
        if len(QA_pairs) == 0:
            return None

        # Step 4: Answer Replacement
        to_predict = []
        replace_type = []
        for qa in QA_pairs:
            ans_ent_text = qa['answer']
            ans_ent_type = qa['answer_type']
            if ans_ent_text == "" or ans_ent_type == "":
                continue
            replacement = entity_replacement_dict[ans_ent_text]
            if not replacement is None:
                to_predict.append(qa['question'] + ' [SEP] ' + replacement[0])
                replace_type.append(ans_ent_type)

        # Step 5: QA2D
        if len(to_predict) == 0:
            return None
        results = []
        try:
            results = self.QA2D_model.predict(to_predict)
        except:
            return None
        if len(results) == 0:
            return None

        generated_samples = []
        claim_set = set()
        for ind in range(len(results)):
            claim = results[ind]
            evidence = texts
            ID = str(uuid.uuid4())
            # avoid repeative sample
            if claim not in claim_set:
                generated_samples.append({'id': ID, 'context': evidence, 
                                        'claim': claim, 'label': 'REFUTED', 'replace_type': replace_type[ind]})
                claim_set.add(claim)

        return generated_samples

    '''
    Generate REFUTE claim with local replacement
    '''
    def generate_refute_local_claims(self, sample):
        texts, ID = sample['context'], str(sample['id'])
        
        # Step 1: load entities in text
        passage_entities = {}
        for ent_text, ent_type in self.entity_dict[ID]:
            if ent_text not in ['-LRB-', '-RRB-', 'LRB', 'RRB']:
                if ent_type not in passage_entities:
                    passage_entities[ent_type] = set()
                passage_entities[ent_type].add(ent_text)
        if len(passage_entities) == 0:
            return None

        # Step 2: QG for entities that satisfy the condition
        valid_entities = set()
        valid_entities_with_types = []
        for ent_type, ents in passage_entities.items():
            if len(ents) > 1:
                for ent in ents:
                    if not ent in valid_entities:
                        valid_entities.add(ent)
                        valid_entities_with_types.append([ent, ent_type])

        if ID not in self.QA_dict:
            return None
        QA_for_sample = self.QA_dict[ID]
        QA_pairs = []
        for ent_text, ent_type in valid_entities_with_types:
            entity = f'{ent_text}:::{ent_type}'
            if entity in QA_for_sample:
                question, answer = QA_for_sample[entity]
                QA_pairs.append({'question': question, 'answer': answer, 'ans_type': ent_type})
        if len(QA_pairs) == 0:
            return None
    
        # build-in function
        def phrase_match(long_str, short_str):
            long_str = ' ' + long_str + ' '
            short_str = ' ' + short_str + ' '
            return long_str.find(short_str)

        # Step 3: Answer Replacement
        to_predict = []
        replace_type = []
        for qa in QA_pairs:
            ans_ent_text = qa['answer']
            ans_ent_type = qa['ans_type']
            if ans_ent_text == "" or ans_ent_type == "":
                continue

            if ans_ent_type in passage_entities:
                candidate_list = list(passage_entities[ans_ent_type] - set([ans_ent_text]))
                if len(candidate_list) > 0:
                    replacement = random.sample(candidate_list, 1)[0]
                    # Rule 1: the replacement should not have phrase overlap with the ans_ent_text
                    # for example: October 24 -> October 24, 1983
                    phrase_overlap = -1
                    if len(replacement) > len(ans_ent_text):
                        phrase_overlap = phrase_match(replacement, ans_ent_text)
                    else:
                        phrase_overlap = phrase_match(ans_ent_text, replacement)
                    
                    # Rule 2: the question should not already contain the replacement
                    # for example: Who created Google? replacement = Google
                    contain_check = phrase_match(qa['question'], replacement)

                    if phrase_overlap == -1 and contain_check == -1:
                        to_predict.append(qa['question'] + ' [SEP] ' + replacement)
                        replace_type.append(ans_ent_type)

        # Step 4: QA2D
        if len(to_predict) == 0:
            return None
        results = []
        try:
            results = self.QA2D_model.predict(to_predict)
        except:
            return None
        if len(results) == 0:
            return None

        generated_samples = []
        claim_set = set()
        for ind in range(len(results)):
            claim = results[ind]
            evidence = texts
            ID = str(uuid.uuid4())
            # avoid repeative sample
            if claim not in claim_set:
                generated_samples.append({'id': ID, 'context': evidence, 
                                        'claim': claim, 'label': 'REFUTED', 'replace_type': replace_type[ind]})
                claim_set.add(claim)

        return generated_samples


    '''
    Generate NEI claim with extra contexts
    '''
    def generate_NEI_claims(self, sample):
        texts, ID = sample['context'], str(sample['id'])

        # Step 1: load entities in text
        passage_entities = set()
        for ent_text, ent_type in self.entity_dict[ID]:
            if ent_text not in ['-LRB-', '-RRB-', 'LRB', 'RRB']:
                passage_entities.add(ent_text)
        if len(passage_entities) == 0:
            return None

        # We consider the first entity of the evidence as the topic entity
        # the generated claim should contain the topic entity
        # first_entity = None
        # for ent in passage_entities:
        #     if texts.find(ent) == 0:
        #         first_entity = ent
        #         break
        
        # if first_entity is None:
        #     return None

        # Step 2: create extra context
        candidate_sentences = {}
        for eve in sample['ori_evidence']:
            wiki_id, sent_ind = eve[0], eve[1]
            if wiki_id not in candidate_sentences:
                candidate_sentences[wiki_id] = set()
            candidate_sentences[wiki_id].add(sent_ind)

        # extra context = sentence 0 + sentence 1 + sentence 2
        # then we exclude the evidence sentence
        # if still extra context left, then generate Qs
        # else: just skip
        QG_input_contexts = []
        QG_input_answers = []

        context_scope = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # context_scope = [0, 1, 2]

        for wiki_id in candidate_sentences:
            if wiki_id not in self.wiki_dict:
                continue
            eve_sents = candidate_sentences[wiki_id]
            valid_sents = set(context_scope) - eve_sents
            # get extra contexts
            context_sents = ["", "", "", "", "", "", "", "", "", ""]
            # context_sents = ["", "", ""]
            if len(valid_sents) > 0:
                for sent_id in context_scope:
                    if str(sent_id) in self.wiki_dict[wiki_id]:
                        context_sents[sent_id] = self.wiki_dict[wiki_id][str(sent_id)]

            extra_contexts = ' '.join(context_sents)
            if extra_contexts.strip() == "":
                continue

            # analyze entities in text
            doc = self.stanza_nlp(extra_contexts)
            extra_entities = set([ent.text for ent in doc.ents])
            allowable_entities = extra_entities - passage_entities

            # add (context, answer) to batch, prepare for QG
            for ent in allowable_entities:
                if ent not in ['-LRB-', '-RRB-', 'LRB', 'RRB']:
                    QG_input_contexts.append(extra_contexts)
                    QG_input_answers.append(ent)

        if len(QG_input_contexts) == 0:
            return None
        
        # Step 3: batch question generation
        QA_pairs = []
        try:
            QA_pairs = self.qg_nlp.batch_qg_with_answer(QG_input_contexts, QG_input_answers)
        except:
            return None


        # Step 4: QA2D
        to_predict = []
        for qa in QA_pairs:
            to_predict.append(qa['question'] + ' [SEP] ' + qa['answer'])

        tmp_results = self.QA2D_model.predict(to_predict)
        if len(tmp_results) == 0:
            return None

        # Step 5: Filteration. 
        # results = [result for result in tmp_results if result.find(first_entity) >= 0]

        # A more relax version: we count the claim if there is at least one overlapping entity
        # results = []
        # for claim in tmp_results:
        #     valid = False
        #     for ent in passage_entities:
        #         if claim.find(ent) >=0:
        #             valid = True
        #             break
        #     if valid == True:
        #         results.append(claim)

        results = tmp_results
            
        generated_samples = []
        claim_set = set()
        for ind in range(len(results)):
            claim = results[ind]
            evidence = texts
            ID = str(uuid.uuid4())
            # don't want repeated claims
            if claim not in claim_set:
                generated_samples.append({'id': ID, 'context': evidence, 'claim': claim, 'label': 'NEI'})
                claim_set.add(claim)
        
        return generated_samples


    def generate_for_FEVER(self, data_range_start, data_range_end):
        synthetic_dataset = []
        for fever_sample in tqdm(self.fever_dataset[data_range_start: data_range_end]):
            sample_list = None
            # claim generation
            if self.claim_type == 'SUPPORTED':
                sample_list = self.generate_support_claims(fever_sample)
            elif self.claim_type == 'REFUTED_LOCAL':
                sample_list = self.generate_refute_local_claims(fever_sample)
            elif self.claim_type == 'REFUTED':
                sample_list = self.generate_refute_global_claims(fever_sample)
            elif self.claim_type == 'NEI':
                sample_list = self.generate_NEI_claims(fever_sample)

            if not sample_list is None:
                synthetic_dataset += sample_list

        random.shuffle(synthetic_dataset)
        with open(self.save_path, 'w') as f:
            f.write(json.dumps(synthetic_dataset, indent=2))

if __name__ == "__main__":
    # parser used to read argument
    parser = argparse.ArgumentParser(description='ClaimGeneration')

    # parameters
    parser.add_argument(
        '--split',
        type=str, help='data split')
    
    parser.add_argument(
        '--train_path',
        type=str, help='path of the FEVER train dataset')

    parser.add_argument(
        '--dev_path',
        type=str, help='path of the FEVER dev dataset')

    parser.add_argument(
        '--wiki_path',
        type=str, help='path to wikipedia database')

    parser.add_argument(
        '--entity_dict',
        type=str, help='path of the entity dict')

    parser.add_argument(
        '--QA_path', 
        type=str, help='path to save the precomputed QAs')

    parser.add_argument(
        '--QA2D_model_path',
        type=str, help='path to the pretrained QA2D model')

    parser.add_argument(
        '--sense_to_vec_path', 
        default='../dependencies/s2v_old',
        type=str, help='path to the pretrained sense2vec model')

    parser.add_argument(
        '--save_path', 
        type=str, help='path to save the generated claims')

    parser.add_argument(
        '--gpu_index', 
        default=0,
        type=int, help='the GPU index used for generation')

    parser.add_argument(
        '--claim_type', 
        default='NEI',
        type=str, help='the type of claim to generate')

    parser.add_argument(
        '--range_start', 
        default=0,
        type=int, help='the start range of the dataset')

    parser.add_argument(
        '--range_end', 
        default=-1,
        type=int, help='the end range of the dataset')
    
    config = parser.parse_args()
    claim_generator = ClaimGenerator(config)

    claim_generator.generate_for_FEVER(config.range_start, config.range_end)