'''
Generate QA pairs in advance
Ouput format: 
{
    'sample_id': {
        'NER_text': [question_text, answer_text], 
        ... 
    }
    ...
}
'''

import json
from dataset_loader import FEVER_Dataset
from T5_QG import pipeline
import argparse

from tqdm import tqdm

# QG NLP object
gpu_index = 0

print('Loading QG module >>>>>>>>')
qg_nlp = pipeline("question-generation", model='valhalla/t5-base-qg-hl', qg_format="highlight", gpu_index = gpu_index)
print('QG module loaded.')

def generate_QA_pairs(args):
    FEVER = FEVER_Dataset(args)
    all_samples = FEVER.FEVER_train if args.data_split == 'train' else FEVER.FEVER_dev

    # load entity dict
    with open(args.entity_dict, 'r') as f:
        entity_dict = json.load(f)

    invalid_sample = 0
    QG_RESULTS = {}
    for sample in tqdm(all_samples):
        entities = entity_dict[str(sample['id'])]
        # create a batch
        sources, answers = [], []
        for ent_text, ent_type in entities:
            sources.append(sample['context'])
            answers.append(ent_text)
        # question generation
        if len(sources) > 0 and len(sources) == len(answers):
            results = []
            try:
                results = qg_nlp.batch_qg_with_answer(sources, answers)
            except:
                invalid_sample += 1

            if len(results) == 0:
                continue
            
            # save results
            result_for_sample = {}
            for ind, QA in enumerate(results):
                ent_text, ent_type = entities[ind]
                question = QA['question']
                answer = QA['answer']
                result_for_sample[f'{ent_text}:::{ent_type}'] = [question, answer]

            QG_RESULTS[str(sample['id'])] = result_for_sample
        else:
            invalid_sample += 1

    print(f'Number of invalid samples: {invalid_sample}')
    with open(args.save_path, 'w') as f:
       f.write(json.dumps(QG_RESULTS, indent=2))

if __name__ == "__main__":
    # parser used to read argument
    parser = argparse.ArgumentParser(description='ClaimGeneration')

    # input files
    parser.add_argument(
        '--train_path',
        type=str, help='path of the FEVER train dataset')

    parser.add_argument(
        '--dev_path',
        type=str, help='path of the FEVER dev dataset')

    parser.add_argument(
        '--data_split',
        type=str, help='data split to process')

    parser.add_argument(
        '--entity_dict',
        type=str, help='path of the entity dict')

    parser.add_argument(
        '--save_path',
        type=str, help='path to save the QG result')

    generate_QA_pairs(parser.parse_args())

