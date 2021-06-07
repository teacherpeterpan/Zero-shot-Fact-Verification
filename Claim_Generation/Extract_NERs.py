import json
import stanza
from dataset_loader import FEVER_Dataset
import argparse

from tqdm import tqdm

# Stanza NLP object
stanza_nlp = stanza.Pipeline('en', use_gpu = True)

def entity_extraction_for_FEVER(args):
    FEVER = FEVER_Dataset(args)
    # all_samples = FEVER.FEVER_train
    all_samples = FEVER.FEVER_dev

    entity_dict = {}
    for sample in tqdm(all_samples):
        texts = sample['context']
        pass_doc = stanza_nlp(texts)
        passage_entities = [(ent.text, ent.type) for ent in pass_doc.ents]
        entity_dict[sample['id']] = passage_entities

    with open(args.save_path, 'w') as f:
       f.write(json.dumps(entity_dict, indent=2))


if __name__ == "__main__":
    # parser used to read argument
    parser = argparse.ArgumentParser(description='ClaimGeneration')

    # input files
    parser.add_argument(
        '--train_path',
        default='/mnt/edward/data/liangming/Projects/FactChecking/FEVER/processed/train_nli.processed.json',
        type=str, help='path of the FEVER train dataset')

    parser.add_argument(
        '--dev_path',
        default='/mnt/edward/data/liangming/Projects/FactChecking/FEVER/processed/dev_nli.processed.json',
        type=str, help='path of the FEVER dev dataset')

    parser.add_argument(
        '--save_path',
        default='./data/entity_dict_dev.json',
        type=str, help='path of the entity dict')

    entity_extraction_for_FEVER(parser.parse_args())

    