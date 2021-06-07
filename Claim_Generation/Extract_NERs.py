import json
import stanza
from dataset_loader import FEVER_Dataset
import argparse

from tqdm import tqdm

# Stanza NLP object
stanza_nlp = stanza.Pipeline('en', use_gpu = True)

def entity_extraction_for_FEVER(args):
    FEVER = FEVER_Dataset(args)

    splits = ['train', 'dev']
    for split in splits:
        print(f'Extracting NERs for {split} set of FEVER...')
        all_samples = FEVER.FEVER_train if split == 'train' else FEVER.FEVER_dev
        entity_dict = {}
        for sample in tqdm(all_samples):
            texts = sample['context']
            pass_doc = stanza_nlp(texts)
            passage_entities = [(ent.text, ent.type) for ent in pass_doc.ents]
            entity_dict[sample['id']] = passage_entities

        with open(args.save_path + f'entity_dict_{split}.json', 'w') as f:
            f.write(json.dumps(entity_dict, indent=2))


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
        '--save_path',
        type=str, help='path of the entity dict')

    entity_extraction_for_FEVER(parser.parse_args())

    