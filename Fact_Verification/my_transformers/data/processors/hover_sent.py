import json
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from transformers.tokenization_bert import whitespace_tokenize
from transformers.data.processors.utils import DataProcessor

import torch
from torch.utils.data import TensorDataset

IGNORE_INDEX = -100

logger = logging.getLogger(__name__)


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def hover_convert_example_to_features(example, max_seq_length, max_doc_num, max_sent_num, doc_stride, max_query_length, is_training):
    features = []
    sp_set = set(list(map(tuple, example.supporting_facts)))
    all_docs_start_end_facts = []
    all_docs_tokens = []
    for (i, doc_tokens) in enumerate(example.docs_tokens):
        all_doc_tokens, start_end_facts = [], []
        cur_title = example.titles[i]
        for (sent_id, sent_tokens) in enumerate(doc_tokens):
            is_sup_fact = (cur_title, sent_id) in sp_set
            all_sent_tokens = []
            N_tokens = len(all_doc_tokens)
            for (k, token) in enumerate(sent_tokens):
                #orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    all_sent_tokens.append(sub_token)
                    all_doc_tokens.append(sub_token)

            start_end_facts.append((N_tokens, N_tokens+len(all_sent_tokens), is_sup_fact))
        all_docs_start_end_facts.append(start_end_facts)
        all_docs_tokens.append(all_doc_tokens)
    assert len(all_docs_tokens) == max_doc_num, (len(all_docs_tokens))
    assert len(all_docs_start_end_facts) == max_doc_num

    spans = []

    truncated_claim = tokenizer.encode(example.claim_text, add_special_tokens=False, max_length=max_query_length)
    sequence_added_tokens = (
        tokenizer.max_len - tokenizer.max_len_single_sentence + 1
        if "roberta" in str(type(tokenizer)) or "camembert" in str(type(tokenizer))
        else tokenizer.max_len - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair
    num_tok_before_para = len(truncated_claim) + sequence_pair_added_tokens
    all_input_ids, all_attention_masks, all_token_type_ids, all_tokens = [], [], [], []

    for all_doc_tokens in all_docs_tokens:
        span_doc_tokens = all_doc_tokens

        encoded_dict = tokenizer.encode_plus(
            truncated_claim if tokenizer.padding_side == "right" else span_doc_tokens,
            span_doc_tokens if tokenizer.padding_side == "right" else truncated_claim,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            pad_to_max_length=True,
            stride=max_seq_length - doc_stride - len(truncated_claim) - sequence_pair_added_tokens,
            truncation_strategy="only_second",
            return_token_type_ids=True,
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_claim) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        span = spans[-1]
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = np.array(span["token_type_ids"])

        p_mask = np.minimum(p_mask, 1)

        if tokenizer.padding_side == "right":
            # Limit positive values to one
            p_mask = 1 - p_mask

        p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1

        # Set the CLS index to '0'
        p_mask[cls_index] = 0

        all_input_ids.append(span["input_ids"])
        all_attention_masks.append(span["attention_mask"])
        all_token_type_ids.append(span["token_type_ids"])
        all_tokens.append(span["tokens"])

    start_mapping = np.zeros([max_doc_num, max_seq_length, max_sent_num])
    end_mapping = np.zeros([max_doc_num, max_seq_length, max_sent_num])
    # all_mapping = np.zeros([max_doc_num, max_seq_length, max_sent_num])
    is_support = - np.ones([max_doc_num, max_sent_num])
    mask = np.zeros([max_doc_num, max_sent_num])

    for i, start_end_facts in enumerate(all_docs_start_end_facts):
        for j, cur_sp_dp in enumerate(start_end_facts):
            if j >= max_sent_num: break
            if len(cur_sp_dp) == 3:
                start, end, is_sp_flag = tuple(cur_sp_dp)
            else:
                start, end, is_sp_flag, is_gold = tuple(cur_sp_dp)
            
            start, end = start + num_tok_before_para, end + num_tok_before_para
            if end > max_seq_length: break

            if start < end:
                start_mapping[i, start, j] = 1
                end_mapping[i, end-1, j] = 1
                # all_mapping[i, start:end, j] = 1
                is_support[i, j] = int(is_sp_flag)
                mask[i, j] = 1

    while len(all_input_ids) < max_doc_num:
        all_input_ids.append([0] * max_seq_length)
        all_attention_masks.append([1] * max_seq_length)
        all_token_type_ids.append([0] * max_seq_length)

    assert len(all_input_ids) == max_doc_num, (len(all_input_ids))
    assert len(all_attention_masks) == max_doc_num
    assert len(all_token_type_ids) == max_doc_num

    features.append(
        HoverFeatures(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            example_index=0,
            unique_id=0,
            tokens=all_tokens,
            labels=is_support,
            start_mapping=start_mapping,
            end_mapping=end_mapping,
            sent_mask=mask,
            # all_mapping=all_mapping,
        )
    )
    return features


def hover_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def hover_convert_examples_to_features(
    examples, tokenizer, max_seq_length, max_doc_num, max_sent_num, doc_stride, max_query_length, is_training, return_dataset=False, threads=1
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.
    Args:
        examples: list of :class:`~transformers.data.processors.hover.HoverExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi
    Returns:
        list of :class:`~transformers.data.processors.hover.HoverFeatures`
    """

    # Defining helper methods
    examples = examples
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=hover_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            hover_convert_example_to_features,
            max_seq_length=max_seq_length,
            max_doc_num=max_doc_num,
            max_sent_num=max_sent_num,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert hover examples to features",
            )
        )
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(features, total=len(features), desc="add example index and unique id"):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    if return_dataset == "pt":
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        # all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        # all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        # all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)
        all_start_mapping = torch.tensor([f.start_mapping for f in features], dtype=torch.long)
        all_end_mapping = torch.tensor([f.end_mapping for f in features], dtype=torch.long)
        all_sent_mask = torch.tensor([f.sent_mask for f in features], dtype=torch.long)
        # all_all_mapping = torch.tensor([f.all_mapping for f in features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        if not is_training:
            
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids,
                all_start_mapping,
                all_end_mapping,
                all_sent_mask,
                all_example_index,
                # all_all_mapping,
            )
        else:
            # all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            # all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            all_labels = torch.tensor([f.labels for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                # all_start_positions,
                # all_end_positions,
                # all_cls_index,
                # all_p_mask,
                # all_is_impossible,
                all_labels,
                all_start_mapping,
                all_end_mapping,
                all_sent_mask,
                all_example_index,
                # all_all_mapping,
            )

        return features, dataset
    elif return_dataset == "tf":
        if not is_tf_available():
            raise RuntimeError("TensorFlow must be installed to return a TensorFlow dataset.")

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    {
                        "start_position": ex.start_position,
                        "end_position": ex.end_position,
                        "cls_index": ex.cls_index,
                        "p_mask": ex.p_mask,
                        "is_impossible": ex.is_impossible,
                    },
                )

        return tf.data.Dataset.from_generator(
            gen,
            (
                {"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32},
                {
                    "start_position": tf.int64,
                    "end_position": tf.int64,
                    "cls_index": tf.int64,
                    "p_mask": tf.int32,
                    "is_impossible": tf.int32,
                },
            ),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                {
                    "start_position": tf.TensorShape([]),
                    "end_position": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                    "is_impossible": tf.TensorShape([]),
                },
            ),
        )

    return features


class HoverProcessor(DataProcessor):
    """
    Processor for the HoVer data set.
    """

    train_file = None
    dev_file = None

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        if not evaluate:
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
            answers = []
        else:
            answers = [
                {"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
                for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
            ]

            answer = None
            answer_start = None

        return HoverExample(
            qas_id=tensor_dict["id"].numpy().decode("utf-8"),
            claim_text=tensor_dict["claim"].numpy().decode("utf-8"),
            context_text=tensor_dict["context"].numpy().decode("utf-8"),
            answer_text=answer,
            start_position_character=answer_start,
            title=tensor_dict["title"].numpy().decode("utf-8"),
            answers=answers,
        )

    def get_examples_from_dataset(self, dataset, evaluate=False):
        """
        Creates a list of :class:`~transformers.data.processors.hover.HoverExample` using a TFDS dataset.
        Args:
            dataset: The tfds dataset loaded from `tensorflow_datasets.load("squad")`
            evaluate: boolean specifying if in evaluation mode or in training mode
        Returns:
            List of HoverExample
        Examples::
            import tensorflow_datasets as tfds
            dataset = tfds.load("squad")
            training_examples = get_examples_from_dataset(dataset, evaluate=False)
            evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
        """

        if evaluate:
            dataset = dataset["validation"]
        else:
            dataset = dataset["train"]

        examples = []
        for tensor_dict in tqdm(dataset):
            examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))

        return examples

    def get_train_examples(self, data_dir, filename=None):
        """
        Returns the training examples from the data directory.
        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")
        print(os.path.join(data_dir, self.train_file if filename is None else filename))
        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)
        return self._create_examples(input_data, "train")

    def get_dev_examples(self, data_dir, filename=None):
        """
        Returns the evaluation example from the data directory.
        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)
        return self._create_examples(input_data, "dev")

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            qas_id = entry["id"]
            claim = entry["claim"]
            context = entry["context"]
            supporting_facts = entry["supporting_facts"]
            titles, paras = [], []
            for doc in context:
                titles.append(doc[0])
                paras.append(doc[1])

            example = HoverExample(
                qas_id=qas_id,
                claim_text=claim,
                paras=paras,
                titles=titles,
                supporting_facts=supporting_facts,
            )

            examples.append(example)
        return examples


class HoverV1Processor(HoverProcessor):
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"


class HoverExample(object):
    """
    A single training/test example for the HoVer dataset, as loaded from disk.
    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        claim_text,
        paras,
        titles,
        supporting_facts,
    ):
        self.qas_id = qas_id
        self.claim_text = claim_text
        self.paras = paras
        self.titles = titles
        self.supporting_facts = supporting_facts

        docs_tokens = []
        # char_to_word_offset = []

        # Split on whitespace so that different tokens may be attributed to their original position.
        for para in self.paras:
            #prev_is_whitespace = True
            doc_tokens = []
            for sent in para:
                prev_is_whitespace = True
                sent_tokens = []
                for c in sent:
                    if _is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            sent_tokens.append(c)
                        else:
                            sent_tokens[-1] += c
                        prev_is_whitespace = False
                    # char_to_word_offset.append(len(doc_tokens) - 1)
                doc_tokens.append(sent_tokens)
            docs_tokens.append(doc_tokens)

        self.docs_tokens = docs_tokens
        # self.char_to_word_offset = char_to_word_offset

        # # Start and end positions only has a value during evaluation.
        # if start_position_character is not None and not is_impossible:
        #     self.start_position = char_to_word_offset[start_position_character]
        #     self.end_position = char_to_word_offset[
        #         min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
        #     ]


class HoverFeatures(object):
    """
    Single hover example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.hover.HoverExample`
    using the :method:`~transformers.data.processors.hover.hover_convert_examples_to_features` method.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        example_index,
        unique_id,
        tokens,
        labels,
        start_mapping,
        end_mapping,
        sent_mask,
        # all_mapping,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        # self.cls_index = cls_index
        # self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        # self.paragraph_len = paragraph_len
        # self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.labels = labels
        self.start_mapping = start_mapping
        self.end_mapping = end_mapping
        self.sent_mask = sent_mask
        # self.all_mapping = all_mapping
        # self.token_to_orig_map = token_to_orig_map

        # self.start_position = start_position
        # self.end_position = end_position
        # self.is_impossible = is_impossible


class HoverResult(object):
    """
    Constructs a HoverResult which can be used to evaluate a model's output on the HoVer dataset.
    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, logits, probs):
        self.logits = logits
        self.unique_id = unique_id
        self.probs = probs

