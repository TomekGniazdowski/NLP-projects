from collections import defaultdict
from datasets import DatasetDict, load_dataset


def read_dataset(langs: list, fracs: list):
    panx_ch = defaultdict(DatasetDict)
    for lang, frac in zip(langs, fracs):
        ds_lang = load_dataset('xtreme', name=f'PAN-X.{lang}')
        for split in ds_lang:
            panx_ch[lang][split] = ds_lang[split].shuffle(0).select(range(int(ds_lang[split].num_rows * frac)))
    return panx_ch


def tokenize_adjust_inputs(dataset, tokenizer):
    dataset_tokenized = tokenizer(dataset['tokens'], is_split_into_words=True, truncation=True)
    dataset_labels = []
    for _id, example_labels in enumerate(dataset['ner_tags']):
        example_word_ids = dataset_tokenized.word_ids(batch_index=_id)
        example_labels_adjusted = adjust_labels(example_word_ids, example_labels)
        assert len(example_labels_adjusted) == len(dataset_tokenized.tokens(batch_index=_id))
        dataset_labels.append(example_labels_adjusted)
    return {'labels': dataset_labels, 'input_ids': dataset_tokenized['input_ids']}


def adjust_labels(example_word_ids, example_labels):
    example_labels_adjusted = []
    word_id_prev = None
    for word_id in example_word_ids:
        if word_id is None or word_id_prev == word_id:
            label = -100
        else:
            label = example_labels[word_id]
        example_labels_adjusted.append(label)
        word_id_prev = word_id
    return example_labels_adjusted