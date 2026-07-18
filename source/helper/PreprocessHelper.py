import os
import json
import logging
import torch
import numpy as np
from collections import defaultdict

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from fairseq.binarizer import Binarizer
from fairseq.data import indexed_dataset

from source.helper.Helper import Helper

logger = logging.getLogger(__name__)


class PreprocessHelper(Helper):
    def __init__(self, params):
        super(PreprocessHelper, self).__init__()
        self.params = params
        self.samples = self._load_samples()
        self.label_hierarchy = self._get_label_hierarchy()

        # Initialize the tokenizer required by the baseline
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def _get_label_hierarchy(self):
        label_taxonomy = self._load_label_taxonomy()
        labels_map = {"root": "Root"}

        for sample in self.samples:
            for label_idx, label in zip(sample["labels_ids"], sample["labels"]):
                labels_map[label_idx] = label

        label_hierarchy = {}
        for parent_idx, children_ids in label_taxonomy.items():
            label_hierarchy[labels_map[parent_idx]] = [labels_map[child] for child in children_ids]

        return label_hierarchy

    def _generate_baseline_global_artifacts(self):
        """
        Generates the Fairseq binaries and PyTorch dictionaries for the entire dataset.
        train.py uses this coupled with split.pt to optimize memory usage.
        """
        logger.info("Generating global artifacts (Fairseq .bin / .pt)...")

        source = []
        labels = []
        label_dict = {}
        label_ids = []
        hiera = defaultdict(set)

        # 1. Tokenization and collection
        for sample in self.samples:
            source.append(self.tokenizer.encode(sample["text"].strip().lower(), truncation=True))
            labels.append(sample["labels"])

        # 2. Dynamic label dictionary
        for l in labels:
            for label in l:
                if label not in label_dict:
                    label_dict[label] = len(label_dict)

        # 3. Construction of multi-hot arrays and slot taxonomy
        for l in labels:
            current_ids = [label_dict[label] for label in l]
            label_ids.append(current_ids)
            for i in range(len(current_ids) - 1):
                hiera[current_ids[i]].add(current_ids[i + 1])

        # 4. Save .pt dictionaries to the root of the data directory
        value_dict = {i: self.tokenizer.encode(v.lower(), add_special_tokens=False) for v, i in label_dict.items()}
        torch.save(value_dict, os.path.join(self.params.data.dir, 'bert_value_dict.pt'))
        torch.save(hiera, os.path.join(self.params.data.dir, 'slot.pt'))

        # 5. Fairseq Binarization
        tok_path = os.path.join(self.params.data.dir, 'tok.txt')
        y_path = os.path.join(self.params.data.dir, 'Y.txt')

        with open(tok_path, 'w', encoding='utf-8') as f:
            for s in source:
                f.write(' '.join(map(str, s)) + '\n')

        with open(y_path, 'w', encoding='utf-8') as f:
            for s in label_ids:
                one_hot = [0] * len(label_dict)
                for i in s:
                    one_hot[i] = 1
                f.write(' '.join(map(str, one_hot)) + '\n')

        for data_name in ['tok', 'Y']:
            txt_file = os.path.join(self.params.data.dir, f'{data_name}.txt')
            bin_file = os.path.join(self.params.data.dir, f'{data_name}.bin')
            idx_file = os.path.join(self.params.data.dir, f'{data_name}.idx')

            offsets = Binarizer.find_offsets(txt_file, 1)
            ds = indexed_dataset.make_builder(
                bin_file,
                impl='mmap',
                vocab_size=self.tokenizer.vocab_size,
            )
            Binarizer.binarize(
                txt_file, None, lambda t: ds.add_item(t),
                offset=0, end=offsets[1], already_numberized=True, append_eos=False
            )
            ds.finalize(idx_file)

            os.remove(txt_file)  # Clean up temporary TXT files

    def run(self):
        # Ensure the directory exists
        os.makedirs(self.params.data.dir, exist_ok=True)

        # save the Taxonomy (.taxonomy)
        with open(f"{self.params.data.dir}{self.params.data.name}.taxonomy", 'w', encoding='utf-8') as f:
            for parent, children in self.label_hierarchy.items():
                line_elements = [parent] + children
                f.write('\t'.join(line_elements) + '\n')

        # save the Vocabulary (_label.vocab)
        unique_labels = set()

        # collect all unique child nodes to build the vocabulary
        for children in self.label_hierarchy.values():
            unique_labels.update(children)

        with open(f"{self.params.data.dir}{self.params.data.name}_label.vocab", 'w', encoding='utf-8') as f:
            for label in sorted(list(unique_labels)):
                f.write(label + '\n')

        logger.info(f"Taxonomy saved to: {self.params.data.dir}{self.params.data.name}.taxonomy")
        logger.info(f"Vocabulary saved to: {self.params.data.dir}{self.params.data.name}_label.vocab")

        # Generation of global training artifacts
        self._generate_baseline_global_artifacts()

        # process and save the JSONL splits per fold
        for fold_idx in self.params.data.folds:
            logger.info(
                f"\nPreprocessing {self.params.data.name} dataset (fold {fold_idx}) with the following "
                f"params\n {OmegaConf.to_yaml(self.params)}\n"
            )

            # Dictionary that PyTorch will use for Dataloader partitioning
            split_dict = {}

            for split in self.params.data.splits:
                split_samples = []
                indices = self._load_split_ids(fold_idx, split)

                # The baseline usually expects the keys 'train', 'val', 'test'
                split_key = 'val' if split == 'dev' else split
                split_dict[split_key] = indices

                for idx in indices:
                    split_samples.append({
                        "doc_token": self.samples[idx]["text"].split(),
                        "doc_label": self.samples[idx]["labels"]
                    })

                # save the specific JSONL for this split and fold
                jsonl_path = f"{self.params.data.dir}{self.params.data.name}_{split}_{fold_idx}.jsonl"
                with open(jsonl_path, 'w', encoding='utf-8') as f:
                    for sample in split_samples:
                        f.write(json.dumps(sample) + '\n')

                logger.info(f"Saved {len(split_samples)} samples to: {jsonl_path}")

            # Save the split index tensor exclusive to this fold
            split_pt_path = f"{self.params.data.dir}split_fold_{fold_idx}.pt"
            torch.save(split_dict, split_pt_path)
            logger.info(f"Fold tensor indices saved to: {split_pt_path}")