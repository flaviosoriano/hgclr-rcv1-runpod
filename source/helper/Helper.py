import logging
import pickle

from pathlib import Path

import numpy as np
import pandas as pd

from ranx import evaluate, Qrels, Run, fuse



class Helper:
    def __int__(self, params):
        self.params = params


    def _load_relevance_map(self):
        with open(f"{self.params.data.dir}relevance_map.pkl", "rb") as relevances_file:
            data = pickle.load(relevances_file)
        relevance_map = {}
        for text_idx, labels_ids in data.items():
            d = {}
            for label_idx in labels_ids:
                d[f"label_{label_idx}"] = 1.0
            relevance_map[f"text_{text_idx}"] = d
        return relevance_map

    def _get_metrics(self):
        metrics = []
        for metric in self.params.eval.metrics:
            for threshold in self.params.eval.thresholds:
                metrics.append(f"{metric}@{threshold}")
        return metrics

    def _load_labels_cls(self):
        with open(f"{self.params.data.dir}label_cls.pkl", "rb") as cls_file:
            return pickle.load(cls_file)

    def _load_texts_cls(self):
        with open(f"{self.params.data.dir}text_cls.pkl", "rb") as cls_file:
            return pickle.load(cls_file)

    def _load_samples(self):
        with open(f"{self.params.data.dir}samples.pkl", "rb") as samples_file:
            return pickle.load(samples_file)

    def _load_label_taxonomy(self):
        with open(f"{self.params.data.dir}label_taxonomy.pkl", "rb") as label_taxonomy_file:
            return pickle.load(label_taxonomy_file)

    def _load_split_samples(self, fold_idx, split):
        split_ids = self._load_split_ids(fold_idx, split)
        with open(f"{self.params.data.dir}samples.pkl", "rb") as samples_file:
            samples = pickle.load(samples_file)
            return [sample for sample in samples if sample["idx"] in split_ids]

    def _get_ids(self, fold_idx, split):
        with open(f"{self.params.data.dir}fold_{fold_idx}/{split}.pkl", "rb") as ids_file:
            return pickle.load(ids_file)

    def _load_split_ids(self, fold_idx, split):
        with open(f"{self.params.data.dir}fold_{fold_idx}/{split}.pkl", "rb") as ids_file:
            return set(pickle.load(ids_file))



    def _eval_ranking(self, rankings, fold_idx):
        results = []
        relevance_map = self._load_relevance_map()
        metrics = self._get_metrics()
        for split in ["train", "val", "test"]:
            for cls in ["tail", "head"]:
                ranking = rankings[fold_idx][split][cls]
                result = evaluate(
                    Qrels(
                        {key: value for key, value in relevance_map.items() if key in ranking.keys()}
                    ),
                    Run(ranking),
                    metrics
                )
                result = {k: round(v, 3) for k, v in result.items()}
                result["fold_idx"] = fold_idx
                result["split"] = split
                result["cls"] = cls
                results.append(result)
        return pd.DataFrame(results)



    def _checkpoint_rankings(self, rankings, fold_idx):
        ranking_dir = f"{self.params.ranking.dir}{self.params.model.name}_{self.params.data.name}/"
        Path(ranking_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving ranking {fold_idx} on {ranking_dir}")
        with open(f"{ranking_dir}{self.params.model.name}_{self.params.data.name}_{fold_idx}.rnk",
                  "wb") as ranking_file:
            pickle.dump(rankings, ranking_file)



    def _checkpoint_results(self, results, fold_idx):
        result_dir = f"{self.params.result.dir}{self.params.model.name}_{self.params.data.name}/"
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving result for fold {fold_idx} on {result_dir}")
        pd.DataFrame(results).to_csv(
            f"{result_dir}{self.params.model.name}_{self.params.data.name}_{fold_idx}.rts",
            sep='\t', index=False, header=True)






