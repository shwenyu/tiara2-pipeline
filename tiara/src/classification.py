"""Two-stage classification driver.

v2.2.0 rewrite notes
--------------------
The original implementation worked but did not scale, for four reasons:

1. ``list(SimpleFastaParser(handle))`` pulled the ENTIRE fasta into RAM before
   a single prediction was made. On the v2.1.3 test panel that is hundreds of
   GiB. Records are now streamed and processed in bounded batches.

2. ``predict_proba`` was called ONCE PER RECORD. Every call pays skorch/torch
   dispatch, a host->device copy and a kernel launch for what is usually a
   handful of 5 kbp fragments. Fragments from a whole batch of records are now
   concatenated into one matrix, predicted in one call, and split back apart
   with ``np.add.reduceat``.

3. ``Parallel(n_jobs=...)`` was re-entered per stage, re-pickling the tf-idf
   model to every worker each time. The executor is now created once with
   ``max_nbytes=None`` and reused.

4. The net was left in TRAIN mode with grad enabled. Inference now runs under
   ``eval()`` + ``torch.inference_mode()``.

Behaviour is unchanged: same chopping, same tf-idf, same mean-over-fragments,
same threshold rule, same output ordering contract.
"""
import gzip
from typing import Dict, Union, List
from contextlib import suppress, contextmanager

import numpy as np
from Bio.SeqIO.FastaIO import SimpleFastaParser
from tqdm import tqdm
from joblib import Parallel, delayed

from skorch import NeuralNetClassifier
import torch

from tiara.src.prediction import Prediction, SingleResult, predict_with_threshold
from tiara.src.models import NNet1, NNet2
from tiara.src.transformations import Transformer, TfidfWeighter
from tiara.src.utilities import parse_params, chop, time_context_manager


def fun(seq, layer, transformer, fragment_len):
    """Chop one sequence and featurise its fragments (kept for compatibility)."""
    chopped = chop(seq, fragment_len)
    return transformer.transform(chopped)


allowed_letters = set("ACGT")

# How many RECORDS are held in flight at once. Memory is roughly
# batch_records * fragments_per_record * 4**k * 4 bytes, so the default is
# deliberately modest for k=8 (65,536 floats = 256 KiB per fragment).
DEFAULT_BATCH_RECORDS = 512


@contextmanager
def _no_grad():
    with torch.inference_mode():
        yield


class Classification:
    """Class that performs classification given neural net and tf-idf models provided.

    Methods
    -------
        classify: classifies an entire fasta file
        classify_iter: yields results batch by batch (streaming, bounded RAM)
    """

    def __init__(
        self,
        min_len: int,
        nnet_weights: List[str],
        params: List[Union[Dict[str, int], str]],
        tfidf: List[str],
        threads: int = 1,
        models=(NNet1, NNet2),
        batch_records: int = DEFAULT_BATCH_RECORDS,
        device: str = None,
    ):
        """Init method.

        Parameters
        ----------
            min_len: minimal length of the sequence to classify
            nnet_weights: a list of paths to nnet weights (one per stage)
            params: per-stage dicts (or paths to .csv param files) carrying
                k, fragment_len, prob_cutoff, hidden_1, hidden_2, dim_out, dropout
            tfidf: a list of folders holding params.txt + model.npy (one per stage)
            models: an iterable of torch model classes describing model used
            batch_records: records featurised + predicted per batch
            device: "cuda", "cpu" or None to auto-detect
        """
        self.threads = threads
        self.batch_records = int(batch_records) if batch_records else DEFAULT_BATCH_RECORDS
        self.params = [
            parse_params(param) if isinstance(param, str) else param for param in params
        ]
        # parse_params returns strings; coerce so 4 ** k and nn.Linear work
        # whether params came from a dict or from a .csv file.
        self.params = [self._coerce_params(p) for p in self.params]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.nnets = []
        for model, nnet_weight, params_dict in zip(models, nnet_weights, self.params):
            params_filtered = {
                key: value
                for key, value in params_dict.items()
                if key not in ["k", "fragment_len", "prob_cutoff", "fname"]
            }
            params_filtered.update({"dim_in": 4 ** params_dict["k"]})
            module = model(**params_filtered)
            net = NeuralNetClassifier(
                module, lr=0.0001, criterion=torch.nn.NLLLoss, device=self.device
            )
            net.initialize()
            net.load_params(f_params=nnet_weight)
            # Inference only: disable dropout and autograd bookkeeping.
            net.module_.eval()
            self.nnets.append(net)
        self.transformers = [
            TfidfWeighter.load_params(tfidf_param_folder)
            for tfidf_param_folder in tfidf
        ]
        self._check_stage_consistency(nnet_weights, tfidf)
        self.min_len = min_len
        self.layers = len(nnet_weights)
        self.predictors = [
            Prediction(
                prob_cutoff=self.params[layer]["prob_cutoff"],
                layer=layer,
                nnet=self.nnets[layer],
                fragment_len=self.params[layer]["fragment_len"],
                k=self.params[layer]["k"],
                tnf=self.transformers[layer],
                transformer=Transformer(
                    fragment_len=self.params[layer]["fragment_len"],
                    k=self.params[layer]["k"],
                    model=self.transformers[layer],
                ),
            )
            for layer in range(self.layers)
        ]
        # One executor for the whole run instead of one per stage per file.
        # max_nbytes=None stops joblib memmapping the tiny idf vector to /tmp.
        self._executor = Parallel(n_jobs=self.threads, max_nbytes=None)

    @staticmethod
    def _coerce_params(params_dict):
        out = {}
        for key, value in params_dict.items():
            if key in ("k", "fragment_len", "hidden_1", "hidden_2", "dim_out"):
                if value is None or str(value).lower() in ("none", ""):
                    out[key] = None
                else:
                    out[key] = int(value)
            elif key in ("prob_cutoff", "dropout"):
                out[key] = float(value)
            else:
                out[key] = value
        return out

    def _check_stage_consistency(self, nnet_weights, tfidf):
        """Fail loudly if a stage's net, tf-idf model and k disagree.

        Mixing e.g. ``second_k-7_*.pkl`` with ``k6-second-stage/model.npy``
        silently produced garbage before -- the shapes only clash if the tf-idf
        vector length happens to differ, and a wrong-but-same-length pairing is
        possible once k lists are configurable. Each stage must be a strict
        (weights, tf-idf, k) triple.
        """
        problems = []
        for layer, (weights, folder, tfidf_model) in enumerate(
            zip(nnet_weights, tfidf, self.transformers)
        ):
            k = int(self.params[layer]["k"])
            tfidf_k = int(tfidf_model.k)
            if tfidf_k != k:
                problems.append(
                    f"stage {layer}: params say k={k} but tf-idf model in "
                    f"{folder} was trained with k={tfidf_k}"
                )
            idf_dim = int(np.asarray(tfidf_model.idfs).shape[0])
            if idf_dim != 4 ** k:
                problems.append(
                    f"stage {layer}: tf-idf idf vector has {idf_dim} entries, "
                    f"expected {4 ** k} for k={k} ({folder})"
                )
            fragment_len = int(self.params[layer]["fragment_len"])
            tfidf_fragment = int(getattr(tfidf_model, "fragment_len", fragment_len))
            if tfidf_fragment != fragment_len:
                problems.append(
                    f"stage {layer}: params say fragment_len={fragment_len} but "
                    f"tf-idf model was fit with {tfidf_fragment} ({folder})"
                )
            if f"k-{k}" not in str(weights) and f"k{k}" not in str(weights):
                problems.append(
                    f"stage {layer}: weights file {weights} does not look like a "
                    f"k={k} model; refusing to guess"
                )
        if problems:
            raise ValueError(
                "inconsistent model set:\n  " + "\n  ".join(problems)
            )

    # ------------------------------------------------------------------
    # batched prediction
    # ------------------------------------------------------------------
    def _predict_batch(self, layer, records):
        """Featurise + predict a whole batch of (desc, seq) records.

        Returns a list of SingleResult in the same order as ``records``.
        """
        if not records:
            return []
        predictor = self.predictors[layer]
        fragment_len = self.params[layer]["fragment_len"]
        do = delayed(fun)
        mats = self._executor(
            do(seq, layer, predictor.transformer, fragment_len) for _, seq in records
        )
        counts = np.array([m.shape[0] for m in mats], dtype=np.int64)
        # One matrix, one predict_proba call, one kernel launch.
        stacked = np.concatenate(mats, axis=0).astype(np.float32, copy=False)
        with _no_grad():
            probs = predictor.nnet.predict_proba(stacked)
        probs = np.asarray(probs, dtype=np.float64)
        # Mean over each record's own fragments without a Python loop.
        starts = np.zeros(len(counts), dtype=np.int64)
        np.cumsum(counts[:-1], out=starts[1:])
        sums = np.add.reduceat(probs, starts, axis=0)
        means = sums / counts.reshape(-1, 1)
        cutoff = predictor.prob_cutoff
        return [
            predict_with_threshold(means[i], records[i], cutoff, layer)
            for i in range(len(records))
        ]

    def _read_records(self, sequences_fname):
        """Stream (desc, seq) pairs, filtered by min_len, without buffering all."""
        opener = gzip.open if sequences_fname.endswith(".gz") else open
        with opener(sequences_fname, "rt") as handle:
            for desc, seq in SimpleFastaParser(handle):
                if len(seq) >= self.min_len:
                    yield desc, seq

    def classify_iter(self, sequences_fname: str, verbose=False):
        """Stream results batch by batch. Memory stays bounded by batch_records."""
        batch = []
        bar = tqdm(desc="classifying", unit="seq") if verbose else None
        try:
            for record in self._read_records(sequences_fname):
                batch.append(record)
                if len(batch) >= self.batch_records:
                    for result in self._classify_batch(batch):
                        yield result
                    if bar is not None:
                        bar.update(len(batch))
                    batch = []
            if batch:
                for result in self._classify_batch(batch):
                    yield result
                if bar is not None:
                    bar.update(len(batch))
        finally:
            if bar is not None:
                bar.close()

    def _classify_batch(self, records):
        """Run both stages over one batch, preserving input order."""
        first = self._predict_batch(0, records)
        second_idx = [
            i for i, prediction in enumerate(first)
            if prediction.cls[0] == "organelle"
        ]
        if not second_idx:
            return first
        second_records = [(first[i].desc, first[i].seq) for i in second_idx]
        second = self._predict_batch(1, second_records)
        for i, snd in zip(second_idx, second):
            fst = first[i]
            assert fst.desc == snd.desc, "Descriptions not the same"
            assert fst.seq == snd.seq, "Sequences not the same"
            first[i] = SingleResult(
                desc=fst.desc,
                seq=fst.seq,
                cls=[fst.cls[0], snd.cls[1]],
                probs=[fst.probs[0], snd.probs[1]],
            )
        return first

    def classify(self, sequences_fname: str, verbose=False) -> List[SingleResult]:
        """Perform a two-step classification over a whole fasta file.

        Kept for API compatibility: it materialises every result. Prefer
        ``classify_iter`` for large inputs.

        Returns
        -------
            predictions: a list of SingleResult objects.
        """
        cont_manager = (
            time_context_manager("Classification") if verbose else suppress()
        )
        with cont_manager:
            return list(self.classify_iter(sequences_fname, verbose=verbose))
