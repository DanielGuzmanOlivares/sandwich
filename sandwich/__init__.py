import json
from pathlib import Path
from typing import Dict, Iterable, Tuple, List, Optional

from sentence_transformers import CrossEncoder
import torch

from sandwich.utils import check_json_file_exists


class Sandwich:
    def __init__(
        self,
        cross_encoder_nv_path: Path,
        cross_encoder_v_path: Path,
        definitions_path: Path,
        neighbours_path: Path,
        device: str = "cpu",
    ):
        """Initialize the Sandwich model predictor.

        Parameters
        ----------
        cross_encoder_nv_path : Path
            Path to the non-verbs cross-encoder model.
        cross_encoder_v_path : Path
            Path to the verbs cross-encoder model.
        definitions_path : Path
            Path to the definitions file. A JSON file in which the keys are babelnet synsets and the
            values are the defintions of the synsets.
        neighbours_path : Path
            Path to the neighbours file. A JSON file in which the keys are synsets and the values are a list of
            the synsets neighbours.
        device : str, optional
            The device in which the models will be stored and operate, by default 'cpu'.

        Raises
        ------
        FileNotFoundError
            If the definitions_path or the neighbours_path do not exist or are not JSON files.
        """
        self.cross_encoder_nv = CrossEncoder(
            cross_encoder_nv_path,
            num_labels=2,
            device=device,
        )
        self.cross_encoder_v = CrossEncoder(
            cross_encoder_v_path,
            num_labels=2,
            device=device,
        )
        if not check_json_file_exists(definitions_path):
            raise FileNotFoundError(
                f"Defintions file {definitions_path} not found or not a JSON file."
            )
        if not check_json_file_exists(neighbours_path):
            raise FileNotFoundError(
                f"Neighbours file {neighbours_path} not found or not a JSON file."
            )
        with open(definitions_path) as f:
            self.definitions = json.load(f)
        with open(neighbours_path) as f:
            self.neighbours = json.load(f)

    def __get_clusters(
        self, synsets: Iterable[str]
    ) -> Tuple[Dict[str, Iterable[int]], Iterable[str]]:
        """Returns a list of all the neighbours associated with the given synsets and a dictionary with the
        indexes of the neighbours in the list.

        Parameters
        ----------
        synsets : Iterable[str]
            The synsets for which the neighbours will be retrieved.

        Returns
        -------
        Tuple[Dict[str, Iterable[int]], Iterable[str]]
            A dictionary of the shape {synset: [indexes of the neighbours in the list]} and a list of all the
            neighbours.
        """
        clusters_idx = {}
        all_synsets = []
        for s in synsets:
            clusters_idx[s] = [
                i
                for i in range(
                    len(all_synsets), len(all_synsets) + len(self.neighbours[s])
                )
            ]
            all_synsets += self.neighbours[s]
        return clusters_idx, all_synsets

    def __get_pairs(
        self, sentence: str, synsets: Iterable, word: str
    ) -> List[Tuple[str, str]]:
        """Generate a list of (sentence, candidate synset definition) pairs of sentences and
        definitions to be used in the cross-encoder model.

        Parameters
        ----------
        sentence : str
            The sentence for which the `word` appears in context.
        synsets : Iterable
            The candidate synsets for the `word` appearing in the `sentence`.
        word : str
            Keyword to be disambiguated.

        Returns
        -------
        List[Tuple[str, str]]
            A list of (sentence, candidate synset definition) pairs.
        """
        pairs = []
        for s in synsets:
            pairs.append(
                [self.__add_target_tokens(sentence, word), self.definitions[s]]
            )
        return pairs

    def __add_target_tokens(self, sentence: str, target: str) -> str:
        """Add the target tokens to the sentence to mark the keyword to the encoder models. If the word is not
        found in the sentece, the function will try to find the closest word containing the `target`. If no word
        contains the `target` then the sentene is returned as is.

        Parameters
        ----------
        sentence : str
            The sentence
        target : str
            _description_

        Returns
        -------
        str
            The sentence with the target tokens added if the target is found in the sentence. Otherwise, closest fully
            containing word is used as target. If no word contains the target, the sentence is returned as is.
        """
        # find target index
        try:
            target_index = sentence.index(target)
        except ValueError:
            i = 0
            for s in sentence:
                if target in s:
                    target_index = i
                    break
                i += 1

        if target_index == len(sentence) - 1:
            return " ".join(sentence)
        else:
            return " ".join(
                sentence[:target_index]
                + ["[D]"]
                + [target]
                + ["[D]"]
                + sentence[target_index + 1 :]
            )

    def disambiguate(
        self,
        sentence: str,
        word: str,
        synsets: Iterable[str],
        batch_size: Optional[int] = 32,
    ) -> str:
        """Disambiguate the given `word` in the `sentence` by selecting the most likely synset from the provided
        `synsets` list.

        Parameters
        ----------
        sentence : str
            The sentence in which the `word` appears.
        word : str
            The word to be disambiguated.
        synsets : Iterable[str]
            The candidate synsets for the `word`.

        Returns
        -------
        str
            The most likely synset from the `synsets` list.
        """
        clusters_idx, all_synsets = self.__get_clusters(synsets)
        pairs = self.__get_pairs(sentence, all_synsets, word)
        pred_labels_nv = self.cross_encoder_nv.predict(
            pairs,
            show_progress_bar=False,
            convert_to_tensor=True,
            batch_size=batch_size,
        )
        pred_labels_v = self.cross_encoder_v.predict(
            pairs,
            show_progress_bar=False,
            convert_to_tensor=True,
            batch_size=batch_size,
        )
        scores = {}
        for s in synsets:
            norm_1 = torch.softmax(pred_labels_nv[clusters_idx[s]], dim=1)
            norm_2 = torch.softmax(pred_labels_v[clusters_idx[s]], dim=1)
            combined = (norm_1 + norm_2) / 2
            top_idxs = torch.topk(abs(combined[:, 1]), len(combined)).indices
            score = combined[top_idxs, 1] * torch.softmax(
                abs(combined[top_idxs, 1] - combined[top_idxs, 0]), dim=0
            )
            scores[s] = score.sum()
        synset = max(scores, key=scores.get)
        return synset
