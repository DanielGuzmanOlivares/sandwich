from typing import Dict, Iterable
import pandas as pd


def __compute_partials(
    pred_dict: Dict[str, Iterable], gold_dict: Dict[str, Iterable]
) -> Dict:
    """Computes the partial hits and misses for each dataset and POS tag.

    Parameters
    ----------
    pred_dict : Dict[str, Iterable]
        Dictionary with the predictions, format is key: [list of predicted synsets in order of importance].
    gold_dict : Dict[str, Iterable]
        Dictionary with the gold standard, format is key: [list of valid synsets].

    Returns
    -------
    Dict
        Dictionary with the partial hits and misses for each dataset and POS tag.
    """
    # metrics format
    metrics = {
        "semeval2010": {"total": 0, "ok": 0, "not_ok": 0, "f1": 0},
        "senseval2": {"total": 0, "ok": 0, "not_ok": 0, "f1": 0},
        "senseval3": {"total": 0, "ok": 0, "not_ok": 0, "f1": 0},
        "semeval2013": {"total": 0, "ok": 0, "not_ok": 0, "f1": 0},
        "semeval2015": {"total": 0, "ok": 0, "not_ok": 0, "f1": 0},
        "semeval2007": {"total": 0, "ok": 0, "not_ok": 0, "f1": 0},
        "42D": {"total": 0, "ok": 0, "not_ok": 0, "f1": 0},
        "n": {"total": 0, "ok": 0, "not_ok": 0, "f1": 0},
        "v": {"total": 0, "ok": 0, "not_ok": 0, "f1": 0},
        "a": {"total": 0, "ok": 0, "not_ok": 0, "f1": 0},
        "r": {"total": 0, "ok": 0, "not_ok": 0, "f1": 0},
        "TOTAL": {"total": 0, "ok": 0, "not_ok": 0, "f1": 0},
    }
    ok, not_ok = 0, 0
    for k in pred_dict:
        data_from = k.split(".")[0]  # identification of the ID
        # semeval 2017 has a different format
        if data_from.startswith("d"):
            data_from = "semeval2007"

        # get POS from babelnet synset
        pos = pred_dict[k][0][-1]

        metrics[data_from]["total"] += 1
        metrics[pos]["total"] += 1
        metrics["TOTAL"]["total"] += 1

        if k not in gold_dict:
            continue
        local_ok, local_not_ok = 0, 0
        for c in pred_dict[k]:
            if set([c]) & set(gold_dict[k]):
                local_ok += 1
            else:
                local_not_ok += 1
        ok += local_ok / len(pred_dict[k])
        not_ok += local_not_ok / len(pred_dict[k])
        metrics[data_from]["ok"] += local_ok
        metrics[data_from]["not_ok"] += local_not_ok
        metrics[pos]["ok"] += local_ok
        metrics[pos]["not_ok"] += local_not_ok
        metrics["TOTAL"]["ok"] += local_ok
        metrics["TOTAL"]["not_ok"] += local_not_ok

    return metrics


def eval_f1(
    pred_dict: Dict[str, Iterable], gold_dict: Dict[str, Iterable]
) -> pd.DataFrame:
    """Computes the F1 score for the given predictions and gold standard per part of speech, dataset and in total.

    Parameters
    ----------
    pred_dict : Dict[str, Iterable]
        Dictionary with the predictions, format is key: [list of predicted synsets in order of importance].
    gold_dict : Dict[str, Iterable]
        Dictionary with the gold standard, format is key: [list of valid synsets].

    Returns
    -------
    pd.DataFrame
        DataFrame with the F1 score for each part of speech, dataset and in total.
    """
    metrics = __compute_partials(pred_dict, gold_dict)
    for k in metrics:
        try:
            precission = metrics[k]["ok"] / (metrics[k]["ok"] + metrics[k]["not_ok"])
            recall = metrics[k]["ok"] / metrics[k]["total"]
            metrics[k]["f1"] = 2 * precission * recall / (precission + recall)
        except ZeroDivisionError:
            continue
    non_zero_metrics = {
        k: metrics[k]["f1"] for k in metrics if metrics[k]["total"] != 0
    }
    df = pd.DataFrame.from_dict(non_zero_metrics, orient="index").rename(
        columns={0: "F1"}
    )
    df.F1 = df.F1.round(3) * 100
    return df
