"""Package the existing trained artifacts; never retrains or changes Project 1."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from model import ROOT, HouseModel


def metadata(path):
    return json.loads(next((path / "metadata").glob("part-*.txt")).read_text())


def vectors(frame, dimensions):
    rows, columns, values = [], [], []
    for i, vector in enumerate(frame.features):
        indices = vector["indices"] if vector["type"] == 0 else np.arange(dimensions)
        rows.extend([i] * len(indices))
        columns.extend(indices)
        values.extend(vector["values"])
    return csr_matrix((values, (rows, columns)), shape=(len(frame), dimensions))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    source = args.source or next((ROOT.parent / "Project 1").rglob("spark"))
    stages = sorted((source / "pipeline_model" / "stages").iterdir())
    spec = {"numeric": [], "continuous": [], "impute": {}, "categories": {}}
    for stage in stages:
        meta = metadata(stage)
        if "Imputer" in stage.name:
            cols = meta["paramMap"]["inputCols"]
            spec["numeric"].extend(cols)
            spec["impute"].update(pd.read_parquet(stage / "data").iloc[0].to_dict())
            if meta["paramMap"]["strategy"] == "median":
                spec["continuous"] = cols
        elif "StringIndexer" in stage.name:
            labels = pd.read_parquet(stage / "data").iloc[0].labelsArray
            spec["categories"] = dict(
                zip(meta["paramMap"]["inputCols"], [list(v) for v in labels])
            )
        elif "OneHotEncoder" in stage.name:
            sizes = pd.read_parquet(stage / "data").iloc[0].categorySizes
            assert list(sizes) == [len(v) + 1 for v in spec["categories"].values()]
        elif "StandardScaler" in stage.name:
            assert meta["paramMap"]["withMean"] is False
            spec["std"] = (
                pd.read_parquet(stage / "data").iloc[0]["std"]["values"].tolist()
            )
    model_meta = metadata(source / "mo_hinh_tot_nhat")
    assert model_meta["class"].endswith("RandomForestRegressionModel")
    spec["metadata"] = model_meta
    spec["trees"] = []
    nodes = pd.read_parquet(source / "mo_hinh_tot_nhat" / "data")
    for _, group in nodes.groupby("treeID", sort=True):
        tree = []
        for node in sorted(group.nodeData, key=lambda n: n["id"]):
            split = node["split"]
            assert node["id"] == len(tree)
            if node["leftChild"] >= 0:
                assert split["numCategories"] == -1, "Only continuous splits supported"
            tree.append(
                [
                    node["prediction"],
                    node["leftChild"],
                    node["rightChild"],
                    split["featureIndex"],
                    float(split["leftCategoriesOrThreshold"][0])
                    if node["leftChild"] >= 0
                    else 0.0,
                ]
            )
        spec["trees"].append(tree)
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    with gzip.open(assets / "forest.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(spec, stream, ensure_ascii=False)
    model = HouseModel()
    train = pd.read_parquet(source / "train.parquet")
    test = pd.read_parquet(source / "test.parquet")
    train_x = vectors(train, len(spec["std"]))
    test_x = vectors(test, len(spec["std"]))
    train_pred = model.predict_vectors(train_x)
    test_pred = model.predict_vectors(test_x)
    residual = train.label.to_numpy() - train_pred
    reference = pd.DataFrame(
        {"price_billion": np.exp(train.label), "residual_log": residual}
    )
    for col in ["size", "rooms", "floors"]:
        idx = spec["numeric"].index(col)
        reference[col] = train_x[:, idx].toarray().ravel() * spec["std"][idx]
    offset = len(spec["numeric"])
    for col, labels in spec["categories"].items():
        if col in ["district_name", "house_type"]:
            indices = np.asarray(
                train_x[:, offset : offset + len(labels) + 1].argmax(axis=1)
            ).ravel()
            reference[col] = np.array(labels + ["unknown"])[indices]
        offset += len(labels) + 1
    reference["price_per_m2"] = reference.price_billion * 1000 / reference["size"]
    reference.to_parquet(assets / "reference.parquet", index=False)
    error = test.label.to_numpy() - test_pred
    metrics = {
        "n_train": len(train),
        "n_test": len(test),
        "n_features": len(spec["std"]),
        "n_trees": len(spec["trees"]),
        "r2_log": float(
            1 - np.sum(error**2) / np.sum((test.label - test.label.mean()) ** 2)
        ),
        "rmse_log": float(np.sqrt(np.mean(error**2))),
        "mae_log": float(np.mean(np.abs(error))),
        "mae_billion": float(np.mean(np.abs(np.exp(test.label) - np.exp(test_pred)))),
        "residual_train_mean": float(residual.mean()),
        "residual_train_std": float(residual.std(ddof=1)),
        "source": "Project 1 / model_ready/nha_o/spark",
        "target": "ln(price_billion)",
        "source_sha256": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob("part-*")
        },
    }
    # Independent regression check against the notebook's reported held-out metrics.
    assert round(metrics["r2_log"], 4) == 0.8366, metrics
    assert round(metrics["rmse_log"], 4) == 0.3082, metrics
    (assets / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in metrics.items() if k != "source_sha256"}, indent=2)
    )


if __name__ == "__main__":
    main()
