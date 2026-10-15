"""Exploratory anomaly signals using the supplied training reference, not upload-fit thresholds."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from model import ROOT

PRICE_COLUMNS = [
    "price",
    "price_billion",
    "price_per_m2",
    "gia_ban",
    "gia_m2",
    "don_gia",
]
UNITS = ["Tỷ đồng", "Triệu đồng", "VNĐ", "Triệu đồng/m²"]
METHODS = [
    "Composite Score",
    "Residual-z",
    "Vi phạm Min/Max",
    "Khoảng P10–P90",
    "Isolation Forest",
]


def convert_price(raw, column, unit, size):
    values = pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError(
            "Cột giá phải là số hữu hạn lớn hơn 0; không để trống hoặc kèm chữ tỷ/triệu."
        )
    if unit == "Tỷ đồng":
        return values
    if unit == "Triệu đồng":
        return values / 1000
    if unit == "VNĐ":
        return values / 1e9
    if unit == "Triệu đồng/m²":
        return values * size.to_numpy() / 1000
    raise ValueError("Đơn vị giá không hợp lệ.")


class AnomalyDetector:
    def __init__(self, model):
        self.model = model
        self.reference = pd.read_parquet(ROOT / "assets" / "reference.parquet")
        self.mean = self.reference.residual_log.mean()
        self.std = self.reference.residual_log.std()
        self.forest = IsolationForest(n_estimators=150, random_state=42, n_jobs=1)
        reference_x = self.features(
            self.reference, self.reference.price_billion.to_numpy()
        )
        self.forest.fit(reference_x)
        self.reference_scores = np.sort(-self.forest.score_samples(reference_x))
        self.groups = {}
        for keys, group in self.reference.groupby(["district_name", "house_type"]):
            if len(group) >= 30:
                self.groups[keys] = self.summary(group, "Quận × loại nhà")
        for key, group in self.reference.groupby("district_name"):
            if len(group) >= 30:
                self.groups[(key,)] = self.summary(group, "Quận")
        self.fallback = self.summary(self.reference, "Toàn bộ train")

    @staticmethod
    def summary(group, level):
        return (
            *group.price_per_m2.quantile([0.01, 0.10, 0.50, 0.90, 0.99]).tolist(),
            len(group),
            level,
        )

    def features(self, frame, price):
        numeric = []
        for col in ["size", "rooms", "floors"]:
            values = (
                frame[col] if col in frame else pd.Series(np.nan, index=frame.index)
            )
            numeric.append(
                pd.to_numeric(values, errors="coerce")
                .fillna(self.model.spec["impute"][col])
                .to_numpy()
            )
        return np.column_stack(
            [
                np.log(numeric[0]),
                numeric[1],
                numeric[2],
                np.log(price * 1000 / numeric[0]),
            ]
        )

    def detect(
        self,
        raw,
        column,
        unit,
        method,
        z_threshold=3.0,
        contamination=5,
        weights=(0.25, 0.25, 0.25, 0.25),
        score_threshold=60.0,
    ):
        if method not in METHODS:
            raise ValueError("Phương pháp không hợp lệ.")
        if sum(weights) <= 0:
            raise ValueError("Tổng trọng số phải lớn hơn 0.")
        result, df, notes = self.model.predict(raw)
        price = convert_price(raw, column, unit, df["size"])
        ppm2 = price * 1000 / df["size"].to_numpy()
        stats = []
        for row in df.to_dict("records"):
            stats.append(
                self.groups.get(
                    (row["district_name"], row.get("house_type")),
                    self.groups.get((row["district_name"],), self.fallback),
                )
            )
        bounds = np.array([row[:5] for row in stats])
        z = (
            np.log(price) - np.log(result.gia_du_doan_ty.to_numpy()) - self.mean
        ) / self.std
        s1 = np.clip(np.abs(z) / 5, 0, 1)
        s2 = ((ppm2 < bounds[:, 0]) | (ppm2 > bounds[:, 4])).astype(float)
        distance = np.maximum(bounds[:, 1] - ppm2, 0) + np.maximum(
            ppm2 - bounds[:, 3], 0
        )
        s3 = np.clip(distance / bounds[:, 2], 0, 1)
        if_score = -self.forest.score_samples(self.features(df, price))
        s4 = np.searchsorted(self.reference_scores, if_score, side="right") / len(
            self.reference_scores
        )
        composite = (
            np.column_stack([s1, s2, s3, s4]) @ (np.array(weights) / sum(weights)) * 100
        )
        flags = {
            "Composite Score": composite >= score_threshold,
            "Residual-z": np.abs(z) >= z_threshold,
            "Vi phạm Min/Max": s2.astype(bool),
            "Khoảng P10–P90": distance > 0,
            "Isolation Forest": if_score
            > np.quantile(self.reference_scores, 1 - contamination / 100),
        }
        result["gia_thuc_te_ty"] = price
        result["chenh_lech_pct"] = (price / result.gia_du_doan_ty - 1) * 100
        result["residual_z"] = z
        result["diem_tong_hop"] = composite
        result["bat_thuong"] = flags[method]
        result["huong_chenh_lech"] = np.where(
            price < result.gia_du_doan_ty, "Thấp hơn dự đoán", "Cao hơn dự đoán"
        )
        result["nhom_tham_chieu"] = [v[6] for v in stats]
        result["so_mau_tham_chieu"] = [v[5] for v in stats]
        result["p01_trieu_m2"], result["p10_trieu_m2"] = bounds[:, 0], bounds[:, 1]
        result["p90_trieu_m2"], result["p99_trieu_m2"] = bounds[:, 3], bounds[:, 4]
        result["s1"], result["s2"], result["s3"], result["s4"] = s1, s2, s3, s4
        return result, notes
