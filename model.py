"""Read-only inference for the supplied Spark RandomForest export (no JVM/pickle)."""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

ROOT = Path(__file__).resolve().parent
ALIASES = {
    "dien_tich_dat": "size",
    "dien_tich_su_dung": "living_size",
    "so_phong_ngu": "rooms",
    "so_phong_ve_sinh": "toilets",
    "tong_so_tang": "floors",
    "chieu_ngang": "width",
    "chieu_dai": "length",
    "loai_hinh": "house_type",
    "quan_huyen": "district_name",
    "tinh_trang_noi_that": "furnishing",
    "huong_cua_chinh": "direction",
    "dia_chi": "address",
    "mo_ta": "body",
}
LEGAL = {
    "Đã có sổ": "co_so_rieng",
    "Sổ hồng riêng": "co_so_rieng",
    "Đang chờ sổ": "cho_so",
    "Giấy tờ khác": "khac",
    "Sổ chung / công chứng vi bằng": "so_chung_vi_bang",
    "Giấy tờ viết tay": "giay_tay",
    "Không có sổ": "khong_so",
}
DISTRICT_TYPES = {
    "Quận 12": "suburban",
    "Quận Bình Tân": "suburban",
    "Huyện Nhà Bè": "suburban",
    "Thành phố Thủ Đức": "satellite_city",
    **{f"Huyện {v}": "rural" for v in ["Bình Chánh", "Củ Chi", "Hóc Môn", "Cần Giờ"]},
}


class HouseModel:
    def __init__(self, path=ROOT / "assets" / "forest.json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            self.spec = json.load(stream)
        self.categories = self.spec["categories"]
        self.numeric = self.spec["numeric"]
        self.std = np.array(self.spec["std"])
        self.used = sorted(
            {n[3] for tree in self.spec["trees"] for n in tree if n[1] >= 0}
        )
        self.feature_lookup = {f: i for i, f in enumerate(self.used)}

    def normalize(self, raw):
        if raw.empty:
            raise ValueError("CSV không có dòng dữ liệu.")
        if len(raw) > 5000:
            raise ValueError("Tối đa 5.000 dòng mỗi lần. Hãy chia nhỏ CSV.")
        df = raw.copy().reset_index(drop=True)
        df.columns = df.columns.str.strip()
        if df.columns.duplicated().any():
            raise ValueError("CSV có tên cột trùng nhau.")
        for alias, canonical in ALIASES.items():
            if alias in df:
                if canonical in df:
                    raise ValueError(
                        f"Chỉ giữ một trong hai cột {alias} / {canonical}."
                    )
                df = df.rename(columns={alias: canonical})
        for c in df.select_dtypes(include=["object", "string"]).columns:
            df[c] = df[c].map(lambda x: x.strip() if isinstance(x, str) else x)
            df[c] = df[c].replace({"": np.nan, "Chưa rõ": np.nan})
        for required in ["size", "district_name"]:
            if required not in df:
                raise ValueError(f"Thiếu cột bắt buộc: {required}.")
        if "property_type" in df and not df.property_type.eq("nha_o").all():
            raise ValueError(
                "Mô hình này chỉ hỗ trợ nhà ở (property_type = nha_o), không dùng cho đất/căn hộ."
            )
        bad_district = ~df.district_name.isin(self.categories["district_name"])
        if bad_district.any():
            raise ValueError(
                "Quận/huyện không hợp lệ tại dòng CSV: " + self.rows(bad_district)
            )
        for col in self.numeric:
            if col in df:
                values = pd.to_numeric(df[col], errors="coerce")
                bad = df[col].notna() & ~np.isfinite(values)
                if bad.any():
                    raise ValueError(
                        f"{col} phải là số hữu hạn, dùng dấu chấm thập phân. Dòng: {self.rows(bad)}"
                    )
                df[col] = values
        bad = ~df["size"].between(8, 10000)
        if bad.any():
            raise ValueError(
                "Diện tích đất phải từ 8 đến 10.000 m². Dòng: " + self.rows(bad)
            )
        for col in ["living_size", "width", "length", "rooms", "toilets", "floors"]:
            if col in df:
                bad = df[col].notna() & (
                    df[col] <= 0
                    if col in ["width", "length", "living_size", "floors"]
                    else df[col] < 0
                )
                if bad.any():
                    raise ValueError(
                        f"{col} nằm ngoài miền hợp lệ. Dòng: {self.rows(bad)}"
                    )
        if "living_size" in df and (df.living_size > df["size"] * 20).any():
            raise ValueError("Diện tích sử dụng không được vượt 20 lần diện tích đất.")
        if "legal_group" not in df and "giay_to_phap_ly" in df:
            supplied = df.giay_to_phap_ly
            if (supplied.notna() & ~supplied.isin(LEGAL)).any():
                raise ValueError("Giấy tờ pháp lý không nằm trong danh mục hỗ trợ.")
            df["legal_group"] = supplied.map(LEGAL)
        if "house_type" in df:
            df["house_type"] = df.house_type.replace(
                {"Nhà mặt tiền": "Nhà mặt phố, mặt tiền", "Nhà phố": "Nhà phố liền kề"}
            )

        def add(name, values):
            if name not in df:
                df[name] = values

        def series(name):
            return df.get(name, pd.Series(np.nan, index=df.index))

        for col in [
            "rooms",
            "toilets",
            "floors",
            "width",
            "length",
            "living_size",
            "legal_group",
            "direction",
            "furnishing",
        ]:
            add(col + "_missing", series(col).isna().astype(float))
        add("district_type", df.district_name.map(DISTRICT_TYPES).fillna("inner"))
        add(
            "total_floor_area_est",
            series("living_size").fillna(df["size"] * series("floors")),
        )
        add(
            "is_frontage",
            (
                series("house_type").eq("Nhà mặt phố, mặt tiền")
                | series("char_mat_tien").eq(1)
            ).astype(float),
        )
        add("has_secure_legal", series("legal_group").eq("co_so_rieng").astype(float))
        add("has_project", series("project_name").notna().astype(float))
        if "street_number" in df:
            add(
                "is_alley_address",
                df.street_number.astype(str)
                .str.contains("/", regex=False)
                .astype(float),
            )
        if "ward_id" in df:
            try:
                add(
                    "ward_key",
                    pd.to_numeric(df.ward_id, errors="raise")
                    .astype("Int64")
                    .astype(str),
                )
            except (ValueError, TypeError) as exc:
                raise ValueError("ward_id phải là mã phường dạng số nguyên.") from exc
        if "ward_key" in df:
            add(
                "street_key",
                df.ward_key.astype(str) + "|" + series("street_name").fillna("?"),
            )
        if "is_main_street" in df:
            df["is_main_street"] = df.is_main_street.replace(
                {True: "yes", False: "no", "True": "yes", "False": "no"}
            )
        notes = []
        missing = [
            c for c in self.spec["continuous"] if c not in df or df[c].isna().any()
        ]
        if missing:
            notes.append(
                "Biến thiếu được điền bằng thống kê train: " + ", ".join(missing)
            )
        for col in self.categories:
            if col in df:
                unseen = df[col].notna() & ~df[col].astype(str).isin(
                    self.categories[col]
                )
                if unseen.any():
                    notes.append(
                        f"{col}: {int(unseen.sum())} giá trị chưa gặp khi train; dùng mã unknown."
                    )
        return df, notes

    @staticmethod
    def rows(mask):
        return ", ".join(str(i + 2) for i in np.flatnonzero(mask)[:8])

    def transform(self, df):
        numeric = np.column_stack(
            [
                pd.to_numeric(df[c], errors="coerce")
                .fillna(self.spec["impute"][c])
                .to_numpy(dtype=float)
                if c in df
                else np.full(len(df), self.spec["impute"][c])
                for c in self.numeric
            ]
        )
        blocks = [csr_matrix(numeric)]
        for col, labels in self.categories.items():
            lookup = {v: i for i, v in enumerate(labels)}
            values = (
                df[col].fillna("missing").astype(str)
                if col in df
                else pd.Series("missing", index=df.index)
            )
            indices = values.map(lookup).fillna(len(labels)).to_numpy(dtype=int)
            # Spark: indexer keep adds unknown; OHE keep + dropLast keeps its slot.
            blocks.append(
                csr_matrix(
                    (np.ones(len(df)), (np.arange(len(df)), indices)),
                    shape=(len(df), len(labels) + 1),
                )
            )
        matrix = hstack(blocks, format="csr")
        inverse = np.divide(
            1.0, self.std, out=np.zeros_like(self.std), where=self.std != 0
        )
        return matrix.multiply(inverse).tocsr()

    def predict_vectors(self, matrix):
        values = matrix[:, self.used].toarray()
        result = np.zeros(matrix.shape[0])
        for tree in self.spec["trees"]:
            nodes = np.zeros(matrix.shape[0], dtype=int)
            for _ in range(64):
                active = np.array([tree[n][1] >= 0 for n in nodes])
                if not active.any():
                    break
                rows = np.flatnonzero(active)
                for node_id in np.unique(nodes[rows]):
                    selected = rows[nodes[rows] == node_id]
                    _, left, right, feature, threshold = tree[node_id]
                    go_left = (
                        values[selected, self.feature_lookup[feature]] <= threshold
                    )
                    nodes[selected] = np.where(go_left, left, right)
            else:
                raise ValueError("Cây mô hình không hợp lệ.")
            result += np.array([tree[n][0] for n in nodes])
        return result / len(self.spec["trees"])

    def predict(self, raw):
        df, notes = self.normalize(raw)
        log_price = self.predict_vectors(self.transform(df))
        result = raw.copy().reset_index(drop=True)
        result["gia_du_doan_ty"] = np.exp(log_price)
        result["don_gia_du_doan_trieu_m2"] = np.exp(log_price) * 1000 / df["size"]
        return result, df, notes
