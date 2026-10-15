"""Streamlit GUI for Project 1: house-price prediction and anomaly screening."""

import csv
import hashlib
import io
import json

import numpy as np
import pandas as pd
import streamlit as st

from anomaly import METHODS, PRICE_COLUMNS, UNITS, AnomalyDetector
from model import LEGAL, ROOT, HouseModel

st.set_page_config(
    page_title="Nhà Insight · Project 1",
    page_icon="🏡",
    layout="wide",
)
st.markdown(
    f"<style>{(ROOT / 'style.css').read_text()}</style>", unsafe_allow_html=True
)


@st.cache_resource
def load_model():
    return HouseModel()


@st.cache_resource
def load_detector():
    return AnomalyDetector(load_model())


def download_csv(df):
    # Avoid spreadsheet formulas from untrusted free-text fields in exported CSVs.
    safe = df.copy()
    for col in safe.select_dtypes(include=["object", "string"]).columns:
        safe[col] = safe[col].map(
            lambda v: (
                "'" + v
                if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@"))
                else v
            )
        )
    return safe.to_csv(index=False).encode("utf-8-sig")


def sample_frame(with_price=False):
    frame = pd.read_csv(ROOT / "samples" / "nha_co_gia.csv")
    return frame if with_price else frame.drop(columns=["gia_ban"])


def show_notes(notes):
    if notes:
        with st.expander("Chất lượng đầu vào & biến được điền thiếu", expanded=False):
            for note in notes:
                st.write(note)
            st.caption(
                "Thiếu vị trí chi tiết và giá khu vực có thể làm giảm độ chính xác so với tập kiểm thử."
            )


def upload_section(key, with_price=False):
    a, b = st.columns([3, 2])
    with a:
        file = st.file_uploader(
            "Chọn CSV của bạn", type=["csv"], key=key + "_upload", max_upload_size=10
        )
    with b:
        st.download_button(
            "↓ Tải CSV mẫu",
            download_csv(sample_frame(with_price)),
            file_name=f"mau_{key}.csv",
            mime="text/csv",
            key=key + "_sample_download",
        )
        use_sample = st.checkbox("Dùng dữ liệu minh họa", key=key + "_sample")
    if file is not None:
        content = file.getvalue()
        if len(content) > 10 * 1024 * 1024:
            st.error("Dung lượng tối đa 10 MB.")
            return None, None
        try:
            text = content.decode("utf-8-sig")
            names = [v.strip() for v in next(csv.reader(io.StringIO(text)))]
            if len(names) != len(set(names)):
                raise ValueError("Tên cột CSV bị trùng.")
            frame = pd.read_csv(io.StringIO(text), nrows=5001)
            frame.columns = frame.columns.str.strip()
            if frame.empty or len(frame) > 5000:
                raise ValueError("CSV phải có từ 1 đến 5.000 dòng.")
        except (
            UnicodeError,
            ValueError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
            StopIteration,
        ) as exc:
            st.error(f"Không đọc được CSV UTF-8, phân cách bằng dấu phẩy: {exc}")
            return None, None
        fingerprint = hashlib.sha256(content).hexdigest()
    elif use_sample:
        frame = sample_frame(with_price)
        fingerprint = "sample-" + key
        st.info("Dữ liệu giả định để thử giao diện, không phải tin rao thực tế.")
    else:
        st.info(
            "Tải CSV lên hoặc chọn dữ liệu minh họa để bắt đầu. Tối đa 5.000 dòng / 10 MB."
        )
        return None, None
    st.caption(f"{len(frame):,} căn nhà · {len(frame.columns)} cột · Xem trước 10 dòng")
    st.dataframe(frame.head(10), hide_index=True, width="stretch")
    return frame, fingerprint


def result_table(result, name, anomaly=False):
    if anomaly:
        a, b, c = st.columns(3)
        a.metric("Đã phân tích", f"{len(result):,}")
        b.metric("Tin được gắn cờ", f"{result.bat_thuong.sum():,}")
        c.metric("Tỷ lệ gắn cờ", f"{result.bat_thuong.mean():.1%}")
        only = st.checkbox("Chỉ xem tin bất thường", key=name + "_only")
        visible = result[result.bat_thuong] if only else result
        st.scatter_chart(
            result,
            x="gia_du_doan_ty",
            y="gia_thuc_te_ty",
            color="bat_thuong",
            x_label="Giá dự đoán (tỷ đồng)",
            y_label="Giá nhập vào (tỷ đồng)",
        )
    else:
        a, b, c = st.columns(3)
        a.metric("Đã dự đoán", f"{len(result):,} căn")
        b.metric("Giá trung vị", f"{result.gia_du_doan_ty.median():.2f} tỷ")
        c.metric(
            "Giá thấp nhất / cao nhất",
            f"{result.gia_du_doan_ty.min():.2f} / {result.gia_du_doan_ty.max():.2f} tỷ",
        )
        visible = result
        st.bar_chart(result.gia_du_doan_ty.rename("Giá dự đoán (tỷ đồng)"))
    st.dataframe(visible, hide_index=True, width="stretch")
    st.download_button(
        "↓ Tải toàn bộ kết quả CSV",
        download_csv(result),
        file_name=name + ".csv",
        mime="text/csv",
        key=name + "_download",
        type="primary",
    )


try:
    model = load_model()
    metrics = json.loads((ROOT / "assets" / "metrics.json").read_text())
except (OSError, ValueError, KeyError) as exc:
    st.error(
        f"Không nạp được mô hình đi kèm: {exc}. Xem README để đóng gói lại assets."
    )
    st.stop()

PAGES = [
    "Tổng quan",
    "Dự đoán một căn nhà",
    "Dự đoán từ CSV",
    "Phát hiện bất thường",
    "🛡️ Kiểm duyệt tin (Moderator)",
    "Mô hình & hướng dẫn",
]

with st.sidebar:
    st.markdown(
        """
        <div class="brand-container">
            <span class="brand-icon">🏡</span>
            <div>
                <div class="brand-title">NHÀ INSIGHT</div>
                <div class="brand-subtitle">AI Real Estate Engine</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio("Không gian làm việc", PAGES, key="page")
    st.divider()

    st.markdown(
        """
        <div class="status-badge">
            <span class="pulse-dot"></span>
            <span>Random Forest · 50 cây</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Nhà ở TP.HCM · 22 quận/huyện theo địa giới cũ. Không áp dụng cho đất và căn hộ.")
    st.caption("Dữ liệu CSV chỉ xử lý trong phiên; ứng dụng không chủ động lưu file tải lên.")

    st.divider()
    st.markdown(
        """
        <div class="author-card">
            <div class="author-header">
                <div class="author-avatar">QP</div>
                <div>
                    <div class="author-title">Học viên thực hiện</div>
                    <div class="author-name">Đặng Quốc Phong</div>
                </div>
            </div>
            <div class="author-email">✉️ <a href="mailto:phong.dang@samedtech.edu.vn" style="color:#0f766e; text-decoration:none; font-weight:600;">phong.dang@samedtech.edu.vn</a></div>
            <div class="author-role">🎓 Đồ án tốt nghiệp Data Science<br>Trung tâm Tin học · ĐH KHTN TP.HCM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption("NHÀ INSIGHT     /     HOUSE PRICE & ANOMALY DETECTION")

if page == "Tổng quan":
    st.title("Hiểu giá nhà.\nNhìn rõ bất thường.")
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">DỮ LIỆU THÀNH GÓC NHÌN</div>
            <h2>Một không gian cho<br>hai bài toán bất động sản.</h2>
            <p>Ước lượng giá nhà từ đặc điểm thực tế. Sàng lọc và kiểm duyệt những tin rao cần kiểm tra kỹ hơn.</p>
            <div class="hero-tag">
                <span>📍 TP. HỒ CHÍ MINH</span>
                <span>·</span>
                <span>🏡 NHÀ Ở</span>
                <span>·</span>
                <span>🤖 RANDOM FOREST 50 CÂY</span>
                <span>·</span>
                <span>🛡️ MODERATOR CONSOLE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c, d = st.columns(4)
    a.metric("Mẫu huấn luyện", f"{metrics['n_train']:,}")
    b.metric("Mẫu kiểm thử", f"{metrics['n_test']:,}")
    c.metric("R² trên log giá", f"{metrics['r2_log']:.4f}")
    d.metric("Cây quyết định", metrics["n_trees"])

    st.subheader("Bạn muốn bắt đầu từ đâu?")
    cards_info = [
        ("01", PAGES[1], "Nhập vị trí, diện tích và mức giá mong muốn để nhận định giá & khoảng target.", PAGES[1]),
        ("02", PAGES[2], "Xử lý hàng loạt căn nhà từ tệp CSV và tải kết quả định giá về máy.", PAGES[2]),
        ("03", PAGES[3], "So sánh giá rao với 4 tín hiệu bất thường và phân phối nhóm tham chiếu.", PAGES[3]),
        ("04", PAGES[4], "Bàn làm việc của Kiểm duyệt viên: Thẩm định phiếu tin đăng, phê duyệt hoặc ẩn tin vi phạm.", PAGES[4]),
    ]
    for col, (number, title, body, target) in zip(st.columns(4), cards_info):
        with col, st.container(border=True):
            st.caption(number + " / WORKSPACE")
            st.subheader(title)
            st.write(body)
            st.button(
                "Mở công cụ →",
                key="open_" + number,
                on_click=lambda value=target: st.session_state.update(page=value),
            )
    st.info(
        "Kết quả là ước lượng từ giá rao trong dữ liệu học, không phải giá giao dịch hoặc kết luận thẩm định."
    )

elif page == "Dự đoán một căn nhà":
    st.title("Căn nhà này có giá bao nhiêu?")
    st.write(
        "Nhập thông tin bạn biết. Các mục nâng cao có thể để trống; mô hình dùng bộ điền thiếu đã học."
    )

    with st.form("house_form"):
        st.subheader("01 · Vị trí & loại nhà")
        c1, c2 = st.columns(2)
        district = c1.selectbox(
            "Quận / huyện (địa giới cũ)",
            sorted(model.categories["district_name"]),
            key="district",
        )
        house_type = c2.selectbox(
            "Loại hình nhà ở",
            [v for v in model.categories["house_type"] if v != "missing"],
            key="house_type",
        )

        st.subheader("02 · Không gian sống")
        c1, c2, c3 = st.columns(3)
        size = c1.number_input("Diện tích đất (m²)", 8.0, 10000.0, 60.0, key="size")
        living = c2.number_input(
            "Diện tích sử dụng (m²)", min_value=1.0, value=120.0, key="living_size"
        )
        floors = c3.number_input("Tổng số tầng", 1, 50, 2, key="floors")
        rooms = c1.number_input("Số phòng ngủ", 0, 50, 3, key="rooms")
        toilets = c2.number_input("Số phòng vệ sinh", 0, 50, 2, key="toilets")
        width = c3.number_input(
            "Chiều ngang (m, bỏ trống nếu chưa rõ)",
            min_value=0.1,
            value=None,
            key="width",
        )

        st.subheader("03 · Pháp lý & hoàn thiện")
        c1, c2, c3 = st.columns(3)
        legal = c1.selectbox("Giấy tờ pháp lý", ["Chưa rõ"] + list(LEGAL), key="legal")
        furnishing = c2.selectbox(
            "Tình trạng nội thất",
            ["Chưa rõ"] + [v for v in model.categories["furnishing"] if v != "missing"],
            key="furnishing",
        )
        direction = c3.selectbox(
            "Hướng cửa chính",
            ["Chưa rõ"] + [v for v in model.categories["direction"] if v != "missing"],
            key="direction",
        )

        st.subheader("04 · Mức giá mong muốn & Phân tích mục tiêu (Tùy chọn)")
        col_t1, col_t2 = st.columns(2)
        target_price = col_t1.number_input(
            "Mức giá mong muốn / Kỳ vọng của bạn (tỷ VNĐ)",
            min_value=0.1,
            max_value=500.0,
            value=None,
            step=0.1,
            key="input_target_price",
            help="Nhập mức giá bạn kỳ vọng bán hoặc muốn mua. Hệ thống sẽ so sánh với mô hình, nhận xét giá cao/thấp và đề xuất khoảng giá nên target.",
        )
        user_role = col_t2.selectbox(
            "Góc nhìn phân tích & Mục tiêu chiến lược",
            [
                "Chủ nhà / Người bán (Tìm giá chào bán & khoảng chốt lời)",
                "Người mua / Nhà đầu tư (Tìm khoảng đàm phán mua có lợi)",
                "Môi giới / Thẩm định viên (Đối chiếu mặt bằng định giá)",
            ],
            key="input_user_role",
        )

        with st.expander("Thông tin nâng cao · tăng độ đầy đủ của đầu vào"):
            st.caption(
                "Mã phường phải theo nguồn Chợ Tốt của tập train. Địa chỉ/mô tả chỉ lưu kèm kết quả, không tự định vị hay phân tích NLP."
            )
            address = st.text_input("Địa chỉ tham khảo", key="address")
            body = st.text_area("Mô tả căn nhà", key="body")
            c1, c2 = st.columns(2)
            length = c1.number_input(
                "Chiều dài (m)", min_value=0.1, value=None, key="length"
            )
            ward = c2.text_input("Mã phường (ward_id, tùy chọn)", key="ward")
            area_price = c1.number_input(
                "Trung vị giá khu vực (triệu đồng/m²)",
                min_value=0.1,
                value=None,
                key="area_price",
            )
            street = c2.text_input("Tên đường (đúng tên trong dữ liệu)", key="street")
            lat = c1.number_input(
                "Vĩ độ (nếu biết)",
                min_value=10.0,
                max_value=12.0,
                value=None,
                format="%.6f",
                key="lat",
            )
            lon = c2.number_input(
                "Kinh độ (nếu biết)",
                min_value=106.0,
                max_value=108.0,
                value=None,
                format="%.6f",
                key="lon",
            )
        submitted = st.form_submit_button("Ước lượng giá nhà →", type="primary")

    if submitted:
        st.session_state.pop("manual_result", None)
        raw = {
            "district_name": district,
            "house_type": house_type,
            "size": size,
            "living_size": living,
            "floors": floors,
            "rooms": rooms,
            "toilets": toilets,
            "width": width,
            "length": length,
            "giay_to_phap_ly": legal,
            "furnishing": furnishing,
            "direction": direction,
            "address": address,
            "body": body,
            "area_median_price_per_m2": area_price,
            "latitude": lat,
            "longitude": lon,
        }
        if ward:
            raw.update(ward_id=ward, street_name=street or None)
        try:
            result, _, notes = model.predict(pd.DataFrame([raw]))
            st.session_state.manual_result = (result, notes, target_price, user_role)
        except ValueError as exc:
            st.error(str(exc))

    if "manual_result" in st.session_state:
        manual_data = st.session_state.manual_result
        result, notes = manual_data[0], manual_data[1]
        target_price = manual_data[2] if len(manual_data) > 2 else None
        user_role = manual_data[3] if len(manual_data) > 3 else "Chủ nhà / Người bán"

        st.subheader("Kết quả lần dự đoán gần nhất")

        price_val = float(result.gia_du_doan_ty.iloc[0])
        unit_val = float(result.don_gia_du_doan_trieu_m2.iloc[0])
        rmse_log = metrics.get("rmse_log", 0.3082)
        p_low = price_val * np.exp(-rmse_log)
        p_high = price_val * np.exp(rmse_log)

        # Tính toán các dải khoảng giá mục tiêu thực tế
        p_fair_min = round(price_val * 0.93, 2)
        p_fair_max = round(price_val * 1.07, 2)
        p_seller_listing = round(price_val * 1.05, 2)
        p_seller_floor = round(price_val * 0.96, 2)
        p_buyer_bid = round(price_val * 0.91, 2)
        p_buyer_cap = round(price_val * 0.98, 2)
        u_target_min = round(unit_val * 0.93, 1)
        u_target_max = round(unit_val * 1.07, 1)

        # Sleek compact valuation banner
        st.markdown(
            f"""
            <div class="valuation-banner">
                <div>
                    <div class="val-main-title">Ước lượng bởi Random Forest (50 cây)</div>
                    <div class="val-price-text">{price_val:.2f} <span>TỶ ĐỒNG</span></div>
                </div>
                <div class="val-meta-chips">
                    <div class="val-chip">📐 {unit_val:.1f} triệu/m²</div>
                    <div class="val-chip">📍 {result['district_name'].iloc[0]}</div>
                    <div class="val-chip">🏡 {result['house_type'].iloc[0]}</div>
                    <div class="val-chip">🎯 Khoảng Target: {p_fair_min:.2f} ~ {p_fair_max:.2f} tỷ</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Giá mô hình định giá", f"{price_val:.2f} tỷ đồng")
        col_m2.metric("Đơn giá đất ước lượng", f"{unit_val:.1f} triệu/m²")
        col_m3.metric(
            "Vùng giá mục tiêu (Target Band)",
            f"{p_fair_min:.2f} ~ {p_fair_max:.2f} tỷ",
            help="Biên độ dao động giá hợp lý ±7% quanh trung vị",
        )

        # Cho phép người dùng nhập / điều chỉnh mức giá mục tiêu trực tiếp để nhận khuyến nghị tức thì
        st.write("")
        col_adj1, col_adj2 = st.columns([2, 1])
        with col_adj1:
            live_target = st.number_input(
                "🎯 Thử nghiệm & Điều chỉnh mức giá kỳ vọng của bạn (tỷ VNĐ):",
                min_value=0.1,
                max_value=500.0,
                value=float(target_price) if target_price is not None else float(round(price_val, 2)),
                step=0.1,
                key="live_target_price_input",
                help="Thay đổi giá này để xem ngay nhận xét giá cao/thấp và chiến lược đàm phán tức thì mà không cần bấm dự đoán lại.",
            )
        with col_adj2:
            st.write("")
            st.caption(f"Khoảng giá khuyến nghị tự nhiên: **{p_fair_min:.2f} ~ {p_fair_max:.2f} tỷ**")

        target_price = live_target

        # Phân tích chiến lược & Nhận xét so với mức giá mong muốn của người dùng
        if target_price is not None:
            diff = float(target_price - price_val)
            diff_pct = float((diff / price_val) * 100)

            st.write("")
            st.markdown(
                f"""
                <div class="target-box">
                    <div class="target-title">
                        <span>📊</span> ĐỐI CHIẾU MỨC GIÁ MONG MUỐN VỚI THỊ TRƯỜNG
                    </div>
                    <div style="display:flex; flex-wrap:wrap; gap:12px; margin-bottom:12px;">
                        <span style="background:#f1f5f9; padding:4px 10px; border-radius:6px; font-size:0.85rem;">
                            Giá bạn kỳ vọng: <b>{target_price:.2f} tỷ</b>
                        </span>
                        <span style="background:#f1f5f9; padding:4px 10px; border-radius:6px; font-size:0.85rem;">
                            Giá AI định giá: <b>{price_val:.2f} tỷ</b>
                        </span>
                        <span style="background:{'#fee2e2' if diff_pct > 12 else ('#fef3c7' if diff_pct > 5 else ('#dcfce7' if abs(diff_pct) <= 5 else '#e0e7ff'))}; padding:4px 10px; border-radius:6px; font-size:0.85rem; font-weight:700;">
                            Chênh lệch: {diff:+.2f} tỷ ({diff_pct:+.1f}%)
                        </span>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            if diff_pct > 15:
                st.markdown(
                    f"""
                    <div class="verdict-banner verdict-very-high">
                        <div style="font-weight:750; font-size:0.98rem; margin-bottom:4px;">
                            ⚠️ NHẬN XÉT: MỨC GIÁ KỲ VỌNG CAO HƠN ĐÁNG KỂ SO VỚI THỊ TRƯỜNG (+{diff_pct:.1f}%)
                        </div>
                        <div>Mức giá <b>{target_price:.2f} tỷ</b> đang cao hơn giá định giá hợp lý ({price_val:.2f} tỷ) tới <b>{diff:.2f} tỷ đồng</b>.</div>
                        <div style="margin-top:6px; font-size:0.88rem;">
                            <b>Khuyến nghị chiến lược:</b><br>
                            • <b>Thanh khoản:</b> Nguy cơ tin đăng rất khó tiếp cận người mua, tỷ lệ xem nhà thấp và thời gian tồn sàn kéo dài.<br>
                            • <b>Hành động:</b> Nên cân nhắc điều chỉnh mức giá về khoảng target <b>[{p_fair_min:.2f} ~ {p_fair_max:.2f} tỷ]</b>.<br>
                            • Nếu quyết định giữ mức giá này, bạn cần chứng minh được các thặng dư đặc biệt: sổ hồng riêng chính chủ, vị trí lô góc kinh doanh sầm uất hoặc tặng kèm nội thất cao cấp đã hoàn thiện.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif diff_pct > 5:
                st.markdown(
                    f"""
                    <div class="verdict-banner verdict-high">
                        <div style="font-weight:750; font-size:0.98rem; margin-bottom:4px;">
                            📈 NHẬN XÉT: MỨC GIÁ KỲ VỌNG HƠI CAO HƠN MẶT BẰNG (+{diff_pct:.1f}%)
                        </div>
                        <div>Mức giá <b>{target_price:.2f} tỷ</b> nhỉnh hơn giá định giá ({price_val:.2f} tỷ) khoảng <b>{diff:.2f} tỷ đồng</b>.</div>
                        <div style="margin-top:6px; font-size:0.88rem;">
                            <b>Khuyến nghị chiến lược:</b><br>
                            • <b>Vị thế:</b> Rất thích hợp làm <b>Giá chào bán ban đầu (Listing Price)</b> cho người bán.<br>
                            • Bạn sẽ có sẵn biên độ đàm phán 3–5% để "bớt lộc" cho khách mua thiện chí mà thực thu vẫn đạt ngưỡng kỳ vọng quanh {price_val:.2f} tỷ.<br>
                            • Đối với người mua: Nên chủ động trả giá khởi điểm từ vùng {p_buyer_bid:.2f} tỷ để kéo giá về mức cân bằng.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif abs(diff_pct) <= 5:
                st.markdown(
                    f"""
                    <div class="verdict-banner verdict-optimal">
                        <div style="font-weight:750; font-size:0.98rem; margin-bottom:4px;">
                            🎯 NHẬN XÉT: MỨC GIÁ RẤT HỢP LÝ & BÁM SÁT THỊ TRƯỜNG (Chênh {diff_pct:+.1f}%)
                        </div>
                        <div>Mức giá <b>{target_price:.2f} tỷ</b> hoàn toàn trùng khớp với phân phối định giá cân bằng của mô hình ({price_val:.2f} tỷ).</div>
                        <div style="margin-top:6px; font-size:0.88rem;">
                            <b>Khuyến nghị chiến lược:</b><br>
                            • <b>Thanh khoản:</b> Đây là mức giá có độ cạnh tranh cao nhất trong khu vực, cơ hội chốt giao dịch nhanh chóng.<br>
                            • Khuyên người bán giữ vững mức giá này; đối với người mua, đây là mức giá hợp lý có thể xuống cọc không lo bị hớ.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif diff_pct >= -15:
                st.markdown(
                    f"""
                    <div class="verdict-banner verdict-low">
                        <div style="font-weight:750; font-size:0.98rem; margin-bottom:4px;">
                            📉 NHẬN XÉT: MỨC GIÁ KỲ VỌNG THẤP HƠN THỊ TRƯỜNG ({diff_pct:.1f}%)
                        </div>
                        <div>Mức giá <b>{target_price:.2f} tỷ</b> thấp hơn định giá thị trường <b>{abs(diff):.2f} tỷ đồng</b>.</div>
                        <div style="margin-top:6px; font-size:0.88rem;">
                            <b>Khuyến nghị chiến lược:</b><br>
                            • <b>Nếu bạn là Người bán:</b> Phù hợp khi cần thanh khoản dòng tiền gấp (bán nhanh trong 1–2 tuần). Nếu không vội, bạn nên nâng giá lên vùng <b>[{p_fair_min:.2f} ~ {p_fair_max:.2f} tỷ]</b> để tối ưu lợi nhuận.<br>
                            • <b>Nếu bạn là Người mua:</b> Đây là cơ hội mua được bất động sản giá hời (dưới giá trị tài sản), nên khẩn trương khảo sát pháp lý và tiến hành thương lượng.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="verdict-banner verdict-very-low">
                        <div style="font-weight:750; font-size:0.98rem; margin-bottom:4px;">
                            🚨 NHẬN XÉT: MỨC GIÁ THẤP HƠN RẤT NHIỀU SO VỚI THỊ TRƯỜNG ({diff_pct:.1f}%)
                        </div>
                        <div>Mức giá <b>{target_price:.2f} tỷ</b> thấp hơn định giá thị trường tới <b>{abs(diff):.2f} tỷ đồng</b>.</div>
                        <div style="margin-top:6px; font-size:0.88rem;">
                            <b>Khuyến nghị chiến lược:</b><br>
                            • <b>Cảnh báo người bán:</b> Nguy cơ bị ép giá hoặc bán hớ sâu. Hãy tăng giá chào bán lên ít nhất mức cận dưới target {p_fair_min:.2f} tỷ.<br>
                            • <b>Cảnh báo người mua / Moderator:</b> Tin rao giá rẻ bất thường này có thể là bẫy câu khách (tin ảo), dính quy hoạch nặng hoặc giấy tờ tay chưa có sổ đỏ. Cần kiểm tra quy hoạch tại UBND quận trước khi giao dịch.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

        # Hiển thị bảng khoảng giá khuyến nghị theo vai trò
        with st.container(border=True):
            st.markdown("#### 🎯 Khoảng giá mục tiêu & Chiến lược đề xuất (Actionable Targets)")
            col_targ1, col_targ2 = st.columns(2)
            with col_targ1:
                st.markdown(
                    f"""
                    **🏠 Dành cho Người bán / Chủ nhà:**
                    - **Vùng giá niêm yết chào bán (Listing Target):** `{p_seller_listing:.2f} tỷ` *(Dự trù đàm phán bớt lộc)*
                    - **Khoảng giá chốt bán kỳ vọng (Closing Range):** `{p_fair_min:.2f} ~ {p_fair_max:.2f} tỷ`
                    - **Mức giá sàn tối thiểu không nên bán dưới:** `{p_seller_floor:.2f} tỷ`
                    """
                )
            with col_targ2:
                st.markdown(
                    f"""
                    **💼 Dành cho Người mua / Nhà đầu tư:**
                    - **Mức giá nên trả mở đầu (First Bid):** `{p_buyer_bid:.2f} tỷ` *(Đàm phán có lợi nhất)*
                    - **Khoảng giá mua hợp lý (Fair Buy Range):** `{p_fair_min:.2f} ~ {price_val:.2f} tỷ`
                    - **Mức giá trần không nên mua vượt quá:** `{p_buyer_cap:.2f} tỷ`
                    """
                )
            st.caption(f"Đơn giá đất tham chiếu mục tiêu: **{u_target_min:.1f} ~ {u_target_max:.1f} triệu/m²**")

        st.caption(
            "Kết quả thuộc bộ thông tin đã gửi gần nhất. Thay đổi thông tin cần bấm ước lượng lại."
        )
        show_notes(notes)
        with st.expander("Xem bảng thông số kỹ thuật chi tiết"):
            st.dataframe(result, hide_index=True)
        st.download_button(
            "↓ Tải kết quả định giá CSV", download_csv(result), "du_doan_mot_can.csv", "text/csv"
        )

elif page == "Dự đoán từ CSV":
    st.title("Một tệp. Nhiều dự đoán.")
    st.write(
        "Bắt buộc: `district_name`, `size`. Hỗ trợ tên cột tiếng Việt theo mẫu ITViec và tên cột của Project 1."
    )
    frame, fingerprint = upload_section("predict")
    if frame is not None:
        price_columns = [c for c in PRICE_COLUMNS if c in frame]
        if price_columns:
            st.warning(
                "Có cột giá: "
                + ", ".join(price_columns)
                + ". Các cột này không được dùng làm đặc trưng dự đoán."
            )
        if st.button("Dự đoán toàn bộ CSV →", type="primary", key="predict_batch"):
            st.session_state.pop("batch_result", None)
            try:
                with st.spinner("Đang dự đoán bằng 50 cây Random Forest…"):
                    result, _, notes = model.predict(frame)
                    result["khoang_target_min_ty"] = (result["gia_du_doan_ty"] * 0.93).round(2)
                    result["khoang_target_max_ty"] = (result["gia_du_doan_ty"] * 1.07).round(2)
                    detected_prices = [c for c in PRICE_COLUMNS if c in frame]
                    if detected_prices:
                        p_col = detected_prices[0]
                        try:
                            act_price = pd.to_numeric(frame[p_col], errors="coerce")
                            if p_col == "price":
                                act_price = act_price / 1e9
                            elif p_col in ("price_per_m2", "gia_m2", "don_gia"):
                                act_price = (act_price * frame["size"]) / 1000
                            diff_s = (act_price - result["gia_du_doan_ty"]).round(2)
                            pct_s = ((diff_s / result["gia_du_doan_ty"]) * 100).round(1)
                            result["gia_nhap_ty"] = act_price.round(2)
                            result["chenh_lech_gia_ty"] = diff_s
                            result["chenh_lech_pct"] = pct_s
                            def _eval_tag(p):
                                if pd.isna(p):
                                    return "Chưa rõ"
                                if p > 15:
                                    return "Giá cao (+15%) ⚠️"
                                elif p > 5:
                                    return "Hơi cao (+5%..15%)"
                                elif abs(p) <= 5:
                                    return "Hợp lý (±5%) 🎯"
                                elif p >= -15:
                                    return "Giá thấp (-5%..-15%) 📉"
                                else:
                                    return "Thấp bất thường (<-15%) 🚨"
                            result["nhan_xet_gia"] = pct_s.map(_eval_tag)
                        except Exception:
                            pass
                st.session_state.batch_result = (fingerprint, result, notes)
            except ValueError as exc:
                st.error(str(exc))
        stored = st.session_state.get("batch_result")
        if stored and stored[0] == fingerprint:
            st.subheader("Kết quả dự đoán")
            show_notes(stored[2])
            result_table(stored[1], "ket_qua_du_doan")

elif page == "Phát hiện bất thường":
    st.title("Những mức giá cần nhìn kỹ hơn.")
    st.write(
        "Đối chiếu giá nhập vào với mô hình và phân phối đơn giá của nhóm nhà tương đồng trong tập train."
    )
    st.warning(
        "Đây là công cụ sàng lọc. Gắn cờ không đồng nghĩa tin sai, lừa đảo hoặc một cơ hội đầu tư."
    )
    frame, fingerprint = upload_section("anomaly", with_price=True)
    if frame is not None:
        candidates = [c for c in PRICE_COLUMNS if c in frame] + [
            c for c in frame if c not in PRICE_COLUMNS
        ]
        c1, c2, c3 = st.columns(3)
        price_col = c1.selectbox("Cột giá", candidates, key="price_col")
        default_unit = {
            "price": "VNĐ",
            "price_billion": "Tỷ đồng",
            "price_per_m2": "Triệu đồng/m²",
            "gia_m2": "Triệu đồng/m²",
            "don_gia": "Triệu đồng/m²",
        }.get(price_col, "Tỷ đồng")
        unit = c2.selectbox(
            "Đơn vị của cột giá",
            UNITS,
            index=UNITS.index(default_unit),
            key="unit_" + price_col,
        )
        method = c3.selectbox("Phương pháp", METHODS, key="method")
        st.caption(
            "Hãy xác nhận đơn vị; không tự suy đoán đơn vị từ độ lớn của con số. gia_ban mặc định là tỷ đồng trong CSV mẫu."
        )
        z, contamination, threshold, weights = 3.0, 5, 60.0, [0.25] * 4
        if method == "Residual-z":
            z = st.slider("Ngưỡng |Residual-z|", 1.0, 5.0, 3.0, 0.25)
        elif method == "Isolation Forest":
            contamination = st.slider(
                "Ngưỡng theo top % điểm bất thường trong train", 1, 20, 5
            )
        elif method == "Composite Score":
            threshold = st.slider("Ngưỡng điểm tổng hợp (0–100)", 1.0, 100.0, 60.0, 1.0)
            with st.expander("Điều chỉnh trọng số"):
                weights = [
                    col.slider(label, 0.0, 1.0, 0.25, 0.05, key="weight_" + str(i))
                    for i, (col, label) in enumerate(
                        zip(
                            st.columns(4),
                            ["Residual-z", "Min/Max", "P10–P90", "Isolation Forest"],
                        )
                    )
                ]
                st.caption(
                    "Trọng số được chuẩn hóa để tổng bằng 1; không phải xác suất bất thường."
                )
        st.caption(
            "Min/Max dùng P1–P99 làm khung tham chiếu (chưa kèm biểu đồ giá khu vực). P10–P90 dùng nhóm quận × loại nhà ≥ 30 mẫu; thiếu nhóm sẽ lùi lên quận, rồi toàn bộ train."
        )
        settings = (
            fingerprint,
            price_col,
            unit,
            method,
            z,
            contamination,
            threshold,
            tuple(weights),
        )
        if st.button("Phân tích bất thường →", type="primary", key="detect"):
            st.session_state.pop("anomaly_result", None)
            try:
                with st.spinner("Đang tính bốn tín hiệu và đối chiếu nhóm tham chiếu…"):
                    result, notes = load_detector().detect(
                        frame,
                        price_col,
                        unit,
                        method,
                        z,
                        contamination,
                        weights,
                        threshold,
                    )
                st.session_state.anomaly_result = (settings, result, notes)
            except ValueError as exc:
                st.error(str(exc))
        stored = st.session_state.get("anomaly_result")
        if stored and stored[0] == settings:
            st.subheader("Kết quả sàng lọc")
            show_notes(stored[2])
            result_table(stored[1], "ket_qua_bat_thuong", anomaly=True)
        elif stored:
            st.info("Thông số đã thay đổi. Bấm phân tích lại để cập nhật kết quả.")

elif page == "🛡️ Kiểm duyệt tin (Moderator)":
    st.title("🛡️ Bàn làm việc Kiểm duyệt viên (Moderator Workspace)")
    st.write(
        "Không gian thẩm định chuyên nghiệp dành cho Kiểm duyệt viên (Moderator) sàn Nhà Tốt: Rà soát tin rao bị AI cảnh báo, đối chiếu 4 tín hiệu bất thường, ra quyết định phê duyệt hoặc gỡ tin và xuất biên bản kiểm duyệt."
    )

    # Check if we have anomaly records cached in session state
    if "moderator_records" not in st.session_state:
        with st.spinner("Đang nạp và phân tích hàng đợi kiểm duyệt ban đầu..."):
            raw_mod = pd.read_csv(ROOT / "samples" / "nha_co_gia.csv")
            det = load_detector()
            mod_res, _ = det.detect(
                raw_mod, "gia_ban", "Tỷ đồng", "Composite Score", score_threshold=50.0
            )
            st.session_state.moderator_records = mod_res.copy()
            st.session_state.moderator_status = {
                i: "Chờ duyệt" for i in range(len(mod_res))
            }
            st.session_state.moderator_notes = {i: "" for i in range(len(mod_res))}

    records = st.session_state.moderator_records

    # Option to sync from current anomaly detection run if available
    cur_anomaly = st.session_state.get("anomaly_result")
    if cur_anomaly:
        col_sync, _ = st.columns([2, 3])
        if col_sync.button("🔄 Nạp kết quả từ trang 'Phát hiện bất thường' vào hàng đợi kiểm duyệt", key="sync_mod_data"):
            st.session_state.moderator_records = cur_anomaly[1].copy()
            records = st.session_state.moderator_records
            st.session_state.moderator_status = {
                i: "Chờ duyệt" for i in range(len(records))
            }
            st.session_state.moderator_notes = {i: "" for i in range(len(records))}
            st.success(f"Đã nạp {len(records)} tin vào hàng đợi kiểm duyệt!")
            st.rerun()

    # Calculate KPI stats
    n_total = len(records)
    n_flagged = int(records.bat_thuong.sum()) if "bat_thuong" in records else 0
    statuses = st.session_state.moderator_status
    n_approved = sum(1 for v in statuses.values() if v == "Đã phê duyệt ✅")
    n_rejected = sum(1 for v in statuses.values() if v == "Đã từ chối / Ẩn 🚫")
    n_pending = sum(1 for v in statuses.values() if v not in ["Đã phê duyệt ✅", "Đã từ chối / Ẩn 🚫"])

    # Top Metric Ribbon
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Tổng tin trong hàng đợi", f"{n_total:,}")
    k2.metric(
        "Tin bị AI gắn cờ",
        f"{n_flagged:,}",
        delta=f"{n_flagged / n_total:.0%}" if n_total > 0 else "0%",
        delta_color="inverse",
    )
    k3.metric("Chờ xử lý", f"{n_pending:,}")
    k4.metric("Đã phê duyệt ✅", f"{n_approved:,}")
    k5.metric("Đã từ chối / Ẩn 🚫", f"{n_rejected:,}")

    st.write("")
    mod_tab1, mod_tab2 = st.tabs(["🔍 Thẩm định từng phiếu tin đăng", "📋 Danh sách hàng đợi & Xuất biên bản"])

    with mod_tab1:
        # Filter controls for Moderator
        c_f1, c_f2, c_f3 = st.columns(3)
        status_filter = c_f1.selectbox(
            "Lọc theo trạng thái kiểm duyệt",
            ["Tất cả trạng thái", "Chờ duyệt", "Đã phê duyệt ✅", "Cần kiểm tra lại ⚠️", "Đã từ chối / Ẩn 🚫"],
            key="mod_st_filter",
        )
        risk_filter = c_f2.selectbox(
            "Lọc theo cấp độ rủi ro AI",
            [
                "Tất cả cấp độ",
                "Nguy cơ cao (Điểm tổng hợp >= 60 hoặc Gắn cờ)",
                "Nguy cơ vừa (Điểm 30 - 60)",
                "An toàn (Điểm < 30)",
            ],
            key="mod_risk_filter",
        )
        districts_in_mod = ["Tất cả quận/huyện"] + sorted(records["district_name"].dropna().unique().tolist())
        district_filter = c_f3.selectbox(
            "Lọc theo Quận/Huyện",
            districts_in_mod,
            key="mod_dist_filter",
        )

        # Filtered indices
        filtered_indices = []
        for idx in range(len(records)):
            row = records.iloc[idx]
            st_val = statuses.get(idx, "Chờ duyệt")

            # Status match
            if status_filter != "Tất cả trạng thái" and status_filter.replace(" ✅", "").replace(" ⚠️", "").replace(" 🚫", "") not in st_val:
                continue

            # District match
            if district_filter != "Tất cả quận/huyện" and row.get("district_name") != district_filter:
                continue

            # Risk match
            comp_score = float(row.get("diem_tong_hop", 0))
            is_flag = bool(row.get("bat_thuong", False))
            if risk_filter == "Nguy cơ cao (Điểm tổng hợp >= 60 hoặc Gắn cờ)" and not (comp_score >= 60 or is_flag):
                continue
            elif risk_filter == "Nguy cơ vừa (Điểm 30 - 60)" and not (30 <= comp_score < 60 and not is_flag):
                continue
            elif risk_filter == "An toàn (Điểm < 30)" and (comp_score >= 30 or is_flag):
                continue

            filtered_indices.append(idx)

        if not filtered_indices:
            st.info("Không có tin nào thỏa mãn bộ lọc kiểm duyệt.")
        else:
            selected_idx = st.selectbox(
                f"Chọn tin đăng cần thẩm định ({len(filtered_indices)} tin phù hợp)",
                options=filtered_indices,
                format_func=lambda idx: (
                    f"Tin #{idx + 1}: {records.iloc[idx]['district_name']} · {records.iloc[idx]['house_type']} · "
                    f"{float(records.iloc[idx]['gia_thuc_te_ty']):.2f} tỷ "
                    f"[AI: {float(records.iloc[idx].get('diem_tong_hop', 0)):.1f} đ · {statuses.get(idx, 'Chờ duyệt')}]"
                ),
                key="mod_sel_item",
            )

            item = records.iloc[selected_idx]
            cur_status = statuses.get(selected_idx, "Chờ duyệt")
            cur_note = st.session_state.moderator_notes.get(selected_idx, "")

            # Inspection Card
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #e2e8f0; padding-bottom:8px; margin-bottom:12px;">
                        <div>
                            <span style="font-size:0.75rem; font-weight:700; color:#64748b; letter-spacing:0.05em;">PHIẾU THẨM ĐỊNH TIN ĐĂNG #{selected_idx + 1}</span>
                            <div style="font-family:'Outfit', sans-serif; font-size:1.35rem; font-weight:800; color:#064e3b;">
                                {item['district_name']} · {item['house_type']}
                            </div>
                        </div>
                        <div>
                            <span class="badge-status {'badge-approved' if 'phê duyệt' in cur_status.lower() else ('badge-flagged' if 'từ chối' in cur_status.lower() else ('badge-review' if 'kiểm tra' in cur_status.lower() else 'badge-pending'))}">
                                {cur_status}
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_left, col_right = st.columns([1.2, 1])

                with col_left:
                    st.markdown("##### 🏡 Đặc điểm bất động sản")
                    sp1, sp2, sp3 = st.columns(3)
                    sp1.metric("Diện tích đất", f"{item.get('size', 0)} m²")
                    sp2.metric("Diện tích SD", f"{item.get('living_size', item.get('size', 0))} m²")
                    sp3.metric("Số tầng", f"{item.get('floors', 1)} tầng")

                    sp4, sp5, sp6 = st.columns(3)
                    sp4.metric("Phòng ngủ", f"{item.get('rooms', 0)} PN")
                    sp5.metric("Vệ sinh", f"{item.get('toilets', 0)} WC")
                    sp6.metric("Pháp lý", f"{item.get('giay_to_phap_ly', 'Đã có sổ')}")

                    st.write("")
                    st.markdown("##### 💰 Đối chiếu định giá (Giá rao vs AI)")
                    val1, val2, val3 = st.columns(3)
                    val1.metric("Giá người đăng rao", f"{item['gia_thuc_te_ty']:.2f} tỷ")
                    val2.metric("Mô hình AI dự đoán", f"{item['gia_du_doan_ty']:.2f} tỷ")
                    diff_val = item["gia_thuc_te_ty"] - item["gia_du_doan_ty"]
                    diff_pct_val = (diff_val / item["gia_du_doan_ty"]) * 100
                    val3.metric(
                        "Độ lệch thị trường",
                        f"{diff_pct_val:+.1f}%",
                        delta=f"{diff_val:+.2f} tỷ",
                        delta_color="off",
                    )

                with col_right:
                    st.markdown("##### 🛡️ Tín hiệu bất thường từ 4 thuật toán")
                    c_score = float(item.get("diem_tong_hop", 0))
                    z_val = float(item.get("residual_z", 0))
                    p_unit = float(item["gia_thuc_te_ty"] * 1000 / item["size"])
                    p10 = float(item.get("p10_trieu_m2", 0))
                    p90 = float(item.get("p90_trieu_m2", 0))

                    st.markdown(
                        f"""
                        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:12px; margin-bottom:12px;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                                <span style="font-weight:600; font-size:0.85rem;">Điểm tổng hợp bất thường (Composite):</span>
                                <b style="color:{'#dc2626' if c_score >= 60 else ('#d97706' if c_score >= 30 else '#16a34a')};">{c_score:.1f} / 100</b>
                            </div>
                            <div style="background:#e2e8f0; border-radius:6px; height:8px; overflow:hidden; margin-bottom:10px;">
                                <div style="background:{'#dc2626' if c_score >= 60 else ('#d97706' if c_score >= 30 else '#16a34a')}; width:{min(c_score, 100.0)}%; height:100%;"></div>
                            </div>
                            <div style="font-size:0.82rem; color:#475569; line-height:1.45;">
                                • <b>Residual-z:</b> {z_val:+.2f} (S1 = {item.get('s1', 0):.2f})<br>
                                • <b>Đơn giá thực tế:</b> {p_unit:.1f} tr/m² (Khung P10–P90: {p10:.1f} ~ {p90:.1f} tr/m²)<br>
                                • <b>Isolation Forest phân vị:</b> S4 = {item.get('s4', 0) * 100:.1f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # AI Diagnosis reasoning
                    if diff_pct_val < -40:
                        st.error("🚨 **CẢNH BÁO CAO:** Giá rẻ bất thường (> 40% so với mô hình)! Nguy cơ cao là tin ảo, mồi chài lừa cọc hoặc dính tranh chấp nghiêm trọng.")
                    elif diff_pct_val > 50:
                        st.warning("⚠️ **CẢNH BÁO:** Giá rao quá cao (> 50% so với mô hình)! Cần kiểm tra xem người đăng có nhập thừa số 0 hoặc kê giá khống.")
                    elif item.get("s2", 0) > 0:
                        st.warning("⚠️ Đơn giá vượt khung P1–P99 của nhóm quận. Cần rà soát pháp lý.")
                    else:
                        st.success("✓ Các chỉ số nằm trong giới hạn kiểm soát. Không có dấu hiệu vi phạm quy chế đăng tin.")

                st.divider()
                st.markdown("##### ✍️ Quyết định của Kiểm duyệt viên")
                new_note = st.text_input(
                    "Ghi chú kiểm duyệt / Lý do thẩm định",
                    value=cur_note,
                    placeholder="Ví dụ: Đã liên hệ xác minh sổ đỏ; Tin ảo sai giá...",
                    key=f"note_input_{selected_idx}",
                )
                st.session_state.moderator_notes[selected_idx] = new_note

                b_act1, b_act2, b_act3 = st.columns(3)
                if b_act1.button("✅ Phê duyệt & Đưa lên sàn", type="primary", key=f"btn_app_{selected_idx}"):
                    st.session_state.moderator_status[selected_idx] = "Đã phê duyệt ✅"
                    st.success(f"✓ Đã phê duyệt tin #{selected_idx + 1}!")
                    st.rerun()

                if b_act2.button("⚠️ Yêu cầu kiểm tra & Sửa giá", key=f"btn_rev_{selected_idx}"):
                    st.session_state.moderator_status[selected_idx] = "Cần kiểm tra lại ⚠️"
                    st.warning(f"Đã chuyển tin #{selected_idx + 1} sang diện cần sửa giá!")
                    st.rerun()

                if b_act3.button("🚫 Từ chối & Ẩn tin vi phạm", key=f"btn_rej_{selected_idx}"):
                    st.session_state.moderator_status[selected_idx] = "Đã từ chối / Ẩn 🚫"
                    st.error(f"Đã từ chối tin #{selected_idx + 1}!")
                    st.rerun()

    with mod_tab2:
        st.markdown("##### 📋 Bảng tổng hợp hàng đợi kiểm duyệt")
        table_rows = []
        for idx in range(len(records)):
            r = records.iloc[idx]
            table_rows.append({
                "Mã tin": f"TIN-{idx + 1:03d}",
                "Quận": r["district_name"],
                "Loại nhà": r["house_type"],
                "Diện tích (m²)": r["size"],
                "Giá rao (tỷ)": r["gia_thuc_te_ty"],
                "AI định giá (tỷ)": round(r["gia_du_doan_ty"], 2),
                "Chênh lệch (%)": round((r["gia_thuc_te_ty"] - r["gia_du_doan_ty"]) / r["gia_du_doan_ty"] * 100, 1) if r["gia_du_doan_ty"] > 0 else 0,
                "Điểm rủi ro AI": round(r.get("diem_tong_hop", 0), 1),
                "Trạng thái kiểm duyệt": statuses.get(idx, "Chờ duyệt"),
                "Ghi chú": st.session_state.moderator_notes.get(idx, ""),
            })
        df_queue = pd.DataFrame(table_rows)

        quick_col1, quick_col2 = st.columns([2, 1])
        with quick_col1:
            if st.button("⚡ Duyệt nhanh toàn bộ tin an toàn (Điểm rủi ro < 30)"):
                for idx in range(len(records)):
                    if records.iloc[idx].get("diem_tong_hop", 0) < 30 and not records.iloc[idx].get("bat_thuong", False):
                        st.session_state.moderator_status[idx] = "Đã phê duyệt ✅"
                st.success("✓ Đã duyệt nhanh các tin an toàn!")
                st.rerun()

        st.dataframe(df_queue, hide_index=True, width="stretch")

        # CSV Download for audit report
        st.download_button(
            "↓ Xuất biên bản kiểm duyệt CSV (Audit Report)",
            download_csv(df_queue),
            file_name="bien_ban_kiem_duyet_moderator.csv",
            mime="text/csv",
            type="primary",
            key="btn_dl_mod_report",
        )

elif page == "Mô hình & hướng dẫn":
    tab1, tab2, tab3 = st.tabs(
        ["Mô hình & độ tin cậy", "Định dạng CSV", "Phương pháp bất thường"]
    )
    with tab1:
        st.subheader("Mô hình thật từ Project 1")
        st.write(
            "GUI sử dụng **RandomForestRegressionModel** đã lưu từ đồ án, không huấn luyện lại và không tạo giá giả. Bộ tiền xử lý và các cây được xuất thành dữ liệu để chạy trực tiếp bằng NumPy/SciPy, không cần Java/Spark."
        )

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(
                f"""
                <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:1.1rem; margin-bottom:1rem; box-shadow:0 1px 3px rgba(0,0,0,0.03);">
                    <div style="font-weight:700; color:#065f46; margin-bottom:0.6rem; font-size:0.95rem;">📊 Dữ liệu huấn luyện & Cấu hình</div>
                    <div style="display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #f1f5f9; font-size:0.88rem;">
                        <span style="color:#64748b;">Mẫu huấn luyện (Train)</span>
                        <b>{metrics.get('n_train', 24930):,} căn</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #f1f5f9; font-size:0.88rem;">
                        <span style="color:#64748b;">Mẫu kiểm thử (Test)</span>
                        <b>{metrics.get('n_test', 6241):,} căn</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #f1f5f9; font-size:0.88rem;">
                        <span style="color:#64748b;">Số cây quyết định</span>
                        <b>{metrics.get('n_trees', 50)} cây</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding:5px 0; font-size:0.88rem;">
                        <span style="color:#64748b;">Tổng đặc trưng mã hóa</span>
                        <b>{metrics.get('n_features', 7692):,} features</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                f"""
                <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:1.1rem; margin-bottom:1rem; box-shadow:0 1px 3px rgba(0,0,0,0.03);">
                    <div style="font-weight:700; color:#065f46; margin-bottom:0.6rem; font-size:0.95rem;">🎯 Độ chính xác trên tập kiểm thử (Test)</div>
                    <div style="display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #f1f5f9; font-size:0.88rem;">
                        <span style="color:#64748b;">Hệ số xác định R² (log giá)</span>
                        <b style="color:#047857;">{metrics.get('r2_log', 0.8366):.4f}</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #f1f5f9; font-size:0.88rem;">
                        <span style="color:#64748b;">Sai số RMSE (log giá)</span>
                        <b>{metrics.get('rmse_log', 0.3082):.4f}</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #f1f5f9; font-size:0.88rem;">
                        <span style="color:#64748b;">Sai số MAE (log giá)</span>
                        <b>{metrics.get('mae_log', 0.2319):.4f}</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding:5px 0; font-size:0.88rem;">
                        <span style="color:#64748b;">Sai số tuyệt đối MAE (tỷ VNĐ)</span>
                        <b>{metrics.get('mae_billion', 1.3414):.2f} tỷ</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        infographic_path = ROOT / "assets" / "nhatot.jpg"
        if infographic_path.exists():
            with st.expander("🗺️ Xem sơ đồ nghiệp vụ Nhà Tốt & luồng xử lý bài toán", expanded=False):
                st.image(
                    str(infographic_path),
                    caption="Quy trình nghiệp vụ: Dự đoán giá hợp lý & Phát hiện tin đăng bất thường",
                    width=720,
                )

        st.json(
            {
                k: metrics[k]
                for k in [
                    "n_train",
                    "n_test",
                    "n_features",
                    "n_trees",
                    "r2_log",
                    "rmse_log",
                    "mae_log",
                    "mae_billion",
                ]
            }
        )
        st.caption(
            "Các chỉ số tính lại trên 6.241 vector kiểm thử kèm Project 1, không phải đánh giá riêng chất lượng nhập tay. R² và RMSE_log tính trên ln(giá tỷ đồng)."
        )
        st.warning(
            "Notebook còn báo cáo GBT sklearn, nhưng gói bàn giao không chứa mô hình sklearn đó. GUI không dùng chỉ số GBT để đại diện cho Random Forest."
        )
        st.write(
            "Phạm vi: nhà ở thuộc TP.HCM cũ. Các biến vị trí/giá khu vực thiếu được điền theo train, nên dự đoán từ biểu mẫu tối giản có thể kém hơn kết quả kiểm thử. Không có khoảng tin cậy đã hiệu chỉnh."
        )
    with tab2:
        st.write(
            "CSV UTF-8, phân cách dấu phẩy; số dùng dấu chấm thập phân và không có dấu ngăn hàng nghìn. Mỗi dòng là một căn nhà."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    ["district_name / quan_huyen", "Bắt buộc", "Quận Gò Vấp"],
                    ["size / dien_tich_dat", "Bắt buộc, m²", "60"],
                    ["house_type / loai_hinh", "Nên có", "Nhà ngõ, hẻm"],
                    ["living_size / dien_tich_su_dung", "m²", "120"],
                    ["rooms / so_phong_ngu", "Số phòng ngủ", "3"],
                    ["toilets / so_phong_ve_sinh", "Số phòng vệ sinh", "2"],
                    ["floors / tong_so_tang", "Số tầng", "2"],
                    ["width / chieu_ngang; length / chieu_dai", "Mét", "4; 15"],
                    [
                        "giay_to_phap_ly / legal_group",
                        "Nhãn tiếng Việt / mã chuẩn",
                        "Đã có sổ / co_so_rieng",
                    ],
                    [
                        "gia_ban / price_billion",
                        "Giá, chọn rõ đơn vị ở giao diện",
                        "6.5",
                    ],
                    ["price", "Mặc định VNĐ", "6500000000"],
                    [
                        "price_per_m2 / gia_m2 / don_gia",
                        "Mặc định triệu đồng/m²",
                        "108.33",
                    ],
                ],
                columns=["Tên cột", "Ý nghĩa", "Ví dụ"],
            ),
            hide_index=True,
            width="stretch",
        )
        st.write(
            "Tên quận hợp lệ: " + "; ".join(sorted(model.categories["district_name"]))
        )
        st.caption(
            "Có thể cung cấp thêm các biến tiền xử lý đúng schema Project 1; CSV không được chứa hai tên đồng nghĩa cho cùng một biến."
        )
    with tab3:
        st.markdown("""
**Bản tích hợp GUI** sử dụng mô hình giá Random Forest và tập train đi kèm. Đây không phải bản tái lập mô hình bất thường sklearn trong notebook (các artifact đó chưa được bàn giao).

- **Residual-z:** chuẩn hóa sai số log giá bằng trung bình/độ lệch chuẩn residual trên train. Do dùng sai số in-sample, ngưỡng có thể lạc quan; cần bộ calibration độc lập trước vận hành thực tế. S1 = min(|z|/5, 1).
- **Min/Max:** S2 = 1 nếu đơn giá nằm ngoài P1–P99 của nhóm tham chiếu train. Không dùng giá min/max từ biểu đồ 13 tháng vì dữ liệu đó chưa có trong gói.
- **P10–P90:** S3 là khoảng cách ngoài khoảng chia cho trung vị nhóm, chặn ở 1. Quy tắc gắn cờ: nằm ngoài khoảng.
- **Isolation Forest:** fit cố định trên train với log diện tích, phòng ngủ, tầng, log đơn giá. S4 là phân vị điểm so với train. Đây là bộ phát hiện bổ sung cho GUI, không phải artifact đã đánh giá trong notebook.
- **Composite Score:** trung bình có trọng số của S1–S4 × 100. Ngưỡng điều chỉnh được, **không phải xác suất**. Chưa có kiểm định Precision/Recall cho cấu hình GUI này.
""")

st.divider()
st.caption(
    "PROJECT 1 · HOUSE PRICE PREDICTION & ANOMALY DETECTION   /   Chỉ dùng để tham khảo và học tập."
)
