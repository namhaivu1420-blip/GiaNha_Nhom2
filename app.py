from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


# -----------------------------------------------------------------------------
# Cấu hình giao diện
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dự đoán giá nhà AI | SVM & K-means",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏠 DỰ ĐOÁN GIÁ NHÀ AI")
st.caption("Ứng dụng học máy sử dụng SVM (SVR) để ước lượng giá và K-means để phân nhóm nhà ở.")


# -----------------------------------------------------------------------------
# Dữ liệu demo: chỉ dùng để minh họa luồng hoạt động, không phải dữ liệu thị trường
# -----------------------------------------------------------------------------
def make_demo_data(n_samples: int = 650, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    area = rng.integers(40, 251, size=n_samples)
    bedrooms = rng.integers(1, 6, size=n_samples)
    distance = rng.uniform(1, 25, size=n_samples).round(1)
    quality = rng.integers(1, 11, size=n_samples)
    year_built = rng.integers(1980, 2026, size=n_samples)
    location = rng.choice(["Trung tâm", "Cận trung tâm", "Ngoại thành"], size=n_samples, p=[0.22, 0.43, 0.35])
    location_bonus = pd.Series(location).map({"Trung tâm": 0.75, "Cận trung tâm": 0.30, "Ngoại thành": 0.0}).to_numpy()
    noise = rng.normal(0, 0.38, size=n_samples)

    price = (
        0.025 * area
        + 0.28 * bedrooms
        - 0.07 * distance
        + 0.12 * quality
        + 0.018 * (year_built - 2000)
        + location_bonus
        + noise
    )
    price = np.maximum(price, 0.65).round(2)

    return pd.DataFrame(
        {
            "Area_m2": area,
            "Bedrooms": bedrooms,
            "Distance_km": distance,
            "Quality": quality,
            "YearBuilt": year_built,
            "Location": location,
            "Price_ty_VND": price,
        }
    )


def clean_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(col).strip() for col in result.columns]
    return result


def format_price(value: float, unit: str) -> str:
    if "tỷ" in unit.lower():
        return f"{value:,.2f} {unit}"
    return f"{value:,.0f} {unit}"


def build_svr_pipeline(frame: pd.DataFrame, feature_cols: list[str], c_value: float, epsilon: float, kernel: str) -> Pipeline:
    numeric_cols = frame[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("svr", SVR(kernel=kernel, C=c_value, epsilon=epsilon, gamma="scale")),
        ]
    )


def normalize_features(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    data = frame[feature_cols].copy()
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in categorical_cols:
        data[col] = data[col].map(lambda value: str(value) if pd.notna(value) else np.nan)
    return data


def make_prediction_form(
    frame: pd.DataFrame,
    feature_cols: list[str],
    key_prefix: str,
    submit_label: str,
) -> tuple[bool, dict[str, Any]]:
    values: dict[str, Any] = {}
    with st.form(f"{key_prefix}_form"):
        st.markdown("Nhập thông tin căn nhà:")
        columns = st.columns(2)
        for index, col in enumerate(feature_cols):
            with columns[index % 2]:
                series = frame[col]
                if pd.api.types.is_numeric_dtype(series):
                    clean = pd.to_numeric(series, errors="coerce").dropna()
                    default = float(clean.median()) if not clean.empty else 0.0
                    minimum = float(clean.min()) if not clean.empty else None
                    maximum = float(clean.max()) if not clean.empty else None
                    if minimum is not None and maximum is not None and minimum == maximum:
                        minimum = None
                        maximum = None
                    step = 1.0 if pd.api.types.is_integer_dtype(series) else 0.1
                    values[col] = st.number_input(
                        col,
                        value=default,
                        min_value=minimum,
                        max_value=maximum,
                        step=step,
                        key=f"{key_prefix}_{col}",
                    )
                else:
                    options = series.dropna().astype(str).value_counts().head(40).index.tolist()
                    if not options:
                        options = ["Chưa xác định"]
                    values[col] = st.selectbox(col, options=options, key=f"{key_prefix}_{col}")
        submitted = st.form_submit_button(submit_label, type="primary", use_container_width=True)
    return submitted, values


# -----------------------------------------------------------------------------
# Nạp dữ liệu
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Thiết lập dữ liệu")
    uploaded_file = st.file_uploader("Tải dữ liệu CSV", type=["csv"], help="Ví dụ: train.csv của Kaggle House Prices, cần có cột giá nhà SalePrice.")

if uploaded_file is None:
    raw_df = make_demo_data()
    data_source = "Dữ liệu demo giả lập"
    default_target = "Price_ty_VND"
    default_unit = "tỷ VNĐ"
else:
    try:
        raw_df = pd.read_csv(uploaded_file)
        raw_df = clean_column_names(raw_df)
        data_source = f"Tệp tải lên: {uploaded_file.name}"
        default_target = "SalePrice" if "SalePrice" in raw_df.columns else ""
        default_unit = "USD" if default_target == "SalePrice" else "đơn vị giá"
    except Exception as exc:
        st.error(f"Không thể đọc tệp CSV: {exc}")
        st.stop()

raw_df = clean_column_names(raw_df)
if raw_df.empty:
    st.error("Tệp dữ liệu không có dòng nào.")
    st.stop()

numeric_targets = raw_df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_targets:
    st.error("Không tìm thấy cột số để làm biến mục tiêu. Hãy kiểm tra cột giá nhà trong CSV.")
    st.stop()

with st.sidebar:
    target_default_index = numeric_targets.index(default_target) if default_target in numeric_targets else 0
    target_col = st.selectbox("Chọn cột giá nhà (y)", numeric_targets, index=target_default_index)
    price_unit = st.text_input("Đơn vị giá", value=default_unit)

# Chuẩn hóa biến mục tiêu và loại các dòng không có giá.
df = raw_df.copy()
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target_col]).reset_index(drop=True)
if len(df) < 10:
    st.error("Cần ít nhất 10 dòng có giá nhà hợp lệ để huấn luyện và đánh giá mô hình.")
    st.stop()

# Chọn đặc trưng SVM, loại cột ID và cột hằng khỏi gợi ý mặc định.
feature_candidates = [
    col for col in df.columns
    if col != target_col and df[col].notna().any() and df[col].nunique(dropna=True) > 1
]
feature_candidates = [col for col in feature_candidates if col.lower() not in {"id", "index"}]
if not feature_candidates:
    st.error("Không còn cột đặc trưng phù hợp sau khi loại cột giá và cột không có biến thiên.")
    st.stop()

if target_col == "SalePrice":
    preferred_features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt", "LotArea"]
elif data_source.startswith("Dữ liệu demo"):
    preferred_features = ["Area_m2", "Bedrooms", "Distance_km", "Quality", "YearBuilt", "Location"]
else:
    preferred_features = feature_candidates[: min(6, len(feature_candidates))]

svm_defaults = [col for col in preferred_features if col in feature_candidates]
if not svm_defaults:
    svm_defaults = feature_candidates[: min(6, len(feature_candidates))]

with st.sidebar:
    st.subheader("Đặc trưng mô hình SVM")
    svm_features = st.multiselect(
        "Chọn thuộc tính đầu vào",
        options=feature_candidates,
        default=svm_defaults,
        help="Chọn những cột dùng để học và dự đoán giá nhà. Không chọn cột giá nhà làm đầu vào.",
    )

numeric_features = [col for col in feature_candidates if pd.api.types.is_numeric_dtype(df[col])]
with st.sidebar:
    st.subheader("Đặc trưng K-means")
    default_kmeans = [col for col in svm_features if col in numeric_features][:4]
    if len(default_kmeans) < 2:
        default_kmeans = numeric_features[: min(4, len(numeric_features))]
    kmeans_features = st.multiselect(
        "Chọn thuộc tính số để phân cụm",
        options=numeric_features,
        default=default_kmeans,
        help="K-means hoạt động trên dữ liệu số. Nên chọn ít nhất 2 thuộc tính có ý nghĩa.",
    )
    n_clusters = st.slider("Số cụm K", min_value=2, max_value=8, value=3, step=1)

if not svm_features:
    st.warning("Hãy chọn ít nhất một thuộc tính đầu vào cho SVM ở thanh bên.")
    st.stop()

st.info(f"**Nguồn dữ liệu:** {data_source}  |  **Số mẫu hợp lệ:** {len(df):,}  |  **Biến mục tiêu:** `{target_col}`")

# Tabs chính
svm_tab, kmeans_tab, data_tab = st.tabs(["🔮 SVM: Dự đoán giá nhà", "🧩 K-means: Phân cụm nhà", "📊 Khám phá dữ liệu"])


# -----------------------------------------------------------------------------
# TAB 1: SVM Regression
# -----------------------------------------------------------------------------
with svm_tab:
    st.subheader("Support Vector Regression (SVR)")
    st.write("SVM ở chế độ hồi quy (SVR) học mối quan hệ giữa các thuộc tính của căn nhà và giá bán để ước lượng giá cho một căn nhà mới.")

    control1, control2, control3 = st.columns(3)
    with control1:
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], index=0, help="RBF phù hợp để mô hình hóa quan hệ phi tuyến.")
    with control2:
        c_value = st.select_slider("Tham số C", options=[0.1, 1.0, 10.0, 50.0, 100.0], value=10.0)
    with control3:
        epsilon = st.select_slider("Epsilon", options=[0.01, 0.05, 0.1, 0.2, 0.5], value=0.1)

    test_ratio = st.slider("Tỷ lệ dữ liệu kiểm tra", min_value=0.1, max_value=0.4, value=0.2, step=0.05, key="svm_test_ratio")
    train_df = df.copy()
    X = normalize_features(train_df, svm_features)
    y = train_df[target_col].astype(float)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=42
        )
        svr_model = build_svr_pipeline(train_df, svm_features, c_value, epsilon, kernel)
        with st.spinner("Đang huấn luyện mô hình SVM..."):
            svr_model.fit(X_train, y_train)
        y_pred = svr_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, y_pred)

        st.markdown("#### Kết quả đánh giá trên tập kiểm tra")
        metric_cols = st.columns(4)
        metric_cols[0].metric("MAE", f"{mae:,.3f}")
        metric_cols[1].metric("MSE", f"{mse:,.3f}")
        metric_cols[2].metric("RMSE", f"{rmse:,.3f}")
        metric_cols[3].metric("R²", f"{r2:.4f}")
        st.caption("MAE, RMSE càng thấp càng tốt; R² càng gần 1 thường cho thấy mô hình giải thích dữ liệu tốt hơn. Kết quả phụ thuộc vào dữ liệu và cách chọn thuộc tính.")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Giá thực tế và giá dự đoán")
            comparison = pd.DataFrame({"Giá thực tế": y_test.to_numpy(), "Giá dự đoán": y_pred})
            st.scatter_chart(comparison, x="Giá thực tế", y="Giá dự đoán", use_container_width=True)
        with right:
            st.markdown("#### Một số dự đoán trên tập kiểm tra")
            preview = pd.DataFrame({
                "Giá thực tế": y_test.to_numpy()[:10],
                "Giá dự đoán": y_pred[:10],
                "Sai số tuyệt đối": np.abs(y_test.to_numpy()[:10] - y_pred[:10]),
            })
            st.dataframe(preview.style.format({"Giá thực tế": "{:,.2f}", "Giá dự đoán": "{:,.2f}", "Sai số tuyệt đối": "{:,.2f}"}), use_container_width=True)

        st.divider()
        st.markdown("#### Thử dự đoán một căn nhà mới")
        submitted, input_values = make_prediction_form(df, svm_features, "svm_predict", "Dự đoán giá nhà bằng SVM")
        if submitted:
            new_house = pd.DataFrame([input_values])
            new_house = normalize_features(new_house, svm_features)
            estimated_price = float(svr_model.predict(new_house)[0])
            st.success(f"Giá nhà ước lượng: **{format_price(estimated_price, price_unit)}**")
            st.caption("Đây là kết quả ước lượng từ mô hình đã huấn luyện, không phải định giá chính thức hay cam kết giá giao dịch.")

    except Exception as exc:
        st.error(f"Không thể huấn luyện SVM với lựa chọn hiện tại: {exc}")
        st.info("Thử chọn lại thuộc tính, kiểm tra cột giá có phải dạng số và bảo đảm dữ liệu có đủ giá trị hợp lệ.")


# -----------------------------------------------------------------------------
# TAB 2: K-means clustering
# -----------------------------------------------------------------------------
with kmeans_tab:
    st.subheader("K-means: Phân nhóm các căn nhà")
    st.write("K-means là thuật toán học không giám sát. Thuật toán gom những căn nhà có đặc điểm đầu vào tương đồng vào cùng một cụm, không trực tiếp dự đoán giá bán.")

    if len(kmeans_features) < 2:
        st.warning("Hãy chọn ít nhất 2 thuộc tính số trong thanh bên để thực hiện phân cụm K-means.")
    elif len(df) <= n_clusters:
        st.warning("Số dòng dữ liệu phải lớn hơn số cụm K.")
    else:
        km_data = df[kmeans_features].apply(pd.to_numeric, errors="coerce")
        km_preprocessor = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        try:
            with st.spinner("Đang chuẩn hóa dữ liệu và phân cụm..."):
                km_scaled = km_preprocessor.fit_transform(km_data)

            # Elbow: WCSS (inertia) cho K từ 1 đến tối đa 8.
            max_k = min(8, len(df) - 1)
            elbow_values = []
            for k in range(1, max_k + 1):
                candidate = KMeans(n_clusters=k, random_state=42, n_init=10)
                candidate.fit(km_scaled)
                elbow_values.append({"Số cụm K": k, "WCSS": float(candidate.inertia_)})
            elbow_df = pd.DataFrame(elbow_values)

            km_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = km_model.fit_predict(km_scaled)
            cluster_result = df.copy()
            cluster_result["Cụm"] = cluster_labels.astype(str)

            col_elbow, col_profile = st.columns([1, 1])
            with col_elbow:
                st.markdown("#### Phương pháp Elbow")
                st.line_chart(elbow_df, x="Số cụm K", y="WCSS", use_container_width=True)
                st.caption("Quan sát điểm mà WCSS bắt đầu giảm chậm để tham khảo số cụm. Elbow không phải lúc nào cũng có một điểm gãy rõ ràng.")
            with col_profile:
                st.markdown("#### Quy mô và giá trung bình từng cụm")
                summary = cluster_result.groupby("Cụm").agg(
                    So_luong=(target_col, "size"),
                    Gia_trung_binh=(target_col, "mean"),
                ).reset_index()
                summary = summary.rename(columns={"So_luong": "Số căn nhà", "Gia_trung_binh": f"Giá trung bình ({price_unit})"})
                st.dataframe(summary.style.format({f"Giá trung bình ({price_unit})": "{:,.2f}"}), use_container_width=True)

            st.markdown("#### Trực quan hóa các cụm")
            if len(kmeans_features) >= 2:
                axis_col1, axis_col2 = st.columns(2)
                with axis_col1:
                    x_axis = st.selectbox("Trục X", kmeans_features, index=0)
                with axis_col2:
                    y_default = 1 if len(kmeans_features) > 1 else 0
                    y_axis = st.selectbox("Trục Y", kmeans_features, index=y_default)
                if x_axis == y_axis:
                    st.warning("Hãy chọn hai thuộc tính khác nhau cho trục X và Y.")
                else:
                    st.scatter_chart(cluster_result, x=x_axis, y=y_axis, color="Cụm", use_container_width=True)

            st.markdown("#### Thử xác định cụm cho một căn nhà mới")
            submitted_km, km_input_values = make_prediction_form(df, kmeans_features, "kmeans_predict", "Xác định cụm của căn nhà")
            if submitted_km:
                new_km = pd.DataFrame([km_input_values])
                new_km = new_km[kmeans_features].apply(pd.to_numeric, errors="coerce")
                new_scaled = km_preprocessor.transform(new_km)
                assigned_cluster = int(km_model.predict(new_scaled)[0])
                st.success(f"Căn nhà mới được xếp vào **Cụm {assigned_cluster}**.")
                cluster_count = int((cluster_labels == assigned_cluster).sum())
                st.caption(f"Cụm {assigned_cluster} hiện có {cluster_count:,} căn nhà trong bộ dữ liệu. Cụm thể hiện sự tương đồng về thuộc tính đã chọn, không phải thứ hạng chất lượng.")

            with st.expander("Xem dữ liệu sau khi gán nhãn cụm"):
                st.dataframe(cluster_result[[*kmeans_features, target_col, "Cụm"]].head(100), use_container_width=True)

        except Exception as exc:
            st.error(f"Không thể thực hiện K-means với lựa chọn hiện tại: {exc}")
            st.info("Hãy kiểm tra các thuộc tính số, tránh cột có toàn bộ giá trị bị thiếu và bảo đảm số dòng lớn hơn K.")


# -----------------------------------------------------------------------------
# TAB 3: Khám phá dữ liệu
# -----------------------------------------------------------------------------
with data_tab:
    st.subheader("Tổng quan bộ dữ liệu")
    c1, c2, c3 = st.columns(3)
    c1.metric("Số dòng", f"{len(df):,}")
    c2.metric("Số thuộc tính", f"{df.shape[1] - 1:,}")
    c3.metric("Giá nhà trung bình", format_price(float(df[target_col].mean()), price_unit))

    st.markdown("#### Xem trước dữ liệu")
    st.dataframe(df.head(30), use_container_width=True)

    st.markdown("#### Thống kê mô tả")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    st.markdown("#### Tải dữ liệu đang sử dụng")
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Tải dữ liệu CSV",
        data=csv_bytes,
        file_name="du_lieu_gia_nha.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Lưu ý: Dữ liệu demo được tạo ngẫu nhiên để minh họa. Khi làm báo cáo hoặc định giá thực tế, hãy sử dụng dữ liệu đáng tin cậy, cập nhật và phù hợp với khu vực nghiên cứu.")
