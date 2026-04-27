import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

st.set_page_config(page_title="Clinical Text Classifier Demo", page_icon="🩺", layout="centered")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "cp1252", "latin1"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Khong doc duoc file CSV voi cac encoding da thu")

    required = {"Text", "Label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"File CSV thieu cot: {missing}. Cot hien co: {df.columns.tolist()}")

    df = df[["Text", "Label"]].dropna().copy()
    df["Text"] = df["Text"].astype(str).str.lower()
    return df


@st.cache_resource
def train_models(df: pd.DataFrame):
    x_train_text, x_test_text, y_train, y_test = train_test_split(
        df["Text"],
        df["Label"],
        test_size=0.2,
        random_state=42,
        stratify=df["Label"],
    )

    tfidf = TfidfVectorizer(max_features=5000)
    x_train = tfidf.fit_transform(x_train_text)
    x_test = tfidf.transform(x_test_text)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM (LinearSVC)": LinearSVC(max_iter=2000, random_state=42, dual=False),
        "Naive Bayes": MultinomialNB(alpha=1.0),
    }

    scores = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        scores[name] = accuracy_score(y_test, y_pred)

    return tfidf, models, scores


def main():
    st.title("Demo phan loai ho so benh an")
    st.caption("Nhap noi dung trieu chung, chon mo hinh va bam Du doan")

    csv_path = "Clinical Text Data.csv"
    try:
        df = load_data(csv_path)
    except Exception as ex:
        st.error(f"Loi khi doc du lieu: {ex}")
        return

    with st.spinner("Dang huan luyen mo hinh..."):
        tfidf, models, scores = train_models(df)

    with st.expander("Do chinh xac tren tap test", expanded=False):
        for name, score in scores.items():
            st.write(f"- {name}: {score:.4f}")

    col1, col2 = st.columns([3, 2])
    with col1:
        text_input = st.text_area(
            "Noi dung trieu chung", height=180, placeholder="Vi du: patient has thyroid nodule and neck swelling..."
        )
    with col2:
        model_name = st.selectbox("Chon mo hinh", list(models.keys()))
        predict_btn = st.button("Du doan", use_container_width=True)

    if predict_btn:
        if not text_input.strip():
            st.warning("Vui long nhap noi dung truoc khi du doan")
            return

        text_vec = tfidf.transform([text_input.lower()])
        model = models[model_name]
        pred = model.predict(text_vec)[0]

        st.success(f"Ket qua du doan: {pred}")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(text_vec)[0]
            classes = model.classes_
            prob_df = pd.DataFrame({"Class": classes, "Probability": probs}).sort_values(
                "Probability", ascending=False
            )
            st.subheader("Xac suat theo lop")
            st.dataframe(prob_df, use_container_width=True)


if __name__ == "__main__":
    main()
