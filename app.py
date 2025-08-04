import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Reputation Monitoring System", layout="wide")

# ---------- HEADER ----------
st.markdown("""
# 🚀 **Reputation Monitoring System**
Welcome to the **AI-powered sentiment analysis dashboard**.  
This tool helps you:
- ✅ Analyze **single customer reviews** in real time  
- ✅ Process **bulk reviews** from multiple sources  
- ✅ Visualize sentiment trends & keyword insights  
- ✅ Categorize reviews into meaningful topics  

---
""")

# ---------- MODE SELECTION ----------
mode = st.radio("Choose Mode:", ["Single Review", "Bulk Review (CSV Upload)"])

# ---------- SENTIMENT FUNCTION ----------
def predict_sentiment(text):
    text = text.lower()
    if any(w in text for w in ["good", "excellent", "great", "happy", "love", "satisfied", "fantastic", "awesome", "amazing"]):
        return "Positive"
    elif any(w in text for w in ["bad", "worst", "poor", "broken", "long", "scratched", "disappointed", "terrible", "horrible"]):
        return "Negative"
    else:
        return "Neutral"

# ---------- WORD CLOUD GENERATOR ----------
def generate_wordcloud(reviews, title, color):
    if reviews:
        wc = WordCloud(width=600, height=400, background_color="white", colormap=color).generate(" ".join(reviews))
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.write(f"### {title}")
        st.pyplot(fig)

# ---------- TOPIC MODELING ----------
def label_topic(top_words):
    delivery_keywords = {"delivery", "shipping", "late", "fast", "arrived"}
    quality_keywords = {"quality", "design", "durability", "scratched", "packaging"}
    support_keywords = {"support", "service", "helpful", "rude", "customer"}
    words_set = set(top_words)
    if words_set & delivery_keywords:
        return "Delivery"
    elif words_set & quality_keywords:
        return "Product Quality"
    elif words_set & support_keywords:
        return "Customer Support"
    else:
        return "General Feedback"

def perform_topic_modeling_with_labels(reviews, n_topics=3, n_words=6):
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(reviews)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    words = vectorizer.get_feature_names_out()
    topics = []
    topic_labels = {}
    for idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[-n_words:]]
        label = label_topic(top_words)
        topics.append(f"{label}: {', '.join(top_words)}")
        topic_labels[idx] = label
    topic_assignments = [topic_labels[i] for i in lda.transform(dtm).argmax(axis=1)]
    return topics, topic_assignments

# ---------- SINGLE REVIEW MODE ----------
if mode == "Single Review":
    st.subheader("🔹 Enter a single review for instant analysis")
    user_review = st.text_area("✍️ Type or paste your review here:")
    if st.button("Analyze Sentiment"):
        if user_review.strip() != "":
            sentiment = predict_sentiment(user_review)
            st.success(f"**Predicted Sentiment:** 🎯 {sentiment}")
        else:
            st.warning("⚠️ Please enter some review text to analyze.")

# ---------- BULK REVIEW MODE ----------
else:
    st.subheader("🔹 Upload a CSV file containing multiple reviews")
    uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "ReviewText" not in df.columns:
            st.error("❌ CSV must have a column named 'ReviewText'")
        else:
            # Sentiment Analysis
            df["Predicted Sentiment"] = df["ReviewText"].apply(predict_sentiment)

            st.subheader("✅ Uploaded Data Preview")
            st.dataframe(df.head())

            # Source Filter
            if "Source" in df.columns:
                sources = st.multiselect("🔎 Filter by Source", options=df["Source"].unique(), default=df["Source"].unique())
                df = df[df["Source"].isin(sources)]

            # ---------- SENTIMENT DISTRIBUTION ----------
            st.subheader("📌 Sentiment Distribution")
            pie_fig = px.pie(df, names="Predicted Sentiment", title="Sentiment Breakdown",
                             color="Predicted Sentiment",
                             color_discrete_map={"Positive": "green", "Negative": "red", "Neutral": "gray"})
            st.plotly_chart(pie_fig, use_container_width=True)

            # ---------- TREND OVER TIME ----------
            if "Date" in df.columns:
                try:
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                    trend_data = df.groupby([df["Date"].dt.date, "Predicted Sentiment"]).size().reset_index(name="Count")
                    st.subheader("📈 Sentiment Trend Over Time")
                    line_chart = px.line(trend_data, x="Date", y="Count", color="Predicted Sentiment",
                                         title="Sentiment Trend Over Time")
                    st.plotly_chart(line_chart, use_container_width=True)
                except:
                    st.warning("⚠️ Could not parse dates for trend analysis.")

            # ---------- WORD CLOUDS ----------
            st.subheader("☁️ Keyword Insights")
            pos_reviews = df[df["Predicted Sentiment"] == "Positive"]["ReviewText"].tolist()
            neg_reviews = df[df["Predicted Sentiment"] == "Negative"]["ReviewText"].tolist()
            generate_wordcloud(pos_reviews, "✅ Positive Keywords", "Greens")
            generate_wordcloud(neg_reviews, "❌ Negative Keywords", "Reds")

            # ---------- TOPIC MODELING WITH LABELS ----------
            try:
                topics, topic_assignments = perform_topic_modeling_with_labels(df["ReviewText"].tolist())
                df["Topic"] = topic_assignments

                st.write("### 🏷️ Detected Topics")
                for t in topics:
                    st.write(f"- {t}")

                topic_counts = df["Topic"].value_counts().reset_index()
                topic_counts.columns = ["Topic", "Count"]
                st.subheader("📊 Topic Distribution by Category")
                topic_chart = px.bar(topic_counts, x="Topic", y="Count", color="Topic", title="Reviews per Topic Category")
                st.plotly_chart(topic_chart, use_container_width=True)

            except Exception as e:
                st.warning(f"⚠️ Topic modeling could not run: {e}")

            st.success("✅ Advanced analysis completed successfully!")

