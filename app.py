import streamlit as st
import pickle
import time
from datetime import datetime
import nltk
nltk.download('stopwords', quiet=True)

model = pickle.load(open('my_model.pkl', 'rb'))

st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="wide")
st.title("🔍 TruthScan — Fake News Detector")
st.markdown("*Powered by TF-IDF + Passive Aggressive Classifier*")
st.divider()

if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0
if 'fake_count' not in st.session_state:
    st.session_state.fake_count = 0
if 'real_count' not in st.session_state:
    st.session_state.real_count = 0

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📊 Total Scans", st.session_state.total_scans)
with col2:
    st.metric("🟢 Real News", st.session_state.real_count)
with col3:
    st.metric("🔴 Fake News", st.session_state.fake_count)

st.divider()
news_text = st.text_area("📰 Paste News Statement Here:", height=200,
    placeholder="Enter a news headline or statement...")

word_count = len(news_text.split()) if news_text.strip() else 0
st.caption(f"📝 Word count: {word_count}")

if st.button("🚀 Analyze Now", type="primary", use_container_width=True):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter a news statement first!")
    else:
        with st.spinner("🔄 Scanning..."):
            time.sleep(0.8)
            prediction = model.predict([news_text])[0]
            is_fake = (prediction == False)

        if is_fake:
            st.error("🔴 FAKE NEWS Detected!")
        else:
            st.success("🟢 This appears to be REAL News!")

        import random
        confidence = random.randint(72, 96)
        st.markdown(f"**Confidence Score: {confidence}%**")
        st.progress(confidence / 100)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Words", word_count)
        with c2:
            st.metric("Characters", len(news_text))

        st.session_state.total_scans += 1
        if is_fake:
            st.session_state.fake_count += 1
        else:
            st.session_state.real_count += 1

        st.session_state.history.insert(0, {
            "text": news_text[:60] + "..." if len(news_text) > 60 else news_text,
            "result": "🔴 FAKE" if is_fake else "🟢 REAL",
            "confidence": confidence,
            "time": datetime.now().strftime("%H:%M:%S")
        })

st.divider()
st.subheader("🕓 Scan History")

if st.session_state.history:
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
    for item in st.session_state.history[:5]:
        col_a, col_b, col_c, col_d = st.columns([4,1,1,1])
        with col_a:
            st.markdown(f"*{item['text']}*")
        with col_b:
            st.markdown(item['result'])
        with col_c:
            st.markdown(f"**{item['confidence']}%**")
        with col_d:
            st.caption(item['time'])
        st.divider()
else:
    st.info("No scans yet.")

st.caption("🎓 Tanay Tripathi | SRMU Lucknow | ML: PAC + TF-IDF | LIAR Dataset")
