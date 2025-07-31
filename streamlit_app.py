# streamlit_app.py
import streamlit as st
import librosa
import numpy as np
import pandas as pd
import tempfile
import os
import joblib
import json
import gc
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import warnings
warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────────
# Model path
MODEL_DIR = "models"  # Relative path in your repo
# ────────────────────────────────────────────────────────────────
# ### ➡️  NEW: tempo scaling helper
MAX_BPM = 212          # 212 BPM maps to 1.0

def scale_bpm(bpm: float, max_bpm: int = MAX_BPM) -> float:
    """Linearly scale BPM to range 0-1."""
    return float(np.clip(bpm / max_bpm, 0.0, 1.0))
# ────────────────────────────────────────────────────────────────


@st.cache_resource
def load_model_components():
    """Load the trained model components including text processing models."""
    try:
        # Load LightGBM models and scaler
        models = joblib.load(f"{MODEL_DIR}/lgbm_optimized_models_latest.pkl")
        scaler = joblib.load(f"{MODEL_DIR}/feature_scaler_optimized_latest.pkl")
        
        with open(f"{MODEL_DIR}/best_params_optimized_latest.json", 'r') as f:
            best_params = json.load(f)
        
        # Load text processing models
        tfidf_vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer2_100k.pkl")
        svd_model = joblib.load(f"{MODEL_DIR}/svd2_500.pkl")
        doc2vec_model = Doc2Vec.load(f"{MODEL_DIR}/doc2vec_hiphop.bin")
        
        # Check and fix TF-IDF vectorizer
        has_vocab = hasattr(tfidf_vectorizer, 'vocabulary_')
        has_idf   = hasattr(tfidf_vectorizer, 'idf_')
        
        if has_vocab and not has_idf:
            st.warning("⚠️ TF-IDF missing idf_ attribute. Reconstructing…")
            vocab_size = len(tfidf_vectorizer.vocabulary_)
            tfidf_vectorizer.idf_ = np.ones(vocab_size, dtype=np.float64)
            st.success("✅ Reconstructed idf_ attribute with neutral weights")
        
        # Quick smoke-test
        tfidf_vectorizer.transform(["test lyrics here"])
        st.success(f"✅ Models loaded. Vocabulary size: {len(tfidf_vectorizer.vocabulary_):,}")
        
        return models, scaler, best_params, tfidf_vectorizer, svd_model, doc2vec_model
    
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return (None,) * 6


# ────────────────────────────────────────────────────────────────
def extract_audio_features(audio_file):
    """Extract basic audio features from uploaded audio file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    try:
        st.info("Loading audio file…")
        y, sr = librosa.load(tmp_path, sr=22050)
        st.success(f"✅ Audio loaded: {len(y)} samples at {sr} Hz")
        
        features = {}
        features['duration_ms'] = len(y) * 1000 / sr

        # Loudness 0-1
        st.info("Extracting loudness…")
        rms = librosa.feature.rms(y=y)[0]
        loudness_db  = 20 * np.log10(np.mean(rms) + 1e-8)      # −60 … 0 dB
        features['loudness'] = float(np.clip((loudness_db + 60) / 60, 0.0, 1.0))

        # Energy (0-100)
        st.info("Extracting energy…")
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['energy'] = float(np.clip(np.mean(spectral_centroids) / 50, 0, 100))
        
        # Danceability
        st.info("Extracting danceability…")
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

        # ### ➡️  NEW: store scaled tempo
        features['tempo'] = scale_bpm(tempo)      # 0-1 value

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        beat_strength = np.mean(onset_env[beats]) if len(beats) else np.mean(onset_env)
        normalized_beat_strength = np.clip(beat_strength / 10, 0, 1)
        tempo_factor = np.clip(1.0 - abs(tempo - 130) / 100, 0, 1)
        beat_regularity = 0.5
        if len(beats) > 1:
            beat_intervals   = np.diff(beats)
            beat_regularity  = 1.0 - (np.std(beat_intervals) /
                                  (np.mean(beat_intervals) + 1e-6))
            beat_regularity  = np.clip(beat_regularity, 0, 1)
        danceability = (normalized_beat_strength * 40 +
                        tempo_factor            * 30 +
                        beat_regularity         * 30)
        features['danceability'] = float(np.clip(danceability, 0, 100))
        
        # Acousticness etc. … (unchanged)
        features['acousticness'] = float(np.clip(100 - (np.mean(spectral_centroids) / 50), 0, 100))

        # Speechiness / Instrumentalness / Liveness / Valence (unchanged)
        # ...  (code omitted here for brevity — remains identical to your version) ...

        st.success("✅ All audio features extracted successfully!")
        return features

    except Exception as e:
        st.error(f"Error extracting audio features: {type(e).__name__}: {e}")
        return {}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
# ────────────────────────────────────────────────────────────────


def clean_lyrics_for_tfidf(text):
    if not text or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_lyrics_with_models(lyrics_text, tfidf_vectorizer, svd_model, doc2vec_model):
    svd_features     = np.zeros(500, dtype=np.float32)
    doc2vec_features = np.zeros(300, dtype=np.float32)
    if not lyrics_text.strip():
        return svd_features, doc2vec_features
    try:
        lyrics_clean = clean_lyrics_for_tfidf(lyrics_text)
        tfidf_vec    = tfidf_vectorizer.transform([lyrics_clean])
        svd_features = svd_model.transform(tfidf_vec)[0].astype(np.float32)
        doc2vec_features = doc2vec_model.infer_vector(lyrics_clean.split()).astype(np.float32)
    except Exception as e:
        st.error(f"Error processing lyrics: {e}")
    return svd_features, doc2vec_features


def predict_hit(models, scaler, audio_features, svd_features, doc2vec_features):
    order = ['danceability', 'energy', 'key_clean', 'loudness', 'mode_clean',
             'speechiness', 'acousticness', 'instrumentalness', 'liveness',
             'valence', 'tempo', 'duration_ms', 'time_signature']
    audio_vec = np.array([audio_features.get(f, 0.0) for f in order], dtype=np.float32)
    # optional instrumentalness weight
    audio_vec[7] *= 0.3
    X_full   = np.concatenate([audio_vec, svd_features, doc2vec_features]).reshape(1, -1)
    X_scaled = scaler.transform(X_full)
    probs    = [m.predict(X_scaled, num_iteration=m.best_iteration)[0] for m in models]
    avg_prob = float(np.mean(probs))
    return avg_prob, int(avg_prob >= 0.3)  # threshold fixed at 0.3


# ────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Hip-Hop Hit Predictor", layout="wide")
    st.title("🎵 Hip-Hop Hit Predictor")

    models, scaler, best_params, tfidf_v, svd_m, d2v_m = load_model_components()
    if models is None:
        return

    # File uploader
    uploaded_file = st.file_uploader("🎵 Upload audio (optional)", type=['mp3', 'wav', 'flac', 'm4a'])

    # ------------------------------------------------------------------
    # Manual / extracted feature handling
    # ------------------------------------------------------------------
    if uploaded_file:
        st.audio(uploaded_file, format='audio/mp3')
        choice = st.radio("How would you like to set audio features?",
                          ["Extract from uploaded audio", "Enter manually"])
        if choice == "Extract from uploaded audio":
            with st.spinner("Extracting…"):
                extracted = extract_audio_features(uploaded_file)
            if not extracted:
                choice = "Enter manually"
    else:
        choice = "Enter manually"

    # ------------------------------------------------------------------
    # UI for manual entry or editing
    # ------------------------------------------------------------------
    st.subheader("🎵 Audio Features")
    cols = st.columns(4)

    with cols[0]:
        danceability = st.number_input("Danceability", 0.0, 100.0, value=(
            extracted.get('danceability', 50) if uploaded_file and choice=="Extract from uploaded audio" else 50))
        energy       = st.number_input("Energy",       0.0, 100.0,
            value=extracted.get('energy', 50) if uploaded_file and choice=="Extract from uploaded audio" else 50)

    with cols[1]:
        acousticness = st.number_input("Acousticness", 0.0, 100.0,
            value=extracted.get('acousticness', 10) if uploaded_file and choice=="Extract from uploaded audio" else 10)
        speechiness  = st.number_input("Speechiness",  0.0, 100.0,
            value=extracted.get('speechiness', 15)  if uploaded_file and choice=="Extract from uploaded audio" else 15)

    with cols[2]:
        instrumentalness = st.number_input("Instrumentalness", 0.0, 100.0,
            value=extracted.get('instrumentalness', 5) if uploaded_file and choice=="Extract from uploaded audio" else 5)
        liveness = st.number_input("Liveness", 0.0, 100.0,
            value=extracted.get('liveness', 15) if uploaded_file and choice=="Extract from uploaded audio" else 15)

    with cols[3]:
        valence  = st.number_input("Valence", 0.0, 100.0,
            value=extracted.get('valence', 50) if uploaded_file and choice=="Extract from uploaded audio" else 50)
        loudness = st.number_input("Loudness (0-1)", 0.0, 1.0,
            value=extracted.get('loudness', 0.2) if uploaded_file and choice=="Extract from uploaded audio" else 0.2, step=0.01)

    # Technical features ------------------------------------------------
    st.markdown("**Technical Features**")
    c1, c2 = st.columns(2)

    with c1:
        duration_sec = st.number_input("Duration (sec)", 30, 600,
            value=int(extracted.get('duration_ms',180000)/1000) if uploaded_file and choice=="Extract from uploaded audio" else 180)

    with c2:
        tempo_raw = st.number_input("Tempo (BPM)", 10, 300,
            value=int(librosa.beat.beat_track(y=None)[0]) if uploaded_file and choice=="Extract from uploaded audio" else 120)

    tempo = scale_bpm(tempo_raw)   # ### ➡️  scaled value used later
    duration_ms = duration_sec * 1000

    # Key / mode / time sig
    key_options = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    key_name = st.selectbox("Key", key_options)
    mode     = st.selectbox("Mode", ["Minor","Major"])
    time_sig_map = {"3/4":34,"4/4":44,"5/4":54,"6/8":68,"7/4":74}
    time_sig   = time_sig_map[st.selectbox("Time Signature", list(time_sig_map))]

    # Build feature dict
    feats = {
        'danceability':danceability,'energy':energy,'key_clean':key_options.index(key_name),
        'loudness':loudness,'mode_clean':0 if mode=="Minor" else 1,'speechiness':speechiness,
        'acousticness':acousticness,'instrumentalness':instrumentalness,'liveness':liveness,
        'valence':valence,'tempo':tempo,'duration_ms':duration_ms,'time_signature':time_sig
    }

    # Lyrics ------------------------------------------------------------
    lyrics = st.text_area("📝 Lyrics (optional)", height=200)
    svd_f, d2v_f = process_lyrics_with_models(lyrics, tfidf_v, svd_m, d2v_m)

    # Predict -----------------------------------------------------------
    if st.button("🔮 Predict Hit Potential"):
        with st.spinner("Analyzing…"):
            prob, pred = predict_hit(models, scaler, feats, svd_f, d2v_f)
        st.metric("Hit probability", f"{prob*100:.1f}%")
        st.metric("Prediction", "HIT" if pred else "NON-HIT")


if __name__ == "__main__":
    main()
