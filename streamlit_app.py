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

# Update this path to where your model files are stored in your repo
MODEL_DIR = "models"  # Relative path in your repo

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
        tfidf_vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer_100k.pkl")
        svd_model = joblib.load(f"{MODEL_DIR}/svd_500.pkl")
        doc2vec_model = Doc2Vec.load(f"{MODEL_DIR}/doc2vec_hiphop.bin")
        
        return models, scaler, best_params, tfidf_vectorizer, svd_model, doc2vec_model
    except FileNotFoundError as e:
        st.error(f"Model files not found: {str(e)}")
        st.error("Please ensure you have saved all required model files:")
        st.code("""
Required files:
- lgbm_optimized_models_latest.pkl
- feature_scaler_optimized_latest.pkl  
- best_params_optimized_latest.json
- tfidf_vectorizer_100k.pkl
- svd_500.pkl
- doc2vec_hiphop.bin
        """)
        return None, None, None, None, None, None

def clean_lyrics_for_tfidf(lyrics_text):
    """Clean lyrics text similar to your training preprocessing."""
    if not lyrics_text or lyrics_text.strip() == "":
        return ""
    
    # Basic cleaning (adjust based on your original preprocessing)
    lyrics_clean = lyrics_text.lower()
    lyrics_clean = re.sub(r'[^\w\s]', ' ', lyrics_clean)  # Remove punctuation
    lyrics_clean = re.sub(r'\s+', ' ', lyrics_clean)      # Multiple spaces to single
    lyrics_clean = lyrics_clean.strip()
    
    return lyrics_clean

def extract_audio_features(audio_file, audio_cols=None):
    """Extract basic audio features matching your training data."""
    # Default audio columns if not provided
    if audio_cols is None:
        audio_cols = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
            'duration_ms', 'time_signature'
        ]
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    try:
        # Load audio file
        y, sr = librosa.load(tmp_path, sr=22050)
        
        features = {}
        
        # Extract basic features (approximations of Spotify features)
        # Duration
        features['duration_ms'] = len(y) * 1000 / sr
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)
        
        # Loudness (approximation)
        rms = librosa.feature.rms(y=y)[0]
        features['loudness'] = float(20 * np.log10(np.mean(rms) + 1e-8))
        
        # Energy (approximation)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['energy'] = float(np.clip(np.mean(spectral_centroids) / 5000, 0, 1))
        
        # Danceability (approximation based on beat strength)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        features['danceability'] = float(np.clip(len(onset_frames) / (len(y) / sr) / 10, 0, 1))
        
        # Acousticness (inverse of spectral centroid)
        features['acousticness'] = float(np.clip(1 - (np.mean(spectral_centroids) / 5000), 0, 1))
        
        # Speechiness (approximation using zero crossing rate)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['speechiness'] = float(np.clip(np.mean(zcr) * 10, 0, 1))
        
        # Instrumentalness (approximation - inverse of vocal activity)
        # Simple heuristic: low variance in spectral features suggests less vocals
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['instrumentalness'] = float(np.clip(1 - np.var(mfccs) / 100, 0, 1))
        
        # Liveness (approximation using spectral flatness)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['liveness'] = float(np.clip(np.mean(spectral_flatness) * 5, 0, 1))
        
        # Valence (approximation using chroma features)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        # Major chords (C, E, G) tend to sound happier
        major_chord_strength = np.mean([chroma[0], chroma[4], chroma[7]])
        features['valence'] = float(np.clip(major_chord_strength, 0, 1))
        
        return features
        
    except Exception as e:
        st.error(f"Error extracting audio features: {str(e)}")
        return {}
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def process_lyrics_with_models(lyrics_text, tfidf_vectorizer, svd_model, doc2vec_model):
    """Process lyrics using the trained TF-IDF+SVD and Doc2Vec models."""
    
    # Initialize arrays
    svd_features = np.zeros(500, dtype=np.float32)
    doc2vec_features = np.zeros(300, dtype=np.float32)
    
    if not lyrics_text or lyrics_text.strip() == "":
        return svd_features, doc2vec_features
    
    try:
        # Clean lyrics
        lyrics_clean = clean_lyrics_for_tfidf(lyrics_text)
        
        if not lyrics_clean:
            return svd_features, doc2vec_features
        
        # TF-IDF + SVD processing
        tfidf_vector = tfidf_vectorizer.transform([lyrics_clean])
        svd_vector = svd_model.transform(tfidf_vector)
        svd_features = svd_vector[0].astype(np.float32)
        
        # Doc2Vec processing
        words = lyrics_clean.split()
        if words:  # Only if we have words
            doc2vec_vector = doc2vec_model.infer_vector(words)
            doc2vec_features = doc2vec_vector.astype(np.float32)
        
        return svd_features, doc2vec_features
        
    except Exception as e:
        st.error(f"Error processing lyrics: {str(e)}")
        return svd_features, doc2vec_features

def predict_hit(models, scaler, audio_features, svd_features, doc2vec_features, 
                key, mode, time_signature):
    """Make prediction using the ensemble of models."""
    try:
        # Prepare audio features in the correct order
        audio_feature_order = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
            'duration_ms', 'time_signature'
        ]
        
        # Add manual features to audio_features
        audio_features['key'] = key
        audio_features['mode'] = mode  
        audio_features['time_signature'] = time_signature
        
        # Create audio feature vector
        audio_vector = np.array([audio_features.get(feat, 0.0) for feat in audio_feature_order], 
                               dtype=np.float32)
        
        # Combine all features: audio + SVD + Doc2Vec
        X_full = np.concatenate([audio_vector, svd_features, doc2vec_features])
        
        # Reshape for prediction
        X_full = X_full.reshape(1, -1)
        
        st.info(f"Feature vector shape: {X_full.shape}")
        st.info(f"Expected features: {scaler.n_features_in_}")
        
        # Scale features
        X_scaled = scaler.transform(X_full)
        
        # Get predictions from all models
        predictions = []
        for model in models:
            pred = model.predict(X_scaled, num_iteration=model.best_iteration)
            predictions.append(pred[0])
        
        # Average predictions
        avg_probability = np.mean(predictions)
        return float(avg_probability)
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 0.5

def main():
    st.set_page_config(page_title="Hip-Hop Hit Predictor", layout="wide")
    st.title("🎵 Hip-Hop Hit Predictor")
    st.markdown("Upload an audio file and provide lyrics to predict if your song could be a hit!")
    
    # Load model components
    models, scaler, best_params, tfidf_vectorizer, svd_model, doc2vec_model = load_model_components()
    
    if models is None:
        st.error("Failed to load model components. Please check your model files.")
        return
    
    st.success("✅ All model components loaded successfully!")
    
    # File uploader
    st.subheader("🎵 Upload Audio File")
    uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'flac', 'm4a'])
    
    # Lyrics input
    st.subheader("📝 Song Lyrics")
    st.info("Lyrics are crucial for accurate prediction! Your model uses TF-IDF and Doc2Vec text analysis.")
    lyrics_text = st.text_area(
        "Enter the complete song lyrics:",
        height=200,
        placeholder="Paste your lyrics here. The more complete, the better the prediction...",
        help="Your model was trained on lyrics processed with TF-IDF and Doc2Vec, so lyrics significantly improve prediction accuracy."
    )
    
    # Manual inputs for features that can't be extracted
    st.subheader("🎼 Song Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        key_options = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_name = st.selectbox("Song Key", key_options, index=0)
        key_map = {name: i for i, name in enumerate(key_options)}
        song_key = key_map[key_name]
    
    with col2:
        mode = st.selectbox("Mode", ["Minor", "Major"], index=1)
        mode_value = 0 if mode == "Minor" else 1
    
    with col3:
        time_signature = st.selectbox("Time Signature", [3, 4, 5, 6, 7], index=1)
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format='audio/mp3')
        
        # Extract audio features
        with st.spinner("Extracting audio features..."):
            audio_features = extract_audio_features(uploaded_file)
        
        if not audio_features:
            st.error("Failed to extract audio features. Please try a different file.")
            return
        
        # Process lyrics
        with st.spinner("Processing lyrics with TF-IDF and Doc2Vec models..."):
            svd_features, doc2vec_features = process_lyrics_with_models(
                lyrics_text, tfidf_vectorizer, svd_model, doc2vec_model
            )
        
        # Display extracted features
        st.subheader("🎵 Extracted Audio Features")
        cols = st.columns(4)
        cols[0].metric("Tempo", f"{audio_features.get('tempo', 0):.0f} BPM")
        cols[1].metric("Danceability", f"{audio_features.get('danceability', 0):.3f}")
        cols[2].metric("Energy", f"{audio_features.get('energy', 0):.3f}")
        cols[3].metric("Valence", f"{audio_features.get('valence', 0):.3f}")
        
        # Display feature processing info
        st.subheader("📊 Feature Processing Status")
        cols = st.columns(3)
        cols[0].metric("Audio Features", "13", help="Basic Spotify-like features")
        cols[1].metric("TF-IDF + SVD Features", "500", help="Text features from lyrics")
        cols[2].metric("Doc2Vec Features", "300", help="Document embeddings from lyrics")
        
        # Show if lyrics were processed
        if lyrics_text.strip():
            st.success(f"✅ Lyrics processed: {len(lyrics_text.split())} words")
        else:
            st.warning("⚠️ No lyrics provided - using zero vectors for text features")
        
        # Prediction button
        if st.button("🔮 Predict Hit Potential", type="primary", use_container_width=True):
            with st.spinner("Analyzing your song with full feature set..."):
                try:
                    # Make prediction
                    probability = predict_hit(
                        models, scaler, audio_features, svd_features, doc2vec_features,
                        song_key, mode_value, time_signature
                    )
                    
                    # Display results
                    st.markdown("---")
                    st.header("🎯 Prediction Results")
                    
                    # Progress bar
                    st.progress(float(probability))
                    
                    # Main result display
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        if probability > 0.7:
                            st.success(f"🔥 HIT POTENTIAL: {probability*100:.1f}%")
                            st.balloons()
                        elif probability > 0.5:
                            st.warning(f"⭐ PROMISING: {probability*100:.1f}%")
                        else:
                            st.info(f"🌱 DEVELOPING: {probability*100:.1f}%")
                    
                    # Interpretation
                    st.markdown("**What this means:**")
                    if probability > 0.8:
                        st.success("🚀 **Strong Hit Potential!** Your song has characteristics similar to chart-topping hits.")
                    elif probability > 0.6:
                        st.info("⭐ **Good Potential!** Your song shows promising characteristics.")
                    elif probability > 0.4:
                        st.warning("🛠️ **Needs Refinement** This song has some potential but might benefit from adjustments.")
                    else:
                        st.error("🔧 **Early Stage** This song is in early development.")
                    
                    # Recommendations based on missing features
                    if not lyrics_text.strip():
                        st.info("💡 **Tip**: Adding lyrics could significantly improve prediction accuracy!")
                        
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.info("Please check that all model files are compatible and properly saved.")
    
    else:
        st.info("👆 Please upload an audio file to get started!")
    
    # Footer with model info
    st.markdown("---")
    st.markdown("**Model Architecture:**")
    st.markdown("- **Audio Features**: 13 Spotify-like features extracted from audio")
    st.markdown("- **Text Features**: 500 TF-IDF + SVD dimensions + 300 Doc2Vec dimensions")
    st.markdown("- **Total Features**: 813 features processed through LightGBM ensemble")

if __name__ == "__main__":
    main()