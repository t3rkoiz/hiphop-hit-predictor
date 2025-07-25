# streamlit_app.py
import streamlit as st
import librosa
import numpy as np
import pandas as pd
import tempfile
import os
import joblib
import json
from textstat import flesch_reading_ease, flesch_kincaid_grade
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Update this path to where your model files are stored in your repo
MODEL_DIR = "models"  # Relative path in your repo

@st.cache_resource
def load_model_components():
    """Load the trained model components."""
    try:
        models = joblib.load(f"{MODEL_DIR}/lgbm_optimized_models_latest.pkl")
        scaler = joblib.load(f"{MODEL_DIR}/feature_scaler_optimized_latest.pkl")
        with open(f"{MODEL_DIR}/best_params_optimized_latest.json", 'r') as f:
            best_params = json.load(f)
        return models, scaler, best_params
    except FileNotFoundError:
        st.error("Model files not found. Please check your model directory.")
        return None, None, None

def extract_comprehensive_audio_features(audio_file):
    """Extract comprehensive audio features to match training data."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    try:
        # Load audio file
        y, sr = librosa.load(tmp_path, sr=22050)  # Standardize sample rate
        
        features = {}
        
        # Basic audio properties
        features['duration_ms'] = len(y) * 1000 / sr
        
        # ======= SPECTRAL FEATURES =======
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        features['spectral_centroid_median'] = np.median(spectral_centroids)
        features['spectral_centroid_min'] = np.min(spectral_centroids)
        features['spectral_centroid_max'] = np.max(spectral_centroids)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        features['spectral_rolloff_median'] = np.median(spectral_rolloff)
        features['spectral_rolloff_min'] = np.min(spectral_rolloff)
        features['spectral_rolloff_max'] = np.max(spectral_rolloff)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        features['spectral_bandwidth_median'] = np.median(spectral_bandwidth)
        features['spectral_bandwidth_min'] = np.min(spectral_bandwidth)
        features['spectral_bandwidth_max'] = np.max(spectral_bandwidth)
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast_{i}_mean'] = np.mean(spectral_contrast[i])
            features[f'spectral_contrast_{i}_std'] = np.std(spectral_contrast[i])
        
        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['spectral_flatness_mean'] = np.mean(spectral_flatness)
        features['spectral_flatness_std'] = np.std(spectral_flatness)
        
        # ======= ZERO CROSSING RATE =======
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        features['zcr_median'] = np.median(zcr)
        features['zcr_min'] = np.min(zcr)
        features['zcr_max'] = np.max(zcr)
        
        # ======= MFCC FEATURES =======
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # More MFCCs
        for i in range(20):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            features[f'mfcc_{i}_median'] = np.median(mfccs[i])
            features[f'mfcc_{i}_min'] = np.min(mfccs[i])
            features[f'mfcc_{i}_max'] = np.max(mfccs[i])
        
        # ======= CHROMA FEATURES =======
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(12):
            features[f'chroma_{i}_mean'] = np.mean(chroma[i])
            features[f'chroma_{i}_std'] = np.std(chroma[i])
            features[f'chroma_{i}_median'] = np.median(chroma[i])
            features[f'chroma_{i}_min'] = np.min(chroma[i])
            features[f'chroma_{i}_max'] = np.max(chroma[i])
        
        # ======= TONNETZ =======
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        for i in range(6):
            features[f'tonnetz_{i}_mean'] = np.mean(tonnetz[i])
            features[f'tonnetz_{i}_std'] = np.std(tonnetz[i])
        
        # ======= TEMPO AND RHYTHM =======
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        features['beat_count'] = len(beats)
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        features['onset_count'] = len(onset_frames)
        features['onset_rate'] = len(onset_frames) / (len(y) / sr)
        
        # ======= ENERGY FEATURES =======
        # RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_median'] = np.median(rms)
        features['rms_min'] = np.min(rms)
        features['rms_max'] = np.max(rms)
        
        # Loudness
        features['loudness'] = 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-8)
        
        # ======= HARMONIC/PERCUSSIVE SEPARATION =======
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Harmonic features
        features['harmonic_mean'] = np.mean(y_harmonic)
        features['harmonic_std'] = np.std(y_harmonic)
        features['harmonic_energy'] = np.sum(y_harmonic**2)
        
        # Percussive features
        features['percussive_mean'] = np.mean(y_percussive)
        features['percussive_std'] = np.std(y_percussive)
        features['percussive_energy'] = np.sum(y_percussive**2)
        
        # Ratio
        features['harmonic_percussive_ratio'] = features['harmonic_energy'] / (features['percussive_energy'] + 1e-8)
        
        # ======= PITCH FEATURES =======
        # Fundamental frequency
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_clean = f0[~np.isnan(f0)]
        if len(f0_clean) > 0:
            features['f0_mean'] = np.mean(f0_clean)
            features['f0_std'] = np.std(f0_clean)
            features['f0_median'] = np.median(f0_clean)
            features['f0_min'] = np.min(f0_clean)
            features['f0_max'] = np.max(f0_clean)
        else:
            features['f0_mean'] = 0
            features['f0_std'] = 0
            features['f0_median'] = 0
            features['f0_min'] = 0
            features['f0_max'] = 0
        
        features['voiced_ratio'] = np.sum(voiced_flag) / len(voiced_flag)
        
        # ======= STATISTICAL FEATURES =======
        # Audio statistics
        features['audio_mean'] = np.mean(y)
        features['audio_std'] = np.std(y)
        features['audio_skew'] = float(pd.Series(y).skew())
        features['audio_kurtosis'] = float(pd.Series(y).kurtosis())
        
        # Dynamic range
        features['dynamic_range'] = np.max(np.abs(y)) - np.min(np.abs(y))
        
        # ======= APPROXIMATED SPOTIFY FEATURES =======
        # These are rough approximations based on audio analysis
        features['danceability'] = np.clip((features['beat_count'] / (features['duration_ms'] / 1000)) / 10, 0, 1)
        features['energy'] = np.clip(features['rms_mean'] * 10, 0, 1)
        features['acousticness'] = np.clip(1 - (features['spectral_centroid_mean'] / 5000), 0, 1)
        features['instrumentalness'] = np.clip(1 - features['voiced_ratio'], 0, 1)
        features['liveness'] = np.clip(features['spectral_flatness_mean'] * 5, 0, 1)
        features['speechiness'] = np.clip(features['voiced_ratio'] * features['zcr_mean'] * 20, 0, 1)
        features['valence'] = np.clip((features['chroma_0_mean'] + features['chroma_4_mean'] + features['chroma_7_mean']) / 3, 0, 1)
        
        return features
        
    except Exception as e:
        st.error(f"Error extracting audio features: {str(e)}")
        return {}
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def extract_lyrics_features(lyrics_text):
    """Extract comprehensive lyrics features."""
    if not lyrics_text or lyrics_text.strip() == "":
        return get_empty_lyrics_features()
    
    features = {}
    
    # Basic text statistics
    words = lyrics_text.split()
    sentences = re.split(r'[.!?]+', lyrics_text)
    lines = lyrics_text.split('\n')
    
    features['word_count'] = len(words)
    features['char_count'] = len(lyrics_text)
    features['line_count'] = len(lines)
    features['sentence_count'] = len([s for s in sentences if s.strip()])
    
    # Word length statistics
    word_lengths = [len(word) for word in words]
    if word_lengths:
        features['avg_word_length'] = np.mean(word_lengths)
        features['word_length_std'] = np.std(word_lengths)
        features['max_word_length'] = max(word_lengths)
        features['min_word_length'] = min(word_lengths)
    else:
        features['avg_word_length'] = 0
        features['word_length_std'] = 0
        features['max_word_length'] = 0
        features['min_word_length'] = 0
    
    # Readability scores
    try:
        features['flesch_reading_ease'] = flesch_reading_ease(lyrics_text)
        features['flesch_kincaid_grade'] = flesch_kincaid_grade(lyrics_text)
    except:
        features['flesch_reading_ease'] = 0
        features['flesch_kincaid_grade'] = 0
    
    # Vocabulary richness
    unique_words = set(word.lower() for word in words)
    features['unique_word_count'] = len(unique_words)
    features['vocabulary_richness'] = len(unique_words) / len(words) if words else 0
    
    # Repetition analysis
    word_freq = Counter(word.lower() for word in words)
    most_common = word_freq.most_common(10)
    features['most_frequent_word_freq'] = most_common[0][1] if most_common else 0
    features['top_words_freq_sum'] = sum(freq for word, freq in most_common)
    features['repetition_ratio'] = features['top_words_freq_sum'] / len(words) if words else 0
    
    # Emotional/sentiment features (basic)
    positive_words = ['love', 'happy', 'joy', 'good', 'great', 'amazing', 'wonderful', 'beautiful', 'best', 'awesome']
    negative_words = ['sad', 'hate', 'bad', 'terrible', 'awful', 'horrible', 'pain', 'hurt', 'worst', 'angry']
    profanity_words = ['damn', 'shit', 'fuck', 'bitch', 'ass', 'hell']  # Add more as needed
    
    text_lower = lyrics_text.lower()
    features['positive_word_count'] = sum(text_lower.count(word) for word in positive_words)
    features['negative_word_count'] = sum(text_lower.count(word) for word in negative_words)
    features['profanity_count'] = sum(text_lower.count(word) for word in profanity_words)
    
    features['positive_sentiment_ratio'] = features['positive_word_count'] / len(words) if words else 0
    features['negative_sentiment_ratio'] = features['negative_word_count'] / len(words) if words else 0
    features['profanity_ratio'] = features['profanity_count'] / len(words) if words else 0
    
    # Structural features
    features['exclamation_count'] = lyrics_text.count('!')
    features['question_count'] = lyrics_text.count('?')
    features['comma_count'] = lyrics_text.count(',')
    features['period_count'] = lyrics_text.count('.')
    
    # Rhyme approximation (very basic)
    line_endings = [line.strip().split()[-1].lower() if line.strip().split() else "" for line in lines]
    line_endings = [ending for ending in line_endings if ending]
    ending_freq = Counter(ending[-2:] for ending in line_endings if len(ending) >= 2)
    features['rhyme_density'] = sum(1 for freq in ending_freq.values() if freq > 1) / len(line_endings) if line_endings else 0
    
    return features

def get_empty_lyrics_features():
    """Return zero features for empty lyrics."""
    return {
        'word_count': 0, 'char_count': 0, 'line_count': 0, 'sentence_count': 0,
        'avg_word_length': 0, 'word_length_std': 0, 'max_word_length': 0, 'min_word_length': 0,
        'flesch_reading_ease': 0, 'flesch_kincaid_grade': 0,
        'unique_word_count': 0, 'vocabulary_richness': 0,
        'most_frequent_word_freq': 0, 'top_words_freq_sum': 0, 'repetition_ratio': 0,
        'positive_word_count': 0, 'negative_word_count': 0, 'profanity_count': 0,
        'positive_sentiment_ratio': 0, 'negative_sentiment_ratio': 0, 'profanity_ratio': 0,
        'exclamation_count': 0, 'question_count': 0, 'comma_count': 0, 'period_count': 0,
        'rhyme_density': 0
    }

def predict_hit(models, scaler, features):
    """Make prediction using the ensemble of models."""
    try:
        # Convert features dict to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Get the expected feature count
        expected_features = scaler.n_features_in_
        current_features = len(feature_df.columns)
        
        st.info(f"Current features: {current_features}, Expected: {expected_features}")
        
        # If we have fewer features than expected, we need to add missing ones
        if current_features < expected_features:
            # This is a fallback - ideally you should match your training features exactly
            for i in range(current_features, expected_features):
                feature_df[f'missing_feature_{i}'] = 0
        
        # Scale features
        scaled_features = scaler.transform(feature_df)
        
        # Get predictions from all models
        predictions = []
        for model in models:
            pred = model.predict(scaled_features, num_iteration=model.best_iteration)
            predictions.append(pred[0])
        
        # Average predictions
        avg_probability = np.mean(predictions)
        return avg_probability
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 0.5

def main():
    st.set_page_config(page_title="Hip-Hop Hit Predictor", layout="wide")
    st.title("🎵 Hip-Hop Hit Predictor")
    st.markdown("Upload an audio file and provide key information to predict if your song could be a hit!")
    
    # Load model components
    models, scaler, best_params = load_model_components()
    
    if models is None:
        st.error("Failed to load model. Please check your model files.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'flac', 'm4a'])
    
    # Lyrics input
    st.subheader("📝 Song Lyrics (Optional)")
    lyrics_text = st.text_area(
        "Enter the song lyrics to improve prediction accuracy:",
        height=150,
        placeholder="Paste your lyrics here..."
    )
    
    # Manual inputs for features that can't be extracted
    st.subheader("🎼 Additional Song Information")
    
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
        
        # Extract features automatically
        with st.spinner("Extracting comprehensive audio features..."):
            audio_features = extract_comprehensive_audio_features(uploaded_file)
        
        if not audio_features:
            st.error("Failed to extract audio features. Please try a different file.")
            return
        
        # Extract lyrics features
        lyrics_features = extract_lyrics_features(lyrics_text)
        
        # Add manual features
        audio_features['key'] = song_key
        audio_features['mode'] = mode_value
        audio_features['time_signature'] = time_signature
        
        # Combine all features
        all_features = {**audio_features, **lyrics_features}
        
        # Display some key extracted features
        st.subheader("🎵 Key Extracted Features")
        cols = st.columns(4)
        cols[0].metric("Tempo", f"{audio_features.get('tempo', 0):.0f} BPM")
        cols[1].metric("Danceability", f"{audio_features.get('danceability', 0):.3f}")
        cols[2].metric("Energy", f"{audio_features.get('energy', 0):.3f}")
        cols[3].metric("Valence", f"{audio_features.get('valence', 0):.3f}")
        
        if lyrics_text:
            st.subheader("📊 Lyrics Analysis")
            cols = st.columns(4)
            cols[0].metric("Word Count", lyrics_features.get('word_count', 0))
            cols[1].metric("Vocabulary Richness", f"{lyrics_features.get('vocabulary_richness', 0):.3f}")
            cols[2].metric("Sentiment Score", f"{lyrics_features.get('positive_sentiment_ratio', 0) - lyrics_features.get('negative_sentiment_ratio', 0):.3f}")
            cols[3].metric("Rhyme Density", f"{lyrics_features.get('rhyme_density', 0):.3f}")
        
        # Prediction button
        if st.button("🔮 Predict Hit Potential", type="primary", use_container_width=True):
            with st.spinner("Analyzing your song..."):
                try:
                    # Make prediction
                    probability = predict_hit(models, scaler, all_features)
                    
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
                        
                    # Feature importance (if available)
                    with st.expander("📊 View All Extracted Features"):
                        feature_df = pd.DataFrame(list(all_features.items()), columns=['Feature', 'Value'])
                        st.dataframe(feature_df, height=400)
                        
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.info("This might be due to feature mismatch. Please ensure your model files are compatible.")

if __name__ == "__main__":
    main()