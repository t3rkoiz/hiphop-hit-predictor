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
        tfidf_vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer2_100k.pkl")
        svd_model = joblib.load(f"{MODEL_DIR}/svd2_500.pkl")
        doc2vec_model = Doc2Vec.load(f"{MODEL_DIR}/doc2vec_hiphop.bin")
        
        # Check and fix TF-IDF vectorizer
        has_vocab = hasattr(tfidf_vectorizer, 'vocabulary_')
        has_idf = hasattr(tfidf_vectorizer, 'idf_')
        
        if has_vocab and not has_idf:
            st.warning("⚠️ TF-IDF missing idf_ attribute. Reconstructing...")
            try:
                # Get vocabulary and create a dummy idf_ array
                vocab_size = len(tfidf_vectorizer.vocabulary_)
                # Set all idf values to 1.0 (neutral weighting)
                tfidf_vectorizer.idf_ = np.ones(vocab_size, dtype=np.float64)
                st.success("✅ Reconstructed idf_ attribute with neutral weights")
                has_idf = True
            except Exception as reconstruct_error:
                st.error(f"❌ Failed to reconstruct idf_: {reconstruct_error}")
                return None, None, None, None, None, None
        
        # Test TF-IDF vectorizer
        try:
            test_result = tfidf_vectorizer.transform(["test lyrics here"])
            st.success(f"✅ All models loaded successfully! Vocabulary: {len(tfidf_vectorizer.vocabulary_):,}")
        except Exception as e:
            st.error(f"❌ TF-IDF test failed: {e}")
            return None, None, None, None, None, None
        
        return models, scaler, best_params, tfidf_vectorizer, svd_model, doc2vec_model
        
    except FileNotFoundError as e:
        st.error(f"Model files not found: {str(e)}")
        st.error("Please ensure you have saved all required model files in the 'models' directory")
        return None, None, None, None, None, None

def extract_audio_features(audio_file):
    """Extract basic audio features from uploaded audio file."""
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    try:
        # Load audio file
        st.info("Loading audio file...")
        y, sr = librosa.load(tmp_path, sr=22050)
        st.success(f"✅ Audio loaded: {len(y)} samples at {sr} Hz")
        
        features = {}
        
        # Duration
        features['duration_ms'] = len(y) * 1000 / sr
        
        # Loudness in dB mapped to 0-1
        st.info("Extracting loudness...")
        rms = librosa.feature.rms(y=y)[0]
        loudness_db = 20 * np.log10(np.mean(rms) + 1e-8)      # –60 … 0 dB
        loudness_norm = np.clip((loudness_db + 60) / 60, 0.0, 1.0)
        features['loudness'] = float(loudness_norm)

        
        # Energy (approximation) - scale to 0-100
        st.info("Extracting energy...")
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['energy'] = float(np.clip(np.mean(spectral_centroids) / 50, 0, 100))  # Scale to 0-100
        
        # Danceability (approximation) - FIXED: Better scaling
        st.info("Extracting danceability...")
        # Combine tempo regularity and beat strength
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Calculate beat strength (how pronounced the beats are)
        if len(beats) > 0:
            beat_strength = np.mean(onset_env[beats])
        else:
            beat_strength = np.mean(onset_env)
        
        # Normalize beat strength (typically ranges from 0 to ~10)
        normalized_beat_strength = np.clip(beat_strength / 10, 0, 1)
        
        # Tempo factor (120-140 BPM is ideal for dancing)
        tempo_factor = 1.0 - abs(tempo - 130) / 100  # Peaks at 130 BPM
        tempo_factor = np.clip(tempo_factor, 0, 1)
        
        # Beat regularity (how consistent the beat intervals are)
        if len(beats) > 1:
            beat_intervals = np.diff(beats)
            beat_regularity = 1.0 - (np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-6))
            beat_regularity = np.clip(beat_regularity, 0, 1)
        else:
            beat_regularity = 0.5
        
        # Combine factors with weights
        danceability = (normalized_beat_strength * 40 + tempo_factor * 30 + beat_regularity * 30)
        features['danceability'] = float(np.clip(danceability, 0, 100))
        
        # Acousticness (inverse of spectral centroid) - scale to 0-100
        features['acousticness'] = float(np.clip(100 - (np.mean(spectral_centroids) / 50), 0, 100))  # Scale to 0-100
        
        # Speechiness (approximation using zero crossing rate and spectral features) - scale to 0-100
        st.info("Extracting speechiness...")
        
        # Zero crossing rate - but normalized differently for speech detection
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)
        zcr_var = np.var(zcr)
        
        # Spectral centroid - speech has specific frequency characteristics
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        centroid_mean = np.mean(spectral_centroid)
        
        # RMS energy variation - speech has more dynamic range
        rms = librosa.feature.rms(y=y)[0]
        rms_var = np.var(rms)
        
        # Onset detection - speech has fewer strong onsets than music
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_rate = len(onset_frames) / (len(y) / sr)  # Onsets per second
        
        # Speech typically has:
        # - Moderate ZCR (0.02-0.05)
        # - Centroid in speech range (1000-4000 Hz)
        # - High RMS variance (dynamic speech)
        # - Lower onset rate than percussive music
        
        # Normalize features
        zcr_speech_score = 1.0 - abs(zcr_mean - 0.035) / 0.035  # Peak at 0.035
        zcr_speech_score = np.clip(zcr_speech_score, 0, 1)
        
        centroid_speech_score = 1.0 - abs(centroid_mean - 2500) / 2500  # Peak at 2500 Hz
        centroid_speech_score = np.clip(centroid_speech_score, 0, 1)
        
        rms_var_norm = np.clip(rms_var * 100, 0, 1)  # Higher variance = more speech-like
        
        onset_speech_score = np.clip(1.0 - onset_rate / 10, 0, 1)  # Fewer onsets = more speech-like
        
        # Combine features
        speechiness = (
            zcr_speech_score * 25 +
            centroid_speech_score * 25 +
            rms_var_norm * 25 +
            onset_speech_score * 25
        )
        
        features['speechiness'] = float(np.clip(speechiness, 0, 100))
        
        # Instrumentalness (approximation) - scale to 0-100
        st.info("Extracting instrumentalness...")
        
        # For instrumental detection, we need to identify absence of vocals
        # Use harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # 1. Harmonic ratio - vocals increase harmonic content
        harmonic_ratio = np.sum(np.abs(y_harmonic)) / (np.sum(np.abs(y)) + 1e-6)
        
        # 2. Spectral flatness on harmonic component - vocals make it less flat
        spectral_flatness = librosa.feature.spectral_flatness(y=y_harmonic)[0]
        flatness_mean = np.mean(spectral_flatness)
        
        # 3. MFCC variance on harmonic component - vocals increase variance
        mfccs_harmonic = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)
        mfcc_var = np.mean(np.var(mfccs_harmonic[1:5], axis=1))  # Variance of lower MFCCs
        
        # 4. Pitch confidence - continuous pitch suggests vocals
        pitches, magnitudes = librosa.piptrack(y=y_harmonic, sr=sr)
        # Count frames with confident pitch detection
        pitch_confidence = np.sum(magnitudes > np.max(magnitudes) * 0.3) / magnitudes.shape[1]
        
        # 5. Formant-like peaks in spectrum - characteristic of vocals
        D = np.abs(librosa.stft(y_harmonic))
        spectral_peaks = []
        for frame in D.T:
            if np.max(frame) > 0:
                peaks = librosa.util.peak_pick(frame, pre_max=3, post_max=3, 
                                                 pre_avg=3, post_avg=5, delta=0.5, wait=10)
                spectral_peaks.append(len(peaks))
        avg_peaks = np.mean(spectral_peaks) if spectral_peaks else 0
        
        # Normalize features
        harmonic_norm = np.clip(harmonic_ratio, 0.3, 0.7)  # Typical range
        flatness_norm = np.clip(flatness_mean * 10, 0, 1)  # Scale up as it's usually small
        mfcc_var_norm = np.clip(mfcc_var / 50, 0, 1)  # Normalize to typical range
        pitch_conf_norm = np.clip(pitch_confidence, 0, 1)
        peaks_norm = np.clip(avg_peaks / 20, 0, 1)  # Normalize peak count
        
        # Debug info
        st.text(f"Debug - Harmonic: {harmonic_ratio:.3f}, Flatness: {flatness_mean:.4f}, "
                f"MFCC var: {mfcc_var:.2f}, Pitch conf: {pitch_confidence:.3f}, Peaks: {avg_peaks:.1f}")
        
        # Calculate instrumentalness
        # High instrumentalness when:
        # - High spectral flatness (no distinct formants)
        # - Low MFCC variance (no vocal variation)
        # - Low pitch confidence (no clear melody)
        # - Fewer spectral peaks (no formant structure)
        # - Moderate harmonic ratio
        
        vocal_evidence = (
            (1 - flatness_norm) * 0.25 +  # Low flatness = vocals
            mfcc_var_norm * 0.25 +         # High variance = vocals
            pitch_conf_norm * 0.25 +       # High pitch confidence = vocals
            peaks_norm * 0.25              # Many peaks = vocals
        )
        
        # Invert to get instrumentalness (high when no vocals detected)
        instrumentalness = (1 - vocal_evidence) * 100
        
        # Adjust for typical hip-hop range (rarely fully instrumental)
        if instrumentalness > 50:
            instrumentalness = instrumentalness * 0.8  # Scale down high values
        
        features['instrumentalness'] = float(np.clip(instrumentalness, 0, 100))
        
        # Liveness (approximation using spectral flatness) - scale to 0-100
        st.info("Extracting liveness...")
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['liveness'] = float(np.clip(np.mean(spectral_flatness) * 500, 0, 100))  # Scale to 0-100
        
        # Valence (approximation using chroma features) - scale to 0-100
        st.info("Extracting valence...")
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        major_chord_strength = np.mean([chroma[0], chroma[4], chroma[7]])
        features['valence'] = float(np.clip(major_chord_strength * 100, 0, 100))  # Scale to 0-100
        
        st.success("✅ All audio features extracted successfully!")
        return features
        
    except Exception as e:
        st.error(f"Error extracting audio features: {type(e).__name__}: {str(e)}")
        import traceback
        st.text("Full traceback:")
        st.text(traceback.format_exc())
        return {}
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def clean_lyrics_for_tfidf(lyrics_text):
    """Clean lyrics text similar to training preprocessing."""
    if not lyrics_text or lyrics_text.strip() == "":
        return ""
    
    lyrics_clean = lyrics_text.lower()
    lyrics_clean = re.sub(r'[^\w\s]', ' ', lyrics_clean)  # Remove punctuation
    lyrics_clean = re.sub(r'\s+', ' ', lyrics_clean)      # Multiple spaces to single
    lyrics_clean = lyrics_clean.strip()
    
    return lyrics_clean

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
        if words:
            doc2vec_vector = doc2vec_model.infer_vector(words)
            doc2vec_features = doc2vec_vector.astype(np.float32)
        
        return svd_features, doc2vec_features
        
    except Exception as e:
        st.error(f"Error processing lyrics: {str(e)}")
        return svd_features, doc2vec_features

def predict_hit(models, scaler, audio_features, svd_features, doc2vec_features, best_params):
    """Make prediction using the ensemble of models."""
    try:
        # Prepare audio features in the correct order (13 features including mode)
        audio_feature_order = [
            'danceability', 'energy', 'key_clean', 'loudness', 'mode_clean', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
            'duration_ms', 'time_signature'
        ]
        
        # Create audio feature vector
        audio_vector = np.array([audio_features.get(feat, 0.0) for feat in audio_feature_order], 
                               dtype=np.float32)
        
        # OPTIONAL: Reduce weight of instrumentalness feature (index 7)
        # This scales down the instrumentalness value to reduce its impact
        instrumentalness_weight = 0.3  # Reduce to 30% of original weight
        audio_vector[7] = audio_vector[7] * instrumentalness_weight
        
        # Combine all features: audio + SVD + Doc2Vec
        X_full = np.concatenate([audio_vector, svd_features, doc2vec_features])
        
        # Reshape for prediction
        X_full = X_full.reshape(1, -1)
        
        # Scale features
        X_scaled = scaler.transform(X_full)
        
        # Get predictions from all models
        predictions = []
        for model in models:
            pred = model.predict(X_scaled, num_iteration=model.best_iteration)
            predictions.append(pred[0])
        
        # Average predictions
        avg_probability = np.mean(predictions)
        
        # CHANGED: Use 30% threshold instead of the optimized threshold
        threshold = 0.3  # 30% threshold
        prediction = 1 if avg_probability >= threshold else 0
        
        return float(avg_probability), int(prediction), float(threshold)
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 0.5, 0, 0.3

def main():
    st.set_page_config(page_title="Hip-Hop Hit Predictor", layout="wide")
    st.title("🎵 Hip-Hop Hit Predictor")
    st.markdown("Upload an audio file and provide lyrics to predict if your song could be a hit!")
    
    # Load model components
    models, scaler, best_params, tfidf_vectorizer, svd_model, doc2vec_model = load_model_components()
    
    if models is None:
        st.error("Failed to load model components. Please check your model files.")
        return
    
    # Display model performance info
    if best_params and 'performance_metrics' in best_params:
        perf = best_params['performance_metrics']
        st.info(f"🎯 Model Performance: AUC {perf.get('cv_auc', 0):.3f} | "
               f"Optimized with {best_params.get('optimization', {}).get('n_trials', 'N/A')} trials")
    
    # File uploader
    st.subheader("🎵 Upload Audio File (Optional)")
    uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'flac', 'm4a'])
    
    # Feature input method selection
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/mp3')
        
        use_extracted = st.radio(
            "How would you like to set audio features?",
            ["Extract from uploaded audio", "Enter manually"],
            help="Extract features automatically or enter precise values manually"
        )
        
        if use_extracted == "Extract from uploaded audio":
            with st.spinner("Extracting audio features..."):
                extracted_features = extract_audio_features(uploaded_file)
            
            if extracted_features:
                st.success("✅ Audio features extracted successfully!")
                # Use extracted features as defaults, but allow user to modify
                st.info("📝 Extracted features (you can modify these values):")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    danceability = st.number_input("Danceability", min_value=0.0, max_value=100.0, 
                                                  value=float(extracted_features.get('danceability', 50)), step=1.0)
                    energy = st.number_input("Energy", min_value=0.0, max_value=100.0, 
                                            value=float(extracted_features.get('energy', 50)), step=1.0)
                with col2:
                    acousticness = st.number_input("Acousticness", min_value=0.0, max_value=100.0, 
                                                  value=float(extracted_features.get('acousticness', 10)), step=1.0)
                    speechiness = st.number_input("Speechiness", min_value=0.0, max_value=100.0, 
                                                 value=float(extracted_features.get('speechiness', 15)), step=1.0)
                with col3:
                    instrumentalness = st.number_input("Instrumentalness", min_value=0.0, max_value=100.0, 
                                                      value=float(extracted_features.get('instrumentalness', 5)), step=1.0)
                    liveness = st.number_input("Liveness", min_value=0.0, max_value=100.0, 
                                              value=float(extracted_features.get('liveness', 15)), step=1.0)
                with col4:
                    valence = st.number_input("Valence", min_value=0.0, max_value=100.0, 
                                             value=float(extracted_features.get('valence', 50)), step=1.0)
                    loudness = st.number_input("Loudness (0-1)", min_value=0.0, max_value=1.0, value=0.2, step=0.01,
                                            help="Average RMS loudness (0 = silent, 1 = full scale)")

                
                # Use extracted loudness and duration
                loudness = float(extracted_features.get('loudness', -10))
                duration_ms = float(extracted_features.get('duration_ms', 180000))
                duration_seconds = int(duration_ms / 1000)
                
                st.info(f"🔊 Extracted Loudness: {loudness:.1f}")
                st.info(f"⏱️ Extracted Duration: {duration_seconds} seconds")
            else:
                st.error("Failed to extract audio features. Please enter manually.")
                use_extracted = "Enter manually"
    
    # Manual input (either by choice or if no audio uploaded)
    if uploaded_file is None or (uploaded_file is not None and use_extracted == "Enter manually"):
        st.subheader("🎵 Audio Features (0-100 Scale)")
        st.info("💡 Enter values 0-100 (your model was trained on this scale)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            danceability = st.number_input("Danceability", min_value=0.0, max_value=100.0, value=50.0, step=1.0,
                                          help="How suitable for dancing (0-100)")
            energy = st.number_input("Energy", min_value=0.0, max_value=100.0, value=50.0, step=1.0,
                                    help="Intensity and activity (0-100)")
            valence = st.number_input("Valence", min_value=0.0, max_value=100.0, value=50.0, step=1.0,
                                     help="Musical positiveness (0-100)")
        
        with col2:
            acousticness = st.number_input("Acousticness", min_value=0.0, max_value=100.0, value=10.0, step=1.0,
                                          help="Acoustic confidence (0-100)")
            speechiness = st.number_input("Speechiness", min_value=0.0, max_value=100.0, value=15.0, step=1.0,
                                         help="Spoken word presence (0-100)")
            instrumentalness = st.number_input("Instrumentalness", min_value=0.0, max_value=100.0, value=5.0, step=1.0,
                                              help="Likelihood of no vocals (0-100)")
        
        with col3:
            liveness = st.number_input("Liveness", min_value=0.0, max_value=100.0, value=15.0, step=1.0,
                                      help="Live audience presence (0-100)")
        
        # Technical features
        st.markdown("**Technical Features:**")
        col1, col2 = st.columns(2)
        
        with col1:
            loudness = st.number_input("Loudness (dB)", min_value=-60.0, max_value=5.0, value=-10.0, step=0.1,
                                      help="Overall loudness in decibels")
        
        with col2:
            duration_seconds = st.number_input("Duration (seconds)", min_value=30, max_value=600, value=180, step=1,
                                             help="Song duration in seconds")
            duration_ms = duration_seconds * 1000
    
    # Lyrics input
    st.subheader("📝 Song Lyrics")
    st.info("💡 Lyrics are crucial for accurate prediction! The model uses advanced text analysis.")
    lyrics_text = st.text_area(
        "Enter the complete song lyrics:",
        height=200,
        placeholder="Paste your lyrics here. The more complete, the better the prediction...",
        help="The model analyzes lyrics using TF-IDF, SVD, and Doc2Vec for comprehensive text understanding."
    )
    
    # Song information inputs
    st.subheader("🎼 Song Information")
    
    # Basic song info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        key_options = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_name = st.selectbox("Song Key", key_options, index=0)
        key_map = {name: i for i, name in enumerate(key_options)}
        key_clean = key_map[key_name]
    
    with col2:
        mode = st.selectbox("Mode", ["Minor", "Major"], index=1)
        mode_clean = 0 if mode == "Minor" else 1
        
    with col3:
        tempo = st.number_input("Tempo (BPM)", 
                               min_value=60, max_value=200, value=120,
                               help="Enter the beats per minute for your song")
    
    # Time signature (2-digit format)
    time_sig_options = {
        "3/4 (Waltz)": 34,
        "4/4 (Common)": 44, 
        "5/4 (Irregular)": 54,
        "6/8 (Compound)": 68,
        "7/4 (Complex)": 74
    }
    time_sig_display = st.selectbox("Time Signature", list(time_sig_options.keys()), index=1)
    time_signature = time_sig_options[time_sig_display]
    
    # Create audio features dictionary
    audio_features = {
        'danceability': float(danceability),
        'energy': float(energy),
        'key_clean': int(key_clean),
        'loudness': float(loudness),
        'mode_clean': int(mode_clean),
        'speechiness': float(speechiness),
        'acousticness': float(acousticness),
        'instrumentalness': float(instrumentalness),
        'liveness': float(liveness),
        'valence': float(valence),
        'tempo': float(tempo),
        'duration_ms': float(duration_ms),
        'time_signature': int(time_signature)
    }
    
    # Process lyrics
    if lyrics_text.strip():
        with st.spinner("Processing lyrics..."):
            svd_features, doc2vec_features = process_lyrics_with_models(
                lyrics_text, tfidf_vectorizer, svd_model, doc2vec_model
            )
        st.success(f"✅ Lyrics processed: {len(lyrics_text.split())} words")
    else:
        svd_features = np.zeros(500, dtype=np.float32)
        doc2vec_features = np.zeros(300, dtype=np.float32)
        st.warning("⚠️ No lyrics provided - using zero vectors for text features")
    
    # Prediction button
    if st.button("🔮 Predict Hit Potential", type="primary", use_container_width=True):
        with st.spinner("Analyzing your song..."):
            try:
                # Create audio features dictionary here (after all variables are defined)
                audio_features = {
                    'danceability': float(danceability),
                    'energy': float(energy),
                    'key_clean': int(key_clean),
                    'loudness': float(loudness),
                    'mode_clean': int(mode_clean),
                    'speechiness': float(speechiness),
                    'acousticness': float(acousticness),
                    'instrumentalness': float(instrumentalness),
                    'liveness': float(liveness),
                    'valence': float(valence),
                    'tempo': float(tempo),
                    'duration_ms': float(duration_ms),
                    'time_signature': int(time_signature)
                }
                
                probability, prediction, threshold = predict_hit(
                    models, scaler, audio_features, svd_features, doc2vec_features, best_params
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
                    elif probability >= 0.3:  # CHANGED: Updated to reflect 30% threshold
                        st.warning(f"⭐ HIT: {probability*100:.1f}%")
                    else:
                        st.info(f"🌱 NON-HIT: {probability*100:.1f}%")
                
                # Simple metrics
                st.subheader("📊 Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Hit Probability", f"{probability*100:.1f}%")
                
                with col2:
                    st.metric("Prediction", "HIT" if prediction == 1 else "NON-HIT")
                
                # Show threshold info
                st.info(f"ℹ️ Using threshold: 30% - Songs need ≥30% probability to be classified as HITs")
                
                # Interpretation
                st.markdown("**Interpretation:**")
                if probability > 0.8:
                    st.success("🚀 **Exceptional Potential!** Your song has characteristics very similar to major hits.")
                elif probability > 0.6:
                    st.info("⭐ **Strong Potential!** Your song shows many characteristics of successful tracks.")
                elif probability >= 0.3:  # CHANGED: Updated for 30% threshold
                    st.warning("✅ **Hit Classification** - Your song meets the criteria for a hit track.")
                else:
                    st.error("🔧 **Non-Hit** - Song falls below the 30% hit threshold.")
                
                # Recommendations
                st.subheader("💡 Recommendations")
                if not lyrics_text.strip():
                    st.info("🎤 **Add lyrics** to significantly improve prediction accuracy!")
                
                # Feature analysis
                if audio_features['speechiness'] < 10:
                    st.info("🗣️ Consider increasing **speechiness** for hip-hop (typical range: 15-30)")
                
                if audio_features['energy'] < 40:
                    st.info("⚡ Hip-hop tracks typically have higher **energy** (50-80 range)")
                
                if audio_features['danceability'] < 50:
                    st.info("💃 Consider improving **danceability** for broader appeal (60-80 range)")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please check that all inputs are valid.")
    
    # Footer with model info
    st.markdown("---")
    st.markdown("**🤖 Model Information:**")
    if best_params:
        st.markdown(f"- **Algorithm**: LightGBM Ensemble ({len(models) if models else 0} models)")
        st.markdown(f"- **Features**: 813 total (13 audio + 500 SVD + 300 Doc2Vec)")
        st.markdown(f"- **Training**: Hip-hop dataset with lyrics analysis")
        st.markdown(f"- **Scale**: 0-100 for audio features")
        st.markdown(f"- **Hit Threshold**: 30% probability")

if __name__ == "__main__":
    main()