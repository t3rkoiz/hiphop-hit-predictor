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
        
        # Apply optimized threshold if available
        threshold = best_params.get('threshold_optimization', {}).get('best_threshold', 0.5)
        prediction = 1 if avg_probability >= threshold else 0
        
        return float(avg_probability), int(prediction), float(threshold)
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 0.5, 0, 0.5

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
    st.info("💡 Use 0-100 scale for audio features (matching your training data)")
    
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
    
    # Audio features (0-100 scale to match training data)
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
    
    # Display current features
    st.subheader("📊 Current Feature Summary")
    cols = st.columns(4)
    cols[0].metric("Tempo", f"{tempo} BPM")
    cols[1].metric("Danceability", f"{danceability}/100")
    cols[2].metric("Energy", f"{energy}/100")
    cols[3].metric("Valence", f"{valence}/100")
    
    with st.expander("📊 View All Features"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Key", f"{key_name} {mode}")
            st.metric("Time Signature", time_sig_display)
            st.metric("Duration", f"{duration_seconds}s")
            st.metric("Loudness", f"{loudness} dB")
        with col2:
            st.metric("Acousticness", f"{acousticness}/100")
            st.metric("Speechiness", f"{speechiness}/100")
            st.metric("Instrumentalness", f"{instrumentalness}/100")
            st.metric("Liveness", f"{liveness}/100")
        with col3:
            st.info("**Scale Used:**")
            st.markdown("- Audio features: 0-100")
            st.markdown("- Key: 0-11 (C=0)")
            st.markdown("- Mode: 0=Minor, 1=Major")
            st.markdown("- Time sig: 2-digit (4/4=44)")
    
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
    
    # Feature processing status
    st.subheader("🔧 Feature Processing Status")
    cols = st.columns(3)
    cols[0].metric("Audio Features", "13", help="Spotify-like features (0-100 scale)")
    cols[1].metric("TF-IDF + SVD", "500", help="Text features from lyrics")
    cols[2].metric("Doc2Vec", "300", help="Document embeddings")
    
    # Prediction button
    if st.button("🔮 Predict Hit Potential", type="primary", use_container_width=True):
        with st.spinner("Analyzing your song..."):
            try:
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
                    elif probability > 0.5:
                        st.warning(f"⭐ PROMISING: {probability*100:.1f}%")
                    else:
                        st.info(f"🌱 DEVELOPING: {probability*100:.1f}%")
                
                # Detailed results
                st.subheader("📊 Detailed Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Hit Probability", f"{probability:.3f}")
                    st.metric("Prediction", "HIT" if prediction == 1 else "NON-HIT")
                
                with col2:
                    st.metric("Threshold Used", f"{threshold:.3f}")
                    st.metric("Confidence", f"{abs(probability - threshold):.3f}")
                
                with col3:
                    if best_params.get('performance_metrics'):
                        perf = best_params['performance_metrics']
                        st.metric("Model AUC", f"{perf.get('cv_auc', 0):.3f}")
                        st.metric("Model Precision", f"{perf.get('best_precision', 0):.3f}")
                
                # Interpretation
                st.markdown("**Interpretation:**")
                if probability > 0.8:
                    st.success("🚀 **Exceptional Potential!** Your song has characteristics very similar to major hits.")
                elif probability > 0.6:
                    st.info("⭐ **Strong Potential!** Your song shows many characteristics of successful tracks.")
                elif probability > 0.4:
                    st.warning("🛠️ **Room for Improvement** - Consider refining key musical elements.")
                else:
                    st.error("🔧 **Early Development** - Song needs significant work to reach hit potential.")
                
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
        st.markdown(f"- **Training**: Clean dataset with lyrics-only tracks")
        st.markdown(f"- **Scale**: 0-100 for audio features (production-ready)")
        
        if 'optimization' in best_params:
            opt = best_params['optimization']
            st.markdown(f"- **Optimization**: {opt.get('n_trials', 'N/A')} Optuna trials")

if __name__ == "__main__":
    main()