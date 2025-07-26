# streamlit_app.py
import streamlit as st
import librosa
import numpy as np
import tempfile
import os
import joblib
import json

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

def extract_audio_features_approximation(audio_file):
    """Extract approximate Spotify-like features from audio file."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    try:
        # Load audio file
        y, sr = librosa.load(tmp_path)
        
        features = {}
        
        # 1. Danceability
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['danceability'] = np.mean(zcr) * 10  # Normalize
        
        # 2. Energy
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['energy'] = np.mean(spectral_centroids) / 10000  # Normalize
        
        # 3. Loudness (RMS energy in dB-like scale)
        rms = librosa.feature.rms(y=y)[0]
        features['loudness'] = 20 * np.log10(np.mean(rms) + 1e-8)  # Convert to dB
        
        # 4. Speechiness
        features['speechiness'] = np.var(zcr) * 100  # Heuristic
        
        # 5. Acousticness
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['acousticness'] = np.mean(spectral_rolloff) / 5000  # Normalize
        
        # 6. Instrumentalness
        silence_frames = np.sum(rms < np.percentile(rms, 10))
        features['instrumentalness'] = silence_frames / len(rms)
        
        # 7. Liveness
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['liveness'] = np.mean(spectral_flatness) * 10  # Normalize
        
        # 8. Valence
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['valence'] = np.var(mfccs) / 100  # Normalize
        
        # 9. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        
        # 10. Duration
        features['duration_ms'] = len(y) * 1000 / sr
        
        # Note: 'key' and 'time_signature' are handled separately in main()
        # They are NOT extracted here but added by user input.
        
        # Normalize all features to 0-1 range (Spotify style)
        # Only for the features extracted here
        extracted_feature_keys = ['danceability', 'energy', 'loudness', 'speechiness',
                                 'acousticness', 'instrumentalness', 'liveness', 'valence',
                                 'tempo', 'duration_ms']
        for key in extracted_feature_keys:
            features[key] = max(0.0, min(1.0, features[key]))
        
        return features
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)

def predict_hit(models, scaler, features):
    """Make prediction using the ensemble of models."""
    # Prepare features in the correct order for your model
    # This MUST match the order used during training
    feature_order = [
        'danceability', 'energy', 'key', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence',
        'tempo', 'duration_ms', 'time_signature'
    ]
    
    # Create feature vector in the correct order
    # Ensure all keys in feature_order are present in the features dict
    try:
        feature_vector = np.array([features[feat] for feat in feature_order]).reshape(1, -1)
    except KeyError as e:
        st.error(f"Missing feature in input: {e}. Required features: {feature_order}")
        st.write(f"Available features: {list(features.keys())}")
        return None
    
    # Scale features
    try:
        scaled_features = scaler.transform(feature_vector)
    except ValueError as e:
        st.error(f"Error scaling features: {e}. Check if the scaler was trained on {feature_vector.shape[1]} features.")
        return None
    
    # Get predictions from all models
    predictions = []
    for model in models:
        pred = model.predict(scaled_features, num_iteration=model.best_iteration)
        predictions.append(pred[0])
    
    # Average predictions
    avg_probability = np.mean(predictions)
    return avg_probability

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
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format='audio/mp3')
        
        # Extract features automatically
        with st.spinner("Extracting audio features..."):
            audio_features = extract_audio_features_approximation(uploaded_file)
        
        # Display extracted features
        st.subheader("🎵 Extracted Audio Features")
        cols = st.columns(4)
        cols[0].metric("Danceability", f"{audio_features['danceability']:.3f}")
        cols[1].metric("Energy", f"{audio_features['energy']:.3f}")
        cols[2].metric("Tempo", f"{audio_features['tempo']:.0f} BPM")
        cols[3].metric("Valence", f"{audio_features['valence']:.3f}")
        
        # User input for features that can't be reliably extracted
        st.subheader("📝 Additional Information Required")
        st.info("Please provide these details for accurate prediction:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            key_options = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_name = st.selectbox("Key", key_options, index=0)
            # Convert key name to number (0-11)
            key_map = {name: i for i, name in enumerate(key_options)}
            audio_features['key'] = key_map[key_name] # Add 'key' to the features dict
        
        with col2:
            # Mode selection (0 for minor, 1 for major)
            mode_name = st.selectbox("Mode", ['Minor', 'Major'], index=1)
            mode_map = {'Minor': 0, 'Major': 1}
            audio_features['mode'] = mode_map[mode_name] # Add 'mode' to the features dict
            # Note: The original feature_order list has 'key' but not 'mode'.
            # If your model was trained with 'mode', you need to update feature_order.
            # If it was trained with 'key' only, you might not need 'mode'.
            # For now, I'll assume 'mode' was intended to be 'key' in feature_order.
            # Let's check the feature_order in predict_hit...
            # Ah, feature_order has 'key'. So we need to make sure 'key' is in audio_features.
            # The user provides 'key_name' and we convert it to 'key' (number).
            # The 'mode' is separate. Let's assume for now the model only uses 'key'.
            # If the model actually uses 'mode', the feature_order needs to be updated.
            
        with col3:
            time_signature = st.selectbox("Time Signature", [3, 4, 5, 6, 7], index=1)
            audio_features['time_signature'] = time_signature # Add 'time_signature' to the features dict
        
        # Display the manually entered features
        cols = st.columns(3)
        cols[0].metric("Key", key_name)
        cols[1].metric("Mode", mode_name)
        cols[2].metric("Time Signature", f"{time_signature}/4")
        
        # Prediction button
        if st.button("🔮 Predict Hit Potential", type="primary", use_container_width=True):
            with st.spinner("Analyzing your song..."):
                try:
                    # Make prediction
                    probability = predict_hit(models, scaler, audio_features)
                    
                    if probability is not None:
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
                    else:
                        st.error("Prediction failed due to missing features or scaler mismatch.")
                        
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()