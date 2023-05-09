import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import music21
import warnings

warnings.filterwarnings('ignore')

st.title("Audio Analysis Tool")
st.write("Upload an audio file to analyze its tempo, key, or display a beat graph.")


def estimate_tempo(filename):
    y, sr = librosa.load(filename, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return tempo

def plot_beat(filename):
    y, sr = librosa.load(filename, sr=None)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    times = librosa.frames_to_time(beats, sr=sr)
    start = np.random.randint(0, len(y) - sr*10)
    end = start + sr*10
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_title('Waveform with Beat Markers')
    t = np.linspace(0, len(y) / sr, len(y))
    ax.plot(t, y, alpha=0.8)
    ax.set_xlim(start / sr, end / sr)
    ax.vlines(times[(times >= start / sr) & (times < end / sr)], -1, 1, color='r')
    return fig

def estimate_key(filename):
    y, sr = librosa.load(filename, sr=None)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    pitches = np.argmax(chroma, axis=0)
    stream = music21.stream.Stream()
    for p in pitches:
        n = music21.note.Note()
        n.pitch.midi = p
        stream.append(n)
    key = stream.analyze('key')
    return key.tonic.name, key.mode


uploaded_file = st.file_uploader("Upload an audio file (MP3, WAV, etc.)", type=["mp3", "wav", "ogg", "flac"])
options = ['Select an option', 'Identify BPM', 'Identify Key', 'Plot Beat Graph']
selected_option = st.selectbox("Choose an analysis option:", options)

if uploaded_file is not None and selected_option != 'Select an option':
    with st.spinner("Analyzing audio file..."):
        if selected_option == 'Identify BPM':
            tempo = estimate_tempo(uploaded_file)
            st.write(f"Tempo of the uploaded audio file: {tempo:.2f} BPM")
        elif selected_option == 'Identify Key':
            key, mode = estimate_key(uploaded_file)
            st.write(f"Estimated key of the uploaded audio file: {key} {mode}")
        elif selected_option == 'Plot Beat Graph':
            fig = plot_beat(uploaded_file)
            st.pyplot(fig)
