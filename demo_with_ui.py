import streamlit as st
import whisper
import datetime
import wave
import contextlib
import numpy as np
import torch
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
import tempfile
import os

@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str, language: str):
    """
    Load the Whisper ASR model based on model size and language.
    """
    model_name = model_size
    if language == 'English' and model_size != 'large':
        model_name += '.en'
    return whisper.load_model(model_size)

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """
    Load the pretrained speaker embedding model from SpeechBrain via pyannote.audio.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=device)

@st.cache_resource(show_spinner=False)
def load_audio_object():
    """
    Cache the Audio object from pyannote.
    """
    return Audio()

def segment_embedding(segment, duration, audio_path, embedding_model, audio_obj):
    """
    Given a segment from Whisper, calculate the speaker embedding.
    """
    start = segment["start"]
    end = min(duration, segment["end"])  # Adjust for potential overshoot
    clip = Segment(start, end)
    waveform, sample_rate = audio_obj.crop(audio_path, clip)
    # Convert stereo to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(axis=0, keepdims=True)
    # Add a batch dimension: shape becomes (1, channels, samples)
    embedding = embedding_model(waveform[None])
    return embedding

def format_time(secs: float):
    """
    Format seconds into hh:mm:ss.
    """
    return str(datetime.timedelta(seconds=round(secs)))

def run_diarization(
    file_path: str, 
    whisper_model,
    embedding_model,
    audio_obj,
    num_speakers: int
):
    """
    Process the audio file: transcribe, generate embeddings, perform clustering,
    and return the segmented transcript with speaker labels.
    """
    st.info("Transcribing audio file...")
    result = whisper_model.transcribe(file_path, language='en')
    segments = result["segments"]

    # Determine audio duration using wave
    with contextlib.closing(wave.open(file_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    st.info("Extracting embeddings from audio segments...")
    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment, duration, file_path, embedding_model, audio_obj)

    embeddings = np.nan_to_num(embeddings)

    st.info("Performing speaker clustering...")
    clustering = AgglomerativeClustering(n_clusters=num_speakers).fit(embeddings)
    labels = clustering.labels_

    for i, segment in enumerate(segments):
        segment["speaker"] = f'SPEAKER {labels[i] + 1}'

    transcript_lines = []
    prev_speaker = None
    for segment in segments:
        speaker = segment["speaker"]
        start_time = format_time(segment["start"])
        if speaker != prev_speaker:
            transcript_lines.append(f"\n**{speaker}  at {start_time}**\n")
        transcript_lines.append(segment["text"].strip() + " ")
        prev_speaker = speaker

    transcript = "\n".join(transcript_lines)
    return transcript

def save_uploaded_file(uploaded_file) -> str:
    """
    Save the uploaded file to a temporary file so that we can pass its path to our processing functions.
    """
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name


def main():
    st.set_page_config(page_title="CC Transcript Generation", layout="wide")
    
    st.title("Speaker Diarization & Transcription Demo")
    st.markdown(
        """
        Welcome to the demo application that performs **speaker diarization** and **speech transcription** using state-of-the-art AI models.  whisper
        Upload an audio (WAV) file of a conversation or call center recording and adjust the parameters from the sidebar.  
        The application will transcribe the audio and label segments with individual speaker identities.
        """
    )

    st.sidebar.header("Configuration")
    
    num_speakers = st.sidebar.number_input("Number of Speakers", min_value=1, max_value=10, value=2, step=1)
    language = st.sidebar.selectbox("Language", options=["any", "English"], index=0)
    model_size = st.sidebar.selectbox("Model Size", options=["tiny", "base", "small", "medium", "large","large-v1","large-v2","large-v3"], index=2)

    st.sidebar.markdown(
        """
        Larger models may yield improved accuracy but require more compute resources.
        """
    )
    
    uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav","m4"])
    
    if uploaded_file is not None:
        st.success(f"File uploaded: **{uploaded_file.name}**")
        
        file_path = save_uploaded_file(uploaded_file)
        
        with st.spinner("Loading models..."):
            whisper_model = load_whisper_model(model_size, language)
            embedding_model = load_embedding_model()
            audio_obj = load_audio_object()
        
        if st.button("Run Diarization & Transcription"):
            with st.spinner("Processing audio... This may take a while."):
                transcript = run_diarization(file_path, whisper_model, embedding_model, audio_obj, num_speakers)
            
            st.success("Processing complete!")
            st.markdown("### Transcript")
            st.markdown(transcript)
            
            st.download_button(
                label="Download Transcript",
                data=transcript,
                file_name="transcript.txt",
                mime="text/plain"
            )
        
        os.remove(file_path)
    else:
        st.info("Please upload a WAV file to begin.")

if __name__ == "__main__":
    main()
