import whisper
import datetime
import wave
import contextlib
import numpy as np
import torch
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding, SpeechBrainPretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
import os
import pandas as pd
import argparse
from tqdm import tqdm
import logging
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transcription.log"),
        logging.StreamHandler()
    ]
)

class LocalSpeechBrainPretrainedSpeakerEmbedding(SpeechBrainPretrainedSpeakerEmbedding):
    def __init__(self, embedding="speechbrain/spkrec-ecapa-voxceleb", device=None, use_auth_token=None, savedir="/my/local/huggingface_repo/spkrec-ecapa-voxceleb"):
        """
        Derived class that allows specifying a local directory (savedir) from which
        to load the SpeechBrain model.
        """
        if "@" in embedding:
            self.embedding = embedding.split("@")[0]
            self.revision = embedding.split("@")[1]
        else:
            self.embedding = embedding
            self.revision = None

        self.device = device or torch.device("cpu")
        self.use_auth_token = use_auth_token
        self.savedir = savedir

        from speechbrain.inference import EncoderClassifier as SpeechBrain_EncoderClassifier
        self.classifier_ = SpeechBrain_EncoderClassifier.from_hparams(
            source=self.embedding,
            savedir=self.savedir,
            run_opts={"device": self.device},
            use_auth_token=self.use_auth_token,
            revision=self.revision,
        )

    def to(self, device: torch.device):
        self.device = device
        from speechbrain.inference import EncoderClassifier as SpeechBrain_EncoderClassifier
        self.classifier_ = SpeechBrain_EncoderClassifier.from_hparams(
            source=self.embedding,
            savedir=self.savedir,
            run_opts={"device": device},
            use_auth_token=self.use_auth_token,
            revision=self.revision,
        )
        return self

def load_whisper_model(model_size: str, audio_lang: str):
    """
    Load the Whisper ASR model based on model size and audio language.
    
    For English audio (if not large), load the language‑specific variant.
    """
    model_name = model_size
    if audio_lang.lower() == "english" and model_size != "large":
        model_name += ".en"
    logging.info(f"Loading Whisper model: {model_name}")
    return whisper.load_model(model_size)

def load_embedding_model():
    """
    Load the pretrained speaker embedding model from SpeechBrain via pyannote.audio,
    using the local saved directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading speaker embedding model on {device}")
    return LocalSpeechBrainPretrainedSpeakerEmbedding(
        embedding="spkrec-ecapa-voxceleb",
        device=device,
        savedir="spkrec-ecapa-voxceleb"  # Update this path if necessary
    )

def load_audio_object():
    """
    Load the Audio object from pyannote.
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
    if waveform.shape[0] > 1:
        waveform = waveform.mean(axis=0, keepdims=True)
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
    num_speakers: int,
    seed: int,
    audio_lang: str
):
    """
    Process the audio file: transcribe, generate embeddings, perform clustering,
    and return the segmented transcript with speaker labels.
    
    Decoding options are adjusted based on whether the audio is Arabic or English.
    """
    logging.info(f"Transcribing audio file: {file_path}")

    # Set seeds for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set decoding options based on audio language.
    if audio_lang.lower() == "arabic":
        decoding_options = {
            "task": "translate",                    # Translate Arabic speech to English text.
            "language": "ar",                       # Source language is Arabic.
            "beam_size": 5,
            "best_of": 5,
            "temperature": 0.0,
            "patience": 1.0,
            "length_penalty": 1.0,
            "suppress_tokens": [],
            "initial_prompt": "Call center conversation regarding customer support.",
            "condition_on_previous_text": True,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "compression_ratio_threshold": 2.4,
        }
    else:  # For English audio.
        decoding_options = {
            "task": "transcribe",                   # Transcribe English speech.
            "language": "en",                       # Source language is English.
            "beam_size": 5,
            "best_of": 5,
            "temperature": 0.0,
            "patience": 1.0,
            "length_penalty": 1.0,
            "suppress_tokens": [],
            "initial_prompt": "Call center conversation regarding customer support.",
            "condition_on_previous_text": True,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "compression_ratio_threshold": 2.4,
        }
    
    result = whisper_model.transcribe(file_path, **decoding_options)
    segments = result["segments"]

    # Determine audio duration using wave.
    with contextlib.closing(wave.open(file_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    logging.info("Extracting embeddings from audio segments")
    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment, duration, file_path, embedding_model, audio_obj)
    embeddings = np.nan_to_num(embeddings)

    logging.info(f"Performing speaker clustering with {num_speakers} speakers")
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
            transcript_lines.append(f"\n{speaker} at {start_time}\n")
        transcript_lines.append(segment["text"].strip() + " ")
        prev_speaker = speaker

    transcript = "\n".join(transcript_lines)
    return transcript

def process_files_from_csv(csv_path, output_dir, model_size, audio_lang, num_speakers, seed):
    """
    Process all files listed in the CSV.
    CSV should have columns: 'filename' and 'file_path'
    """
    whisper_model = load_whisper_model(model_size, audio_lang)
    embedding_model = load_embedding_model()
    audio_obj = load_audio_object()
    
    try:
        df = pd.read_csv(csv_path)
        if 'filename' not in df.columns or 'file_path' not in df.columns:
            logging.error("CSV must contain 'filename' and 'file_path' columns")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
            filename = row['filename']
            file_path = row['file_path']
            
            if not os.path.exists(file_path):
                logging.warning(f"File not found: {file_path}. Skipping.")
                continue
                
            try:
                base_name = os.path.splitext(filename)[0]
                output_file = os.path.join(output_dir, f"{base_name}.txt")
                
                logging.info(f"Processing {filename}")
                transcript = run_diarization(
                    file_path,
                    whisper_model,
                    embedding_model,
                    audio_obj,
                    num_speakers,
                    seed,
                    audio_lang
                )
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(transcript)
                logging.info(f"Saved transcript to {output_file}")
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
    except Exception as e:
        logging.error(f"Error reading CSV file: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch audio transcription with speaker diarization")
    parser.add_argument("--csv", type=str, default="audio_files.csv", help="Path to CSV file with filename and file_path columns")
    parser.add_argument("--output", type=str, default="transcripts", help="Directory to save transcripts")
    parser.add_argument("--model", type=str, default="large-v3", choices=["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"],
                        help="Whisper model size")
    parser.add_argument("--audio_lang", type=str, default="english", choices=["arabic", "english"], help="Language of the input audio")
    parser.add_argument("--speakers", type=int, default=2, help="Number of speakers to detect")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    logging.info(f"Starting batch processing with model={args.model}, speakers={args.speakers}, seed={args.seed}")
    process_files_from_csv(args.csv, args.output, args.model, args.audio_lang, args.speakers, args.seed)
    logging.info("Batch processing complete")
