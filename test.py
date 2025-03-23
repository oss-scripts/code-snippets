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
import os
import pandas as pd
import argparse
from tqdm import tqdm
import logging
import requests
import time
import re
import subprocess
import tempfile
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transcription.log"),
        logging.StreamHandler()
    ]
)

def load_whisper_model(model_size: str, language: str):
    """
    Load the Whisper ASR model based on model size and language.
    """
    model_name = model_size
    if language == 'English' and model_size != 'large':
        model_name += '.en'
    logging.info(f"Loading Whisper model: {model_name}")
    return whisper.load_model(model_size)

def load_embedding_model():
    """
    Load the pretrained speaker embedding model from SpeechBrain via pyannote.audio.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading speaker embedding model on {device}")
    return PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=device)

def load_audio_object():
    """
    Load the Audio object from pyannote.
    """
    return Audio()

def convert_audio_format(input_path, output_path=None):
    """
    Convert audio file to a standard format that pyannote can process.
    Uses ffmpeg to convert g.711 mu-law to standard PCM WAV.
    """
    if output_path is None:
        # Create a temporary file with .wav extension
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"temp_{os.path.basename(input_path)}")
    
    logging.info(f"Converting audio format: {input_path} -> {output_path}")
    
    try:
        # Convert to standard 16-bit PCM WAV at 16kHz (good for speech processing)
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except Exception as e:
        logging.error(f"Error converting audio format: {str(e)}")
        return input_path  # Return original path if conversion fails

def get_audio_duration(audio_path):
    """
    Get the duration of an audio file using ffprobe.
    """
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        logging.error(f"Error getting audio duration: {str(e)}")
        # Fall back to wave if ffprobe fails
        try:
            with contextlib.closing(wave.open(audio_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                return frames / float(rate)
        except Exception as e2:
            logging.error(f"Wave fallback also failed: {str(e2)}")
            return 0.0

def segment_embedding(segment, duration, audio_path, embedding_model, audio_obj):
    """
    Given a segment from Whisper, calculate the speaker embedding.
    """
    start = segment["start"]
    end = min(duration, segment["end"])  # Adjust for potential overshoot
    clip = Segment(start, end)
    
    try:
        waveform, sample_rate = audio_obj.crop(audio_path, clip)
        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(axis=0, keepdims=True)
        # Add a batch dimension: shape becomes (1, channels, samples)
        embedding = embedding_model(waveform[None])
        return embedding
    except Exception as e:
        logging.error(f"Error in segment embedding: {str(e)}")
        # Return zeros array as fallback
        return np.zeros(192)

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
    logging.info(f"Transcribing audio file: {file_path}")
    # Whisper can handle various formats through ffmpeg
    result = whisper_model.transcribe(file_path, language='en')
    segments = result["segments"]

    # For diarization, convert to standard format that pyannote can handle
    converted_audio_path = convert_audio_format(file_path)
    duration = get_audio_duration(converted_audio_path)

    logging.info("Extracting embeddings from audio segments")
    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment, duration, converted_audio_path, embedding_model, audio_obj)

    embeddings = np.nan_to_num(embeddings)

    logging.info(f"Performing speaker clustering with {num_speakers} speakers")
    clustering = AgglomerativeClustering(n_clusters=num_speakers).fit(embeddings)
    labels = clustering.labels_

    # Clean up temporary file if we created one
    if converted_audio_path != file_path and os.path.exists(converted_audio_path):
        try:
            os.remove(converted_audio_path)
            logging.info(f"Cleaned up temporary file: {converted_audio_path}")
        except Exception as e:
            logging.warning(f"Could not remove temporary file: {str(e)}")

    # Prepare data for LLM post-processing
    transcript_data = []
    for i, segment in enumerate(segments):
        transcript_data.append({
            "speaker": f'SPEAKER {labels[i] + 1}',
            "start_time": format_time(segment["start"]),
            "text": segment["text"].strip(),
            "start_seconds": segment["start"]
        })

    # Return the transcript data
    return transcript_data

def process_transcript_with_llm(transcript_data, llm_endpoint="http://localhost:8503/llama_generate"):
    """
    Process the transcript with LLM to:
    1. Identify agent vs. customer speakers
    2. Clean up transcript text
    3. Mask PII information
    """
    logging.info("Processing transcript with LLM for speaker identification and cleanup")
    
    # Create a string format the LLM can easily understand
    formatted_input = ""
    for segment in transcript_data:
        formatted_input += f"{segment['speaker']} at {segment['start_time']}: {segment['text']}\n\n"
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in analyzing customer service call transcripts. Your task is to:

1. Identify which speaker is the Customer and which is the Agent.
2. Clean up the transcript by removing noise, repetitions, and correcting any obvious transcription errors.
3. Mask PII (Personally Identifiable Information) like full names, complete addresses, credit card numbers, SSNs, etc. with [MASKED PII].
4. Return the improved transcript with correct speaker labels (AGENT or CUSTOMER).

Formatting guidelines:
- Format each speaker turn as "AGENT at [timestamp]: [text]" or "CUSTOMER at [timestamp]: [text]"
- Preserve all timestamps exactly as provided
- Only assign speakers as either AGENT or CUSTOMER
- Ensure the transcript flows naturally with appropriate turn-taking
- Remove filler words and speech artifacts that don't add meaning<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Please process this call transcript, identifying which speaker is the agent and which is the customer, cleaning up the text, and masking any PII:

{formatted_input}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    try:
        response = requests.post(
            llm_endpoint,
            json={
                "prompt": [prompt],
                "kwargs": {
                    "temperature": 0.1,
                    "max_tokens": 4096,
                    "top_p": 0.9
                }
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            processed_transcript = result[0]['outputs'][0]['text']
            logging.info("LLM successfully processed the transcript")
            return processed_transcript
        else:
            logging.error(f"Error from LLM API: {response.status_code}")
            # Fall back to original transcript with generic speaker labels
            return "\n".join([
                f"{segment['speaker']} at {segment['start_time']}: {segment['text']}"
                for segment in transcript_data
            ])
    except Exception as e:
        logging.error(f"Error during LLM processing: {str(e)}")
        # Fall back to original transcript with generic speaker labels
        return "\n".join([
            f"{segment['speaker']} at {segment['start_time']}: {segment['text']}"
            for segment in transcript_data
        ])

def process_files_from_csv(csv_path, output_dir, model_size, language, num_speakers, use_llm=True, llm_endpoint="http://localhost:8503/llama_generate"):
    """
    Process all files listed in the CSV.
    CSV should have columns: 'filename' and 'file_path'
    """
    # Load models once to reuse for all files
    whisper_model = load_whisper_model(model_size, language)
    embedding_model = load_embedding_model()
    audio_obj = load_audio_object()
    
    try:
        df = pd.read_csv(csv_path)
        if 'filename' not in df.columns or 'file_path' not in df.columns:
            logging.error("CSV must contain 'filename' and 'file_path' columns")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
            filename = row['filename']
            file_path = row['file_path']
            
            if not os.path.exists(file_path):
                logging.warning(f"File not found: {file_path}. Skipping.")
                continue
                
            try:
                # Generate output filename (same as input but with .txt extension)
                base_name = os.path.splitext(filename)[0]
                output_file = os.path.join(output_dir, f"{base_name}.txt")
                
                logging.info(f"Processing {filename}")
                transcript_data = run_diarization(
                    file_path,
                    whisper_model,
                    embedding_model,
                    audio_obj,
                    num_speakers
                )
                
                # Use LLM to improve transcript if enabled
                if use_llm:
                    final_transcript = process_transcript_with_llm(transcript_data, llm_endpoint)
                else:
                    # Format without LLM processing
                    transcript_lines = []
                    prev_speaker = None
                    for segment in transcript_data:
                        speaker = segment["speaker"]
                        start_time = segment["start_time"]
                        if speaker != prev_speaker:
                            transcript_lines.append(f"\n{speaker} at {start_time}\n")
                        transcript_lines.append(segment["text"] + " ")
                        prev_speaker = speaker
                    final_transcript = "\n".join(transcript_lines)
                
                # Save transcript
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(final_transcript)
                
                logging.info(f"Saved transcript to {output_file}")
                
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
    
    except Exception as e:
        logging.error(f"Error reading CSV file: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch audio transcription with speaker diarization and LLM enhancement")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file with filename and file_path columns")
    parser.add_argument("--output", type=str, default="transcripts", help="Directory to save transcripts")
    parser.add_argument("--model", type=str, default="small", choices=["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"], 
                        help="Whisper model size")
    parser.add_argument("--language", type=str, default="English", choices=["any", "English"], help="Language setting")
    parser.add_argument("--speakers", type=int, default=2, help="Number of speakers to detect")
    parser.add_argument("--llm", action="store_true", default=True, help="Use LLM for transcript enhancement")
    parser.add_argument("--llm-endpoint", type=str, default="http://localhost:8503/llama_generate", 
                        help="LLM API endpoint URL")
    
    args = parser.parse_args()
    
    logging.info(f"Starting batch processing with model={args.model}, speakers={args.speakers}, llm={args.llm}")
    process_files_from_csv(
        args.csv, 
        args.output, 
        args.model, 
        args.language, 
        args.speakers,
        args.llm,
        args.llm_endpoint
    )
    logging.info("Batch processing complete")
