#!/usr/bin/env python3
"""
Yoga CD Transcript and Smart Splitting Tool

This script processes yoga audio CDs to:
1. Combine fragmented tracks into coherent single audio files
2. Convert between audio formats while preserving quality
3. Transcribe audio using Whisper with speaker diarization
4. Intelligently split audio based on transcript analysis

Author: Built for Rajyogi Caruso CD Processing
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import json
from tqdm import tqdm

# Audio processing
from pydub import AudioSegment

# Transcription
import whisper
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class YogaCDProcessor:
    """Main processor class for yoga CD operations"""
    
    def __init__(self, 
                 use_openai_whisper: bool = False,
                 target_sample_rate: int = 16000,
                 save_debug_files: bool = True):
        self.use_openai_whisper = use_openai_whisper
        self.target_sample_rate = target_sample_rate
        self.save_debug_files = save_debug_files
        self.whisper_model = None
        
    def check_file_exists(self, file_path: Path, prompt_overwrite: bool = True) -> bool:
        """
        Check if file exists and optionally prompt user for overwrite permission
        
        Args:
            file_path: Path to check
            prompt_overwrite: Whether to prompt user for overwrite confirmation
            
        Returns:
            True if we should proceed (file doesn't exist or user confirmed overwrite)
        """
        if not file_path.exists():
            return True
            
        if not prompt_overwrite:
            return False
            
        response = input(f"File {file_path} already exists. Overwrite? (y/N): ").strip().lower()
        return response in ['y', 'yes']
    
    def find_audio_files(self, directory: Path) -> List[Path]:
        """
        Find all audio files in a directory, sorted naturally
        
        Args:
            directory: Directory to search
            
        Returns:
            List of audio file paths
        """
        audio_extensions = {'.m4a', '.mp3', '.wav', '.aac', '.flac', '.ogg'}
        audio_files = []
        
        for file_path in directory.iterdir():
            if file_path.suffix.lower() in audio_extensions:
                audio_files.append(file_path)
        
        # Sort naturally (1, 2, 10 instead of 1, 10, 2)
        audio_files.sort(key=lambda x: self._natural_sort_key(x.name))
        return audio_files
    
    def _natural_sort_key(self, text: str) -> List:
        """Create a natural sorting key for filenames with numbers"""
        import re
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', text)]
    
    def get_audio_info(self, file_path: Path) -> dict:
        """
        Get basic information about an audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        try:
            audio = AudioSegment.from_file(str(file_path))
            return {
                'duration_ms': len(audio),
                'duration_seconds': len(audio) / 1000,
                'channels': audio.channels,
                'frame_rate': audio.frame_rate,
                'sample_width': audio.sample_width,
                'file_size_mb': file_path.stat().st_size / (1024 * 1024)
            }
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return {}
    
    def combine_audio_files(self, 
                          audio_files: List[Path], 
                          output_path: Path,
                          crossfade_ms: int = 500) -> bool:
        """
        Combine multiple audio files into a single master file
        
        Args:
            audio_files: List of audio file paths to combine
            output_path: Where to save the combined audio
            crossfade_ms: Crossfade duration in milliseconds
            
        Returns:
            True if successful, False otherwise
        """
        if not audio_files:
            print("No audio files to combine")
            return False
        
        try:
            print(f"Combining {len(audio_files)} audio files...")
            combined_audio = AudioSegment.empty()
            
            for i, file_path in enumerate(tqdm(audio_files, desc="Combining files")):
                print(f"Processing: {file_path.name}")
                
                # Load audio file
                audio = AudioSegment.from_file(str(file_path))
                
                if i == 0:
                    # First file, add as-is
                    combined_audio = audio
                else:
                    # Subsequent files, add with crossfade
                    combined_audio = combined_audio.append(audio, crossfade=crossfade_ms)
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export combined audio (keep original format for master)
            print(f"Exporting combined audio to: {output_path}")
            # Use correct format identifier for m4a files
            export_format = output_path.suffix[1:]
            if export_format.lower() == 'm4a':
                export_format = 'mp4'
            combined_audio.export(str(output_path), format=export_format)
            
            # Print summary
            total_duration = len(combined_audio) / 1000 / 60  # minutes
            print(f"Combined audio: {total_duration:.2f} minutes")
            
            return True
            
        except Exception as e:
            print(f"Error combining audio files: {e}")
            return False
    
    def convert_audio_format(self, 
                           input_path: Path, 
                           output_path: Path,
                           target_format: str = 'mp3',
                           bitrate: str = '192k') -> bool:
        """
        Convert audio file to different format while preserving quality
        
        Args:
            input_path: Source audio file
            output_path: Destination audio file
            target_format: Target format (mp3, wav, etc.)
            bitrate: Bitrate for lossy formats
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Converting {input_path.name} to {target_format}")
            
            # Load audio
            audio = AudioSegment.from_file(str(input_path))
            
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export with appropriate settings
            if target_format.lower() == 'mp3':
                # High quality MP3 settings
                audio.export(str(output_path), 
                           format='mp3', 
                           bitrate=bitrate,
                           parameters=["-q:a", "0"])  # Highest quality
            elif target_format.lower() == 'wav':
                # Lossless WAV
                audio.export(str(output_path), format='wav')
            elif target_format.lower() == 'm4a':
                # M4A files use mp4 format identifier
                audio.export(str(output_path), format='mp4')
            else:
                # Generic export
                audio.export(str(output_path), format=target_format)
            
            print(f"Converted to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error converting audio: {e}")
            return False
    
    def transcribe_audio(self, 
                        audio_path: Path, 
                        output_dir: Path,
                        model_size: str = "turbo",
                        use_chunking: bool = False,
                        chunk_duration_minutes: int = 10) -> Tuple[bool, Optional[Path]]:
        """
        Transcribe audio file using Whisper (local or OpenAI API)
        
        Args:
            audio_path: Path to audio file to transcribe
            output_dir: Directory to save transcript
            model_size: Whisper model size (tiny, base, small, medium, large, turbo)
            use_chunking: Whether to split audio into chunks for stability
            chunk_duration_minutes: Duration of each chunk in minutes
            
        Returns:
            Tuple of (success, path_to_transcript_file)
        """
        try:
            print(f"Transcribing audio: {audio_path.name}")
            
            if use_chunking:
                return self._transcribe_with_chunking(audio_path, output_dir, model_size, chunk_duration_minutes)
            elif self.use_openai_whisper:
                return self._transcribe_with_openai_api(audio_path, output_dir, model_size)
            else:
                return self._transcribe_with_local_whisper(audio_path, output_dir, model_size)
                
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return False, None
    
    def _transcribe_with_chunking(self,
                                 audio_path: Path,
                                 output_dir: Path,
                                 model_size: str,
                                 chunk_duration_minutes: int) -> Tuple[bool, Optional[Path]]:
        """
        Transcribe audio using chunking approach for memory stability
        """
        try:
            print(f"Using chunked transcription with {chunk_duration_minutes}-minute chunks...")
            
            # Split audio into chunks
            chunks = self._chunk_audio_for_transcription(audio_path, chunk_duration_minutes)
            
            if len(chunks) == 1:
                print("Audio is short enough - transcribing directly")
                if self.use_openai_whisper:
                    return self._transcribe_with_openai_api(audio_path, output_dir, model_size)
                else:
                    return self._transcribe_with_local_whisper(audio_path, output_dir, model_size)
            
            # Transcribe each chunk
            chunk_transcripts = []
            for i, chunk_path in enumerate(chunks):
                print(f"Transcribing chunk {i+1}/{len(chunks)}: {chunk_path.name}")
                
                if self.use_openai_whisper:
                    success, _ = self._transcribe_with_openai_api(chunk_path, output_dir, model_size)
                else:
                    success, _ = self._transcribe_with_local_whisper(chunk_path, output_dir, model_size)
                
                if success:
                    # Load the transcript data for merging
                    chunk_name = chunk_path.stem
                    transcript_dir = output_dir / "transcripts"
                    chunk_transcript_path = transcript_dir / f"{chunk_name}_transcript.json"
                    
                    if chunk_transcript_path.exists():
                        with open(chunk_transcript_path, 'r', encoding='utf-8') as f:
                            chunk_transcript = json.load(f)
                            chunk_transcripts.append(chunk_transcript)
                        
                        # Clean up chunk transcript file
                        chunk_transcript_path.unlink()
                        (transcript_dir / f"{chunk_name}_transcript.txt").unlink(missing_ok=True)
                    else:
                        print(f"Warning: Chunk transcript not found: {chunk_transcript_path}")
                else:
                    print(f"Failed to transcribe chunk {i+1}")
                    return False, None
            
            # Merge chunk transcripts
            if chunk_transcripts:
                return self._merge_chunked_transcripts(chunk_transcripts, audio_path, output_dir)
            else:
                print("No successful chunk transcriptions")
                return False, None
                
        except Exception as e:
            print(f"Error in chunked transcription: {e}")
            return False, None
    
    def _transcribe_with_openai_api(self, 
                                   audio_path: Path, 
                                   output_dir: Path,
                                   model_size: str = "whisper-1") -> Tuple[bool, Optional[Path]]:
        """
        Transcribe using OpenAI API (much lower memory usage)
        """
        if OpenAI is None:
            print("Error: OpenAI library not installed. Run: pip install openai")
            return False, None
        
        try:
            print(f"Using OpenAI API for transcription...")
            
            # Initialize OpenAI client (requires OPENAI_API_KEY environment variable)
            client = OpenAI()
            
            # Check file size (OpenAI has 25MB limit)
            file_size_mb = audio_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 25:
                print(f"Warning: File size ({file_size_mb:.1f}MB) exceeds OpenAI 25MB limit")
                print("Consider chunking the audio file or using local Whisper")
                return False, None
            
            # Transcribe with OpenAI API
            with open(audio_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",  # OpenAI only has one Whisper model
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"]
                )
            
            # Process response
            return self._save_openai_transcript(response, audio_path, output_dir)
            
        except Exception as e:
            print(f"Error with OpenAI API transcription: {e}")
            print("Falling back to local Whisper...")
            return self._transcribe_with_local_whisper(audio_path, output_dir, model_size)
    
    def _transcribe_with_local_whisper(self, 
                                      audio_path: Path, 
                                      output_dir: Path,
                                      model_size: str = "turbo") -> Tuple[bool, Optional[Path]]:
        """
        Transcribe using local Whisper model (higher memory usage)
        """
        print(f"Using local Whisper model: {model_size}")
        
        try:
            # Clear any existing model to free memory
            if hasattr(self, 'whisper_model') and self.whisper_model is not None:
                del self.whisper_model
                import gc
                gc.collect()
            
            # Load Whisper model with memory optimization
            print(f"Loading Whisper model '{model_size}' with memory optimization...")
            
            import torch
            # Use CPU to avoid GPU memory issues
            device = "cpu"
            
            # Load model with explicit device setting
            self.whisper_model = whisper.load_model(model_size, device=device)
            self.whisper_model.model_name = model_size
            
            # Transcribe audio with memory-friendly settings
            print("Starting transcription... (this may take a while)")
            result = self.whisper_model.transcribe(
                str(audio_path),
                language="en",  # Assuming English yoga sessions
                task="transcribe",
                verbose=False,  # Reduce output to save memory
                word_timestamps=True,  # Enable word-level timestamps
                fp16=False,  # Use FP32 for stability on CPU
                no_speech_threshold=0.6,  # More aggressive silence detection
                logprob_threshold=-1.0,  # More permissive
                compression_ratio_threshold=2.4  # More permissive
            )
            
            # Clear model after use to free memory
            del self.whisper_model
            self.whisper_model = None
            import gc
            gc.collect()
            
            return self._save_local_whisper_transcript(result, audio_path, output_dir, model_size)
            
        except Exception as e:
            print(f"Error in local Whisper transcription: {e}")
            # Clean up on error
            if hasattr(self, 'whisper_model'):
                del self.whisper_model
                self.whisper_model = None
                import gc
                gc.collect()
            return False, None
    
    def _save_openai_transcript(self, 
                               response, 
                               audio_path: Path, 
                               output_dir: Path) -> Tuple[bool, Optional[Path]]:
        """
        Save OpenAI API transcript response to files
        """
        # Create transcript output directory
        transcript_dir = output_dir / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate transcript filename
        cd_name = audio_path.stem.replace("_master", "")
        transcript_path = transcript_dir / f"{cd_name}_transcript.json"
        
        # Prepare transcript data
        transcript_data = {
            "cd_title": cd_name,
            "speaker": "Rajyogi Caruso",
            "audio_file": str(audio_path),
            "model_used": "openai-whisper-1",
            "language": response.language,
            "duration_seconds": response.duration,
            "transcript_text": response.text,
            "segments": []
        }
        
        # Process segments
        for segment in response.segments:
            segment_data = {
                "id": segment.id,
                "start_time": segment.start,
                "end_time": segment.end,
                "duration": segment.end - segment.start,
                "text": segment.text.strip(),
                "speaker": "Rajyogi Caruso",
                "confidence": getattr(segment, 'avg_logprob', 0.0)
            }
            
            # Add word-level timestamps if available
            if hasattr(segment, 'words') and segment.words:
                segment_data["words"] = [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "confidence": getattr(word, 'probability', 0.0)
                    }
                    for word in segment.words
                ]
            
            transcript_data["segments"].append(segment_data)
        
        # Save transcript to JSON file
        print(f"Saving transcript to: {transcript_path}")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        
        # Generate readable text version
        text_path = transcript_dir / f"{cd_name}_transcript.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"Yoga CD Transcript: {cd_name}\n")
            f.write(f"Speaker: Rajyogi Caruso\n")
            f.write(f"Duration: {transcript_data['duration_seconds']/60:.1f} minutes\n")
            f.write(f"Transcription Model: openai-whisper-1\n")
            f.write("="*60 + "\n\n")
            
            for segment in transcript_data["segments"]:
                start_min = int(segment["start_time"] // 60)
                start_sec = int(segment["start_time"] % 60)
                f.write(f"[{start_min:02d}:{start_sec:02d}] {segment['text']}\n")
        
        print(f"Transcription completed!")
        print(f"JSON transcript: {transcript_path}")
        print(f"Text transcript: {text_path}")
        
        return True, transcript_path
    
    def _save_local_whisper_transcript(self, 
                                      result, 
                                      audio_path: Path, 
                                      output_dir: Path,
                                      model_size: str) -> Tuple[bool, Optional[Path]]:
        """
        Save local Whisper transcript to files
        """
        # Create transcript output directory
        transcript_dir = output_dir / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate transcript filename
        cd_name = audio_path.stem.replace("_master", "")
        transcript_path = transcript_dir / f"{cd_name}_transcript.json"
        
        # Prepare transcript data with speaker information
        transcript_data = {
            "cd_title": cd_name,
            "speaker": "Rajyogi Caruso",  # Primary speaker
            "audio_file": str(audio_path),
            "model_used": model_size,
            "language": result.get("language", "en"),
            "duration_seconds": len(AudioSegment.from_file(str(audio_path))) / 1000,
            "transcript_text": result["text"],
            "segments": []
        }
        
        # Process segments with timestamps
        for segment in result["segments"]:
            segment_data = {
                "id": segment["id"],
                "start_time": segment["start"],
                "end_time": segment["end"],
                "duration": segment["end"] - segment["start"],
                "text": segment["text"].strip(),
                "speaker": "Rajyogi Caruso",  # Default speaker
                "confidence": segment.get("avg_logprob", 0.0)
            }
            
            # Add word-level timestamps if available
            if "words" in segment:
                segment_data["words"] = [
                    {
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"],
                        "confidence": word.get("probability", 0.0)
                    }
                    for word in segment["words"]
                ]
            
            transcript_data["segments"].append(segment_data)
        
        # Save transcript to JSON file
        print(f"Saving transcript to: {transcript_path}")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        
        # Generate readable text version
        text_path = transcript_dir / f"{cd_name}_transcript.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"Yoga CD Transcript: {cd_name}\n")
            f.write(f"Speaker: Rajyogi Caruso\n")
            f.write(f"Duration: {transcript_data['duration_seconds']/60:.1f} minutes\n")
            f.write(f"Transcription Model: {model_size}\n")
            f.write("="*60 + "\n\n")
            
            for segment in transcript_data["segments"]:
                start_min = int(segment["start_time"] // 60)
                start_sec = int(segment["start_time"] % 60)
                f.write(f"[{start_min:02d}:{start_sec:02d}] {segment['text']}\n")
        
        print(f"Transcription completed!")
        print(f"JSON transcript: {transcript_path}")
        print(f"Text transcript: {text_path}")
        
        return True, transcript_path

    def _chunk_audio_for_transcription(self, 
                                      audio_path: Path, 
                                      chunk_duration_minutes: int = 3) -> List[Path]:
        """
        Split large audio files into smaller chunks for more stable transcription
        
        Args:
            audio_path: Path to the audio file to chunk
            chunk_duration_minutes: Duration of each chunk in minutes
            
        Returns:
            List of paths to the audio chunks
        """
        try:
            audio = AudioSegment.from_file(str(audio_path))
            duration_ms = len(audio)
            chunk_duration_ms = chunk_duration_minutes * 60 * 1000
            
            chunks = []
            chunk_dir = audio_path.parent / f"{audio_path.stem}_chunks"
            chunk_dir.mkdir(exist_ok=True)
            
            # If file is smaller than chunk size, return original
            if duration_ms <= chunk_duration_ms:
                return [audio_path]
            
            print(f"Splitting audio into {chunk_duration_minutes}-minute chunks with optimized encoding...")
            
            for i, start_ms in enumerate(range(0, duration_ms, chunk_duration_ms)):
                end_ms = min(start_ms + chunk_duration_ms, duration_ms)
                chunk = audio[start_ms:end_ms]
                
                # Convert to mono and reduce sample rate for smaller file size
                chunk = chunk.set_channels(1)  # Mono audio
                chunk = chunk.set_frame_rate(16000)  # Lower sample rate (sufficient for speech)
                
                chunk_path = chunk_dir / f"chunk_{i:03d}.mp3"
                # Use lower bitrate and optimize for speech
                chunk.export(str(chunk_path), 
                           format='mp3', 
                           bitrate='64k',  # Much lower bitrate for speech
                           parameters=["-ac", "1", "-ar", "16000"])  # Force mono, 16kHz
                chunks.append(chunk_path)
                print(f"Created chunk {i+1}: {chunk_path.name}")
            
            return chunks
            
        except Exception as e:
            print(f"Error chunking audio: {e}")
            return [audio_path]  # Return original file as fallback
    
    def _merge_chunked_transcripts(self, 
                                  chunk_transcripts: List[dict], 
                                  original_audio_path: Path,
                                  output_dir: Path) -> Tuple[bool, Optional[Path]]:
        """
        Merge transcripts from multiple audio chunks into a single transcript
        """
        try:
            # Create combined transcript data
            cd_name = original_audio_path.stem.replace("_master", "")
            
            combined_transcript = {
                "cd_title": cd_name,
                "speaker": "Rajyogi Caruso",
                "audio_file": str(original_audio_path),
                "model_used": chunk_transcripts[0].get("model_used", "chunked"),
                "language": chunk_transcripts[0].get("language", "en"),
                "duration_seconds": sum(t.get("duration_seconds", 0) for t in chunk_transcripts),
                "transcript_text": "",
                "segments": []
            }
            
            # Merge segments with adjusted timestamps
            segment_id = 0
            time_offset = 0.0
            
            for chunk_transcript in chunk_transcripts:
                combined_transcript["transcript_text"] += chunk_transcript.get("transcript_text", "") + " "
                
                for segment in chunk_transcript.get("segments", []):
                    adjusted_segment = segment.copy()
                    adjusted_segment["id"] = segment_id
                    adjusted_segment["start_time"] += time_offset
                    adjusted_segment["end_time"] += time_offset
                    
                    # Adjust word timestamps if present
                    if "words" in adjusted_segment:
                        for word in adjusted_segment["words"]:
                            word["start"] += time_offset
                            word["end"] += time_offset
                    
                    combined_transcript["segments"].append(adjusted_segment)
                    segment_id += 1
                
                # Update time offset for next chunk
                if chunk_transcript.get("segments"):
                    last_segment = chunk_transcript["segments"][-1]
                    time_offset = last_segment.get("end_time", time_offset)
            
            # Save combined transcript
            transcript_dir = output_dir / "transcripts"
            transcript_dir.mkdir(parents=True, exist_ok=True)
            transcript_path = transcript_dir / f"{cd_name}_transcript.json"
            
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(combined_transcript, f, indent=2, ensure_ascii=False)
            
            # Generate readable text version
            text_path = transcript_dir / f"{cd_name}_transcript.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(f"Yoga CD Transcript: {cd_name}\n")
                f.write(f"Speaker: Rajyogi Caruso\n")
                f.write(f"Duration: {combined_transcript['duration_seconds']/60:.1f} minutes\n")
                f.write(f"Transcription Model: {combined_transcript['model_used']} (chunked)\n")
                f.write("="*60 + "\n\n")
                
                for segment in combined_transcript["segments"]:
                    start_min = int(segment["start_time"] // 60)
                    start_sec = int(segment["start_time"] % 60)
                    f.write(f"[{start_min:02d}:{start_sec:02d}] {segment['text']}\n")
            
            print(f"Combined transcript saved: {transcript_path}")
            print(f"Text transcript: {text_path}")
            
            return True, transcript_path
            
        except Exception as e:
            print(f"Error merging chunked transcripts: {e}")
            return False, None

    def process_single_cd(self, 
                         cd_path: Path, 
                         output_dir: Path,
                         skip_combine: bool = False,
                         skip_convert: bool = False,
                         skip_transcribe: bool = False,
                         mp3_file_path: Optional[Path] = None,
                         whisper_model: str = "turbo") -> Tuple[bool, Optional[Path]]:
        """
        Process a single CD directory
        
        Args:
            cd_path: Path to CD directory containing audio files
            output_dir: Base output directory
            skip_combine: Skip the combine step
            skip_convert: Skip the convert step
            skip_transcribe: Skip the transcribe step
            mp3_file_path: Pre-converted MP3 file path (if skip_convert is True)
            whisper_model: Whisper model size to use for transcription
            
        Returns:
            Tuple of (success, path_to_mp3_file)
        """
        cd_name = cd_path.name
        print(f"\nProcessing CD: {cd_name}")
        
        # Create output directories
        combined_dir = output_dir / "combined" / cd_name
        converted_dir = output_dir / "converted" / cd_name
        
        combined_dir.mkdir(parents=True, exist_ok=True)
        converted_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Combine audio files (unless skipped)
        master_m4a_path = combined_dir / f"{cd_name}_master.m4a"
        
        if skip_combine:
            print("Skipping combine step")
            if not master_m4a_path.exists():
                print(f"Warning: Master file {master_m4a_path} does not exist")
                return False, None
        else:
            # Find audio files
            audio_files = self.find_audio_files(cd_path)
            if not audio_files:
                print(f"No audio files found in {cd_path}")
                return False, None
            
            print(f"Found {len(audio_files)} audio files")
            
            # Check if master file exists
            if not self.check_file_exists(master_m4a_path):
                print("Skipping due to existing file")
                # Still continue if file exists
            else:
                # Combine files
                if not self.combine_audio_files(audio_files, master_m4a_path):
                    return False, None
        
        # Step 2: Convert to MP3 (unless skipped)
        mp3_output_path = converted_dir / f"{cd_name}_master.mp3"
        
        if skip_convert:
            print("Skipping convert step")
            if mp3_file_path:
                mp3_output_path = mp3_file_path
            elif not mp3_output_path.exists():
                print(f"Warning: MP3 file {mp3_output_path} does not exist and no path provided")
                return False, None
        else:
            # Check if MP3 file exists
            if not self.check_file_exists(mp3_output_path):
                print("Skipping due to existing file")
            else:
                # Convert master M4A to MP3
                if not self.convert_audio_format(master_m4a_path, mp3_output_path):
                    return False, None
        
        # Step 3: Transcribe audio (unless skipped)
        transcript_path = None
        if skip_transcribe:
            print("Skipping transcribe step")
        else:
            # Check if transcript already exists
            transcript_dir = output_dir / "transcripts"
            existing_transcript = transcript_dir / f"{cd_name}_transcript.json"
            
            if existing_transcript.exists() and not self.check_file_exists(existing_transcript):
                print("Skipping transcription due to existing file")
                transcript_path = existing_transcript
            else:
                # Transcribe the MP3 file
                success, transcript_path = self.transcribe_audio(
                    mp3_output_path, 
                    output_dir, 
                    whisper_model,
                    use_chunking=getattr(self, 'use_chunking', False),
                    chunk_duration_minutes=getattr(self, 'chunk_duration_minutes', 10)
                )
                if not success:
                    print("Warning: Transcription failed, but continuing...")
        
        print(f"Successfully processed CD: {cd_name}")
        print(f"Master M4A: {master_m4a_path}")
        print(f"Master MP3: {mp3_output_path}")
        if transcript_path:
            print(f"Transcript: {transcript_path}")
        
        return True, mp3_output_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Process yoga CD audio files: combine, convert, transcribe, and split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all CDs in a directory (combine, convert, transcribe)
  python yoga_cd_processor.py /path/to/cds --output ./processed_cds
  
  # Process a single CD with specific Whisper model
  python yoga_cd_processor.py /path/to/single_cd --output ./processed_cds --whisper-model small
  
  # Use OpenAI API for transcription (requires API key)
  python yoga_cd_processor.py /path/to/cds --output ./processed_cds --use-openai-whisper
  
  # Use chunking for long audio files to prevent segmentation faults
  python yoga_cd_processor.py /path/to/cds --output ./processed_cds --chunk-audio --chunk-duration 5
  
  # Skip combine step (files already combined)
  python yoga_cd_processor.py /path/to/cds --output ./processed_cds --skip-combine
  
  # Skip transcription (only combine and convert)
  python yoga_cd_processor.py /path/to/cds --output ./processed_cds --skip-transcribe
  
  # Skip convert step and provide MP3 file
  python yoga_cd_processor.py /path/to/cds --output ./processed_cds --skip-convert --mp3-file /path/to/file.mp3
        """
    )
    
    # Required arguments
    parser.add_argument('input_path', type=Path, 
                       help='Path to CD directory or directory containing multiple CDs')
    parser.add_argument('--output', '-o', type=Path, required=True,
                       help='Output directory for processed files')
    
    # Step control arguments
    parser.add_argument('--skip-combine', action='store_true',
                       help='Skip the combine step (files already combined)')
    parser.add_argument('--skip-convert', action='store_true', 
                       help='Skip the convert step (files already converted)')
    parser.add_argument('--skip-transcribe', action='store_true',
                       help='Skip the transcribe step')
    parser.add_argument('--skip-split', action='store_true',
                       help='Skip the split step')
    
    # File path arguments
    parser.add_argument('--mp3-file', type=Path,
                       help='Path to pre-converted MP3 file (use with --skip-convert)')
    
    # Processing options
    parser.add_argument('--crossfade', type=int, default=500,
                       help='Crossfade duration in milliseconds (default: 500)')
    parser.add_argument('--bitrate', default='192k',
                       help='MP3 bitrate (default: 192k)')
    parser.add_argument('--whisper-model', default='turbo',
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'turbo'],
                       help='Whisper model size (default: turbo)')
    parser.add_argument('--use-openai-whisper', action='store_true',
                       help='Use OpenAI API for Whisper instead of local model')
    parser.add_argument('--chunk-audio', action='store_true',
                       help='Split long audio into chunks for more stable transcription')
    parser.add_argument('--chunk-duration', type=int, default=3,
                       help='Duration of audio chunks in minutes (default: 3)')
    parser.add_argument('--openai-api-key', 
                       help='OpenAI API key (or set OPENAI_API_KEY environment variable)')
    
    args = parser.parse_args()
    
    # Validation
    if not args.input_path.exists():
        print(f"Error: Input path {args.input_path} does not exist")
        return 1
    
    if args.skip_convert and not args.mp3_file:
        print("Error: --mp3-file is required when using --skip-convert")
        return 1
    
    if args.mp3_file and not args.mp3_file.exists():
        print(f"Error: MP3 file {args.mp3_file} does not exist")
        return 1
    
    # Set up OpenAI API key if using OpenAI Whisper
    if args.use_openai_whisper:
        import os
        if args.openai_api_key:
            os.environ['OPENAI_API_KEY'] = args.openai_api_key
        elif not os.environ.get('OPENAI_API_KEY'):
            print("Error: OpenAI API key required for --use-openai-whisper")
            print("Either set OPENAI_API_KEY environment variable or use --openai-api-key")
            return 1
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = YogaCDProcessor(
        use_openai_whisper=args.use_openai_whisper,
        save_debug_files=True
    )
    
    # Set chunking options
    processor.use_chunking = args.chunk_audio
    processor.chunk_duration_minutes = args.chunk_duration
    
    # Determine if input is a single CD or directory of CDs
    audio_files = processor.find_audio_files(args.input_path)
    
    if audio_files:
        # Single CD directory
        print("Processing single CD directory")
        success, mp3_path = processor.process_single_cd(
            args.input_path,
            args.output,
            skip_combine=args.skip_combine,
            skip_convert=args.skip_convert,
            skip_transcribe=args.skip_transcribe,
            mp3_file_path=args.mp3_file,
            whisper_model=args.whisper_model
        )
        
        if not success:
            print("Failed to process CD")
            return 1
    else:
        # Directory containing multiple CDs
        print("Processing directory containing multiple CDs")
        cd_directories = [d for d in args.input_path.iterdir() if d.is_dir()]
        
        if not cd_directories:
            print(f"No subdirectories found in {args.input_path}")
            return 1
        
        successful_cds = []
        failed_cds = []
        
        for cd_dir in cd_directories:
            print(f"\n{'='*60}")
            try:
                success, mp3_path = processor.process_single_cd(
                    cd_dir,
                    args.output,
                    skip_combine=args.skip_combine,
                    skip_convert=args.skip_convert,
                    skip_transcribe=args.skip_transcribe,
                    mp3_file_path=args.mp3_file if len(cd_directories) == 1 else None,
                    whisper_model=args.whisper_model
                )
                
                if success:
                    successful_cds.append(cd_dir.name)
                else:
                    failed_cds.append(cd_dir.name)
            except Exception as e:
                print(f"Error processing {cd_dir.name}: {e}")
                failed_cds.append(cd_dir.name)
        
        # Print summary
        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully processed: {len(successful_cds)} CDs")
        for cd_name in successful_cds:
            print(f"  ✓ {cd_name}")
        
        if failed_cds:
            print(f"\nFailed to process: {len(failed_cds)} CDs")
            for cd_name in failed_cds:
                print(f"  ✗ {cd_name}")
    
    print("\nPhase 1 & 2 (Combine, Convert & Transcribe) completed!")
    print("\nNext steps:")
    print("- Review transcripts in the transcripts/ directory")
    print("- Run with Phase 3 options for AI-assisted splitting")
    print("- Use transcript data for intelligent audio segmentation")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
