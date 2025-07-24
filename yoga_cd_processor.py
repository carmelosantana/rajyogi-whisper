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
            combined_audio.export(str(output_path), format=output_path.suffix[1:])
            
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
            else:
                # Generic export
                audio.export(str(output_path), format=target_format)
            
            print(f"Converted to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error converting audio: {e}")
            return False
    
    def process_single_cd(self, 
                         cd_path: Path, 
                         output_dir: Path,
                         skip_combine: bool = False,
                         skip_convert: bool = False,
                         mp3_file_path: Optional[Path] = None) -> Tuple[bool, Optional[Path]]:
        """
        Process a single CD directory
        
        Args:
            cd_path: Path to CD directory containing audio files
            output_dir: Base output directory
            skip_combine: Skip the combine step
            skip_convert: Skip the convert step
            mp3_file_path: Pre-converted MP3 file path (if skip_convert is True)
            
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
        
        print(f"Successfully processed CD: {cd_name}")
        print(f"Master M4A: {master_m4a_path}")
        print(f"Master MP3: {mp3_output_path}")
        
        return True, mp3_output_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Process yoga CD audio files: combine, convert, transcribe, and split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all CDs in a directory
  python yoga_cd_processor.py /path/to/cds --output ./processed_cds
  
  # Process a single CD
  python yoga_cd_processor.py /path/to/single_cd --output ./processed_cds
  
  # Skip combine step (files already combined)
  python yoga_cd_processor.py /path/to/cds --output ./processed_cds --skip-combine
  
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
    parser.add_argument('--use-openai-whisper', action='store_true',
                       help='Use OpenAI API for Whisper instead of local model')
    
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
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = YogaCDProcessor(
        use_openai_whisper=args.use_openai_whisper,
        save_debug_files=True
    )
    
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
            mp3_file_path=args.mp3_file
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
                    mp3_file_path=args.mp3_file if len(cd_directories) == 1 else None
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
    
    print("\nPhase 1 (Combine & Convert) completed!")
    print("\nNext steps:")
    print("- Run with transcription options for Phase 2")
    print("- Use AI-assisted splitting for Phase 3")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
