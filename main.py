import time
import os
import argparse
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import ffmpeg
import pysrt
import logging
import openai
import sentry_sdk
from typing import List, Tuple
import json
from datetime import datetime
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Sentry
sentry_dsn = os.getenv('SENTRY_DSN')
if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        traces_sample_rate=1.0,
    )
    logger.info("Sentry monitoring initialized")
else:
    logger.warning("SENTRY_DSN not set. Error monitoring is disabled.")

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

class SubtitleTranslator:
    def __init__(self):
        self.context_window = []
        self.max_context_items = 5

    def update_context(self, text: str):
        self.context_window.append(text)
        if len(self.context_window) > self.max_context_items:
            self.context_window.pop(0)

    def create_prompt(self, text: str) -> str:
        context = "\n".join(self.context_window)
        return f"""Translate the following subtitle to Brazilian Portuguese. Maintain the exact same tone and style.
Previous context:
{context}

Subtitle to translate:
{text}

Rules:
- Keep informal language if present
- Maintain any special characters or formatting
- Match the speaking style of previous translations
- Keep the same level of formality as the context
"""

    def strip_formatting_tags(self, text: str) -> str:
        """Remove formatting tags and return clean text content."""
        import re
        # Remove all HTML-like tags
        clean_text = re.sub(r'<[^>]+>', '', text)
        return clean_text.strip()

    async def translate_text(self, text: str) -> str:
        try:
            # Check for empty content, even with formatting tags
            clean_text = self.strip_formatting_tags(text)
            if not clean_text:
                logger.debug(f"Empty subtitle detected (with formatting): {text}")
                return text

            client = openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional subtitle translator. Translate accurately while maintaining the original tone and context."},
                    {"role": "user", "content": self.create_prompt(text)}
                ],
                temperature=0.3,
                max_tokens=150
            )
            translated_text = response.choices[0].message.content.strip()

            # If the API returns an error message about empty subtitles, return original
            if "please provide the text" in translated_text.lower():
                return text

            self.update_context(translated_text)
            return translated_text
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text

class TranslationTracker:
    def __init__(self, directory):
        self.directory = directory
        self.tracker_file = os.path.join(directory, '.subtitle_tracker.json')
        self.load_tracker()

    def load_tracker(self):
        try:
            if os.path.exists(self.tracker_file):
                with open(self.tracker_file, 'r', encoding='utf-8') as f:
                    self.tracked_files = json.load(f)
            else:
                self.tracked_files = {}
        except Exception as e:
            logger.error(f"Error loading tracker file: {str(e)}")
            self.tracked_files = {}

    def save_tracker(self):
        try:
            with open(self.tracker_file, 'w', encoding='utf-8') as f:
                json.dump(self.tracked_files, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving tracker file: {str(e)}")

    def add_translated_file(self, file_path: str, subtitle_count: int):
        try:
            relative_path = os.path.relpath(file_path, self.directory)
            self.tracked_files[relative_path] = {
                "translated_at": datetime.now().isoformat(),
                "subtitle_count": subtitle_count,
                "file_size": os.path.getsize(file_path)
            }
            self.save_tracker()
        except Exception as e:
            logger.error(f"Error tracking translated file: {str(e)}")

    def is_file_translated(self, file_path: str) -> bool:
        try:
            relative_path = os.path.relpath(file_path, self.directory)
            if relative_path in self.tracked_files:
                current_size = os.path.getsize(file_path)
                return current_size == self.tracked_files[relative_path]["file_size"]
            return False
        except Exception as e:
            logger.error(f"Error checking file translation status: {str(e)}")
            return False

class VideoHandler(FileSystemEventHandler):
    def __init__(self, watch_directory):
        self.watch_directory = watch_directory
        self.translator = SubtitleTranslator()
        self.tracker = TranslationTracker(watch_directory)

    async def process_video(self, video_path):
        try:
            # Get video information using ffmpeg
            probe = ffmpeg.probe(video_path)

            # Check for subtitle streams
            subtitle_streams = [stream for stream in probe['streams']
                              if stream['codec_type'] == 'subtitle']

            if not subtitle_streams:
                logger.info(f"No subtitles found in {video_path}")
                return

            # Check for Portuguese subtitles first
            for stream in subtitle_streams:
                if 'tags' in stream and 'language' in stream['tags']:
                    if stream['tags']['language'].lower() in ['por', 'pt', 'pt-br']:
                        logger.info(f"Portuguese subtitles found in {video_path}, skipping processing")
                        return

            # Extract English subtitles
            for stream in subtitle_streams:
                if 'tags' in stream and 'language' in stream['tags']:
                    if stream['tags']['language'].lower() in ['eng', 'en']:
                        stream_index = stream['index']
                        output_path = os.path.splitext(video_path)[0] + '.srt'

                        # Extract subtitle using ffmpeg
                        (
                            ffmpeg
                            .input(video_path)
                            .output(output_path, map=f'0:{stream_index}')
                            .overwrite_output()
                            .run(capture_stdout=True, capture_stderr=True)
                        )

                        logger.info(f"Successfully extracted English subtitles to {output_path}")

                        # Translate the subtitles
                        await self.translate_subtitles(output_path)
                        return

            logger.info(f"No English subtitles found in {video_path}")

        except Exception as e:
            logger.error(f"Error processing {video_path}: {str(e)}")

    def on_created(self, event):
        if event.is_directory:
            return

        if event.src_path.lower().endswith(('.mkv', '.mp4')):
            if not self.tracker.is_file_translated(event.src_path):
                logger.info(f"New video file detected: {event.src_path}")
                # Create new event loop for async operation
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.process_video(event.src_path))
                loop.close()
            else:
                logger.info(f"File already translated: {event.src_path}")

    async def translate_subtitles(self, srt_path: str):
        try:
            # Load the subtitle file
            subs = pysrt.open(srt_path)
            total_subs = len(subs)

            logger.info(f"Starting translation of {total_subs} subtitles")

            # Translate each subtitle while maintaining format
            for i, sub in enumerate(subs):
                logger.info(f"Translating subtitle {i+1}/{total_subs}")
                translated_text = await self.translator.translate_text(sub.text)
                sub.text = translated_text

            # Save the translated subtitles
            translated_path = os.path.splitext(srt_path)[0] + '.translated.srt'
            subs.save(translated_path, encoding='utf-8')
            logger.info(f"Translation completed: {translated_path}")

            # Remove the original subtitle file
            os.remove(srt_path)
            # Rename the translated file to the original name
            os.rename(translated_path, srt_path)

            # Track the successful translation
            self.tracker.add_translated_file(srt_path, total_subs)

        except Exception as e:
            logger.error(f"Error translating subtitles: {str(e)}")

async def process_single_subtitle(subtitle_path: str):
    """Process a single subtitle file for testing purposes."""
    try:
        translator = SubtitleTranslator()
        await VideoHandler('.').translate_subtitles(subtitle_path)
    except Exception as e:
        logger.error(f"Error processing subtitle file: {str(e)}")

async def process_existing_files(directory: str):
    """Scan directory for existing video files and process untranslated ones."""
    logger.info(f"Scanning directory for existing files: {directory}")
    handler = VideoHandler(directory)

    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if not os.path.isfile(file_path):
            continue

        if file_path.lower().endswith(('.mkv', '.mp4')):
            if not handler.tracker.is_file_translated(file_path):
                logger.info(f"Found untranslated video file: {file_path}")
                await handler.process_video(file_path)
            else:
                logger.debug(f"Skipping already translated file: {file_path}")

def validate_watch_directories(directories: List[str]) -> List[str]:
    """
    Validate watch directories and return only valid ones.
    Raises exception if no valid directories are found.
    """
    if not directories:
        raise ValueError("No watch directories specified. Use --watch-dirs or WATCH_DIRECTORY environment variable.")

    # Split comma-separated directories if coming from environment variable
    if isinstance(directories, str):
        directories = [d.strip() for d in directories.split(',')]
    elif len(directories) == 1 and isinstance(directories[0], str):
        directories = [d.strip() for d in directories[0].split(',')]

    valid_directories = []
    for directory in directories:
        if not os.path.exists(directory):
            logger.error(f"Watch directory does not exist: {directory}")
            continue
        if not os.path.isdir(directory):
            logger.error(f"Watch path is not a directory: {directory}")
            continue
        if not os.access(directory, os.R_OK | os.W_OK):
            logger.error(f"Insufficient permissions for directory: {directory}")
            continue

        valid_directories.append(directory)

    if not valid_directories:
        raise ValueError("No valid watch directories found. Please check paths and permissions.")

    return valid_directories

def main():
    # Start profiling
    sentry_sdk.profiler.start_profiler()

    try:
        # Ensure OpenAI API key is set
        if not os.getenv('OPENAI_API_KEY'):
            logger.error("OPENAI_API_KEY environment variable is not set")
            return

        # Create argument parser
        parser = argparse.ArgumentParser(description='Subtitle extraction and translation service')
        parser.add_argument('--subtitle', '-s', help='Path to a single subtitle file for testing')
        parser.add_argument('--watch-dirs', '-w', nargs='+', help='Directories to watch for new video files')
        args = parser.parse_args()

        # If subtitle file is provided, process it and exit
        if args.subtitle:
            if not os.path.exists(args.subtitle):
                logger.error(f"Subtitle file not found: {args.subtitle}")
                return

            logger.info(f"Processing single subtitle file: {args.subtitle}")
            import asyncio
            asyncio.run(process_single_subtitle(args.subtitle))
            return

        try:
            # Get directories from CLI args or environment variable
            watch_dirs = args.watch_dirs or os.getenv('WATCH_DIRECTORY')
            watch_directories = validate_watch_directories(watch_dirs)
        except ValueError as e:
            logger.error(str(e))
            return

        # Process existing files in all directories first
        import asyncio
        for directory in watch_directories:
            asyncio.run(process_existing_files(directory))

        # Create observers for each directory
        observers = []
        for watch_directory in watch_directories:
            logger.info(f"Setting up observer for directory: {watch_directory}")
            event_handler = VideoHandler(watch_directory)
            observer = Observer()
            observer.schedule(event_handler, watch_directory, recursive=False)
            observer.start()
            observers.append(observer)

        logger.info(f"Started watching {len(observers)} directories")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping the service...")
            for observer in observers:
                observer.stop()

        # Join all observers
        for observer in observers:
            observer.join()

    except Exception as e:
        # Capture any unhandled exceptions
        sentry_sdk.capture_exception(e)
        raise
    finally:
        # Stop profiling before exit
        sentry_sdk.profiler.stop_profiler()

if __name__ == "__main__":
    main()