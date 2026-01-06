import os
import re
import json
import time
import logging
from typing import List, Dict, Union, Tuple, Generator
from pydub import AudioSegment
from pydub.generators import Sine
import math
import static_ffmpeg
static_ffmpeg.add_paths()

# Optional imports for AI
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Optional imports for noise reduction
try:
    import noisereduce as nr
    import numpy as np
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False

# Optional imports for Gemini AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Default Gemini API key (can be overridden via environment variable)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCa9rPjlQJL7prZ1RilLwWcf6IJPMc881Q")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioRedactor:
    def __init__(self, use_mock_transcription: bool = True, model_size: str = "base", gemini_api_key: str = None):
        self.use_mock_transcription = use_mock_transcription
        self.model = None
        self.gemini_model = None
        
        if not self.use_mock_transcription:
            try:
                import whisper
                import torch
                logger.info(f"Loading Whisper model: {model_size}...")
                self.model = whisper.load_model(model_size)
                logger.info("Whisper model loaded successfully.")
            except ImportError:
                logger.error("Whisper or torch not installed. Falling back to mock transcription.")
                self.use_mock_transcription = True
        
        # Load NER Model
        self.ner_pipeline = None
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info("Loading BERT NER model...")
                self.ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
                logger.info("BERT NER model loaded.")
            except Exception as e:
                logger.error(f"Failed to load NER model: {e}")
        else:
            logger.warning("Transformers library not found. AI analysis will be disabled.")
        
        # Initialize Gemini
        if GEMINI_AVAILABLE:
            try:
                api_key = gemini_api_key or GEMINI_API_KEY
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
                logger.info("Gemini AI model initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self.gemini_model = None
        else:
            logger.warning("Gemini AI not available. Install google-generativeai package.")

    def load_audio(self, file_path: str) -> AudioSegment:
        """Loads audio file using pydub."""
        logger.info(f"Loading audio from {file_path}")
        return AudioSegment.from_file(file_path)

    def denoise_audio(self, audio: AudioSegment, prop_decrease: float = 0.8) -> AudioSegment:
        """
        Applies noise reduction to an AudioSegment.
        
        Args:
            audio: The input AudioSegment to denoise
            prop_decrease: How much to reduce noise (0.0 to 1.0, default 0.8)
                          Higher values = more aggressive noise reduction
        
        Returns:
            Denoised AudioSegment
        """
        if not NOISEREDUCE_AVAILABLE:
            logger.warning("noisereduce library not available. Skipping noise reduction.")
            return audio
        
        logger.info(f"Applying noise reduction (intensity: {prop_decrease})...")
        
        # Convert AudioSegment to numpy array
        samples = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate
        
        # Handle stereo audio
        if audio.channels == 2:
            # Reshape to (samples, channels)
            samples = samples.reshape((-1, 2))
            # Process each channel separately
            reduced_left = nr.reduce_noise(
                y=samples[:, 0].astype(np.float32), 
                sr=sample_rate,
                prop_decrease=prop_decrease
            )
            reduced_right = nr.reduce_noise(
                y=samples[:, 1].astype(np.float32), 
                sr=sample_rate,
                prop_decrease=prop_decrease
            )
            # Recombine channels
            reduced = np.column_stack((reduced_left, reduced_right)).flatten()
        else:
            # Mono audio
            reduced = nr.reduce_noise(
                y=samples.astype(np.float32), 
                sr=sample_rate,
                prop_decrease=prop_decrease
            )
        
        # Convert back to int16 for AudioSegment
        reduced = np.clip(reduced, -32768, 32767).astype(np.int16)
        
        # Create new AudioSegment from denoised samples
        denoised_audio = AudioSegment(
            reduced.tobytes(),
            frame_rate=sample_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )
        
        logger.info("Noise reduction complete.")
        return denoised_audio

    def _mock_transcribe(self) -> Dict:
        """Returns mock transcription data for testing."""
        # Simulated response format from Whisper with word timestamps
        return {
            "text": "Hello, my email is john.doe@example.com and I want to buy apple stock at $150.25 on Tuesday.",
            "segments": [
                {
                    "words": [
                        {"word": "Hello,", "start": 0.5, "end": 1.0},
                        {"word": "my", "start": 1.0, "end": 1.2},
                        {"word": "email", "start": 1.2, "end": 1.7},
                        {"word": "is", "start": 1.7, "end": 1.9},
                        {"word": "john.doe@example.com", "start": 1.9, "end": 4.0},
                        {"word": "and", "start": 4.0, "end": 4.2},
                        {"word": "I", "start": 4.2, "end": 4.3},
                        {"word": "want", "start": 4.3, "end": 4.6},
                        {"word": "to", "start": 4.6, "end": 4.7},
                        {"word": "buy", "start": 4.7, "end": 5.0},
                        {"word": "apple", "start": 5.0, "end": 5.5},
                        {"word": "stock", "start": 5.5, "end": 6.0},
                        {"word": "at", "start": 6.0, "end": 6.2},
                        {"word": "$150.25", "start": 6.2, "end": 7.5},
                        {"word": "on", "start": 7.5, "end": 7.7},
                        {"word": "Tuesday.", "start": 7.7, "end": 8.5},
                    ]
                }
            ]
        }

    def transcribe(self, audio_path: str) -> Dict:
        """Transcribes audio and returns text with timestamps."""
        if self.use_mock_transcription:
            logger.info("Using Mock Transcription.")
            return self._mock_transcribe()
        
        logger.info("Starting Whisper transcription...")
        # Whisper transcribe with word_timestamps=True is crucial
        result = self.model.transcribe(audio_path, word_timestamps=True)
        return result

    def analyze_text(self, transcript: Dict, entity_list: List[str] = []) -> List[Dict]:
        """
        Analyzes transcript for PII, prices, and entity list matches.
        Returns a list of dicts: {'start': float, 'end': float, 'label': str, 'text': str}
        """
        detections = []
        
        # Regex Patterns
        patterns = {
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            # Match prices: $1234.56, 1234.56, $1,234.56, 1,234.56, 1339.92 
            # Allows any digits before decimal, with optional $ and commas
            "PRICE": r'(?:\$\s?)?\d+(?:,\d{3})*\.\d{2}\b|\$\d+(?:\.\d{2})?\b',
            "PHONE": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            # SSN pattern
            "SSN": r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
        }
        
        # Flatten words from segments
        words = []
        if "segments" in transcript:
            for segment in transcript["segments"]:
                if "words" in segment:
                    words.extend(segment["words"])
        
        # 1. Check for specific words/entities from the Custom List
        # Normalize entity list to lowercase for case-insensitive matching
        normalized_entities = [e.lower() for e in entity_list]
        
        # Track which words we've already processed as part of merged tokens
        processed_indices = set()
        
        for i, word_obj in enumerate(words):
            if i in processed_indices:
                continue
                
            word_text = word_obj["word"].strip(".,!?")
            word_text_lower = word_text.lower()
            word_stripped = word_obj["word"].strip()
            
            # Check for split prices: current word is digits, next word starts with decimal
            # e.g., "1339" followed by ".92"
            if i + 1 < len(words):
                next_word = words[i + 1]["word"].strip()
                current_stripped = word_stripped.lstrip('$').replace(',', '')
                
                # Check if current is digits (possibly with $ prefix) and next starts with decimal
                if (current_stripped.isdigit() or 
                    (word_stripped.startswith('$') and current_stripped.isdigit())):
                    if re.match(r'^\.\d{2}\b', next_word):
                        # This is a split price! Merge them
                        merged_price = word_stripped + next_word
                        detections.append({
                            "start": word_obj["start"],
                            "end": words[i + 1]["end"],
                            "label": "PRICE",
                            "text": merged_price.strip(".,!?")
                        })
                        processed_indices.add(i)
                        processed_indices.add(i + 1)
                        logger.info(f"Detected split price: '{merged_price}' @ {word_obj['start']:.2f}s")
                        continue
            
            # Check PII Patterns on the word
            # Use re.search instead of re.match to find patterns anywhere in the word
            for p_name, p_regex in patterns.items():
                if re.search(p_regex, word_stripped):
                     detections.append({
                         "start": word_obj["start"], 
                         "end": word_obj["end"],
                         "label": p_name,
                         "text": word_text
                     })
            
            # Check Entity List
            if word_text_lower in normalized_entities:
                detections.append({
                    "start": word_obj["start"], 
                    "end": word_obj["end"],
                    "label": "CUSTOM",
                    "text": word_text
                })

        return detections

    def analyze_text_ai(self, transcript: Dict, entity_list: List[str] = []) -> List[Dict]:
        """
        Uses BERT NER to identify PII (PER, LOC, ORG) PLUS Regex/Entity List.
        """
        # 1. Run basic analysis first (Regex + List)
        detections = self.analyze_text(transcript, entity_list)
        
        if not self.ner_pipeline:
            logger.warning("NER pipeline not available. Skipping AI analysis.")
            return detections

        # 2. Prepare text for NER
        full_text = transcript.get("text", "")
        if not full_text:
            return detections

        # 3. Run NER
        logger.info("Running AI PII detection...")
        ner_results = self.ner_pipeline(full_text)
        # Example result: [{'entity_group': 'PER', 'score': 0.99, 'word': 'John Doe', 'start': 10, 'end': 18}]
        
        # 4. Map NER character offsets to Whisper word timestamps
        # ... (Same mapping logic as before) ...
        
        # Flatten words
        words = []
        if "segments" in transcript:
            for segment in transcript["segments"]:
                if "words" in segment:
                    words.extend(segment["words"])
        
        current_pos = 0
        word_char_ranges = [] 
        
        for w in words:
            w_text = w["word"].strip()
            while current_pos < len(full_text) and full_text[current_pos].isspace():
                current_pos += 1
            if current_pos >= len(full_text):
                break
            end_pos = current_pos + len(w_text)
            word_char_ranges.append({
                "char_start": current_pos,
                "char_end": end_pos,
                "word_obj": w
            })
            current_pos = end_pos

        # Now check overlaps
        for entity in ner_results:
            if entity['entity_group'] in ['PER', 'ORG', 'LOC']: 
                e_start = entity['start']
                e_end = entity['end']
                
                for w_map in word_char_ranges:
                    if max(e_start, w_map['char_start']) < min(e_end, w_map['char_end']):
                        detections.append({
                            "start": w_map['word_obj']['start'], 
                            "end": w_map['word_obj']['end'],
                            "label": entity['entity_group'],
                            "text": w_map['word_obj']['word'] # Use the word from Whisper
                        })

        return detections
    
    def analyze_with_gemini(self, transcript: Dict, entity_list: List[str] = []) -> List[Dict]:
        """
        Uses Google Gemini AI to analyze transcript and identify PII.
        Returns list of detections with timestamps.
        """
        if not self.gemini_model:
            logger.warning("Gemini model not available. Falling back to regex analysis.")
            return self.analyze_text(transcript, entity_list)
        
        full_text = transcript.get("text", "")
        if not full_text:
            return []
        
        # Build the prompt for Gemini
        custom_entities_prompt = ""
        if entity_list:
            custom_entities_prompt = f"\n- Also redact these specific words/phrases: {', '.join(entity_list)}"
        
        prompt = f"""Analyze the following audio transcript and identify ALL sensitive information that should be redacted.

TRANSCRIPT:
{full_text}

IDENTIFY AND LIST ALL OF THE FOLLOWING:
- Prices and monetary amounts (e.g., $150.25, 1339.92, "fifteen hundred dollars")
- Email addresses
- Phone numbers
- Social Security Numbers (SSN)
- Names of people (first names, last names, full names)
- Physical addresses
- Credit card numbers
- Bank account numbers
- Dates of birth
- Any other personally identifiable information (PII){custom_entities_prompt}

IMPORTANT: Return your response as a JSON array with this exact format:
[
  {{"text": "the exact text to redact", "label": "CATEGORY"}},
  {{"text": "another item", "label": "CATEGORY"}}
]

Categories should be: PRICE, EMAIL, PHONE, SSN, PERSON, ADDRESS, CARD, DOB, CUSTOM, or OTHER

Return ONLY the JSON array, no other text. If no sensitive items found, return empty array: []
"""

        try:
            logger.info("Sending transcript to Gemini for PII analysis...")
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up response - extract JSON if wrapped in code blocks
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])  # Remove first and last lines (code block markers)
            
            # Parse JSON response
            try:
                gemini_detections = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini response as JSON: {e}")
                logger.debug(f"Raw response: {response_text}")
                return self.analyze_text(transcript, entity_list)
            
            logger.info(f"Gemini identified {len(gemini_detections)} potential PII items.")
            
            # Map Gemini detections back to word timestamps
            detections = []
            
            # Flatten words from segments
            words = []
            if "segments" in transcript:
                for segment in transcript["segments"]:
                    if "words" in segment:
                        words.extend(segment["words"])
            
            # For each Gemini detection, find matching words in transcript
            for gemini_item in gemini_detections:
                item_text = gemini_item.get("text", "").strip()
                item_label = gemini_item.get("label", "OTHER")
                
                if not item_text:
                    continue
                
                # Search for this text in the words
                # Handle multi-word matches and partial matches
                item_words = item_text.split()
                found_match = False  # Track if we found a match for this item
                
                for i, word_obj in enumerate(words):
                    if found_match:
                        break  # Stop searching once we found a match
                    
                    word_clean = word_obj["word"].strip().lower()
                    
                    # Check for exact single word match
                    if len(item_words) == 1:
                        # Check if word matches (case insensitive, handle punctuation)
                        if item_text.lower() in word_clean or word_clean.replace(".", "").replace(",", "") in item_text.lower():
                            detections.append({
                                "start": word_obj["start"],
                                "end": word_obj["end"],
                                "label": item_label,
                                "text": word_obj["word"].strip()
                            })
                            found_match = True  # Stop after first match
                    else:
                        # Multi-word match - check if this is the start of the phrase
                        if item_words[0].lower() in word_clean:
                            # Try to match the full phrase
                            match_end = i
                            matched = True
                            for j, target_word in enumerate(item_words):
                                if i + j >= len(words):
                                    matched = False
                                    break
                                check_word = words[i + j]["word"].strip().lower()
                                if target_word.lower() not in check_word and check_word not in target_word.lower():
                                    matched = False
                                    break
                                match_end = i + j
                            
                            if matched:
                                detections.append({
                                    "start": word_obj["start"],
                                    "end": words[match_end]["end"],
                                    "label": item_label,
                                    "text": item_text
                                })
                                found_match = True  # Stop after first match
            
            # Deduplicate detections (same start/end/label)
            seen = set()
            unique_detections = []
            for d in detections:
                key = (round(d["start"], 2), round(d["end"], 2), d["label"])
                if key not in seen:
                    seen.add(key)
                    unique_detections.append(d)
            
            logger.info(f"Mapped {len(unique_detections)} unique detections to word timestamps.")
            return unique_detections
            
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            logger.info("Falling back to regex analysis.")
            return self.analyze_text(transcript, entity_list)
    
    def generate_summary_with_gemini(self, detections: List[Dict], transcript_text: str, redaction_mode: str = "beep") -> str:
        """
        Uses Gemini to generate a human-readable summary of the redaction results.
        
        Args:
            detections: List of detected PII items with labels, text, and timestamps
            transcript_text: The full transcript text
            redaction_mode: How PII was redacted ('beep' or 'silence')
        
        Returns:
            A human-readable summary string
        """
        if not self.gemini_model or not detections:
            # Fallback to simple summary if Gemini not available
            return self._generate_simple_summary(detections, redaction_mode)
        
        try:
            # Group detections by category
            categories = {}
            for d in detections:
                label = d.get('label', 'UNKNOWN')
                if label not in categories:
                    categories[label] = []
                categories[label].append(d)
            
            # Format detection info for prompt
            detection_details = []
            for label, items in categories.items():
                texts = [item.get('text', 'unknown') for item in items]
                detection_details.append(f"- {label}: {len(items)} item(s) - {', '.join(texts[:3])}" + 
                                        ("..." if len(texts) > 3 else ""))
            
            prompt = f"""You are generating a user-friendly redaction summary for an audio processing app.

The following sensitive information was detected and redacted from an audio file:

{chr(10).join(detection_details)}

Total items redacted: {len(detections)}
Redaction method: Audio {redaction_mode}s were inserted to mask sensitive content.

Generate a brief, friendly summary (2-4 sentences) for the user explaining:
1. What types of sensitive information were found
2. How many items were redacted
3. A reassuring note that their audio is now safe to share

Keep it concise and professional. Do NOT include specific sensitive values in your response.
Write in a friendly, conversational tone. Use bullet points if helpful."""

            logger.info("Generating human-readable summary with Gemini...")
            response = self.gemini_model.generate_content(prompt)
            summary = response.text.strip()
            
            logger.info("Summary generated successfully.")
            return summary
            
        except Exception as e:
            logger.error(f"Gemini summary generation failed: {e}")
            return self._generate_simple_summary(detections, redaction_mode)
    
    def _generate_simple_summary(self, detections: List[Dict], redaction_mode: str = "beep") -> str:
        """Generates a simple summary without Gemini."""
        if not detections:
            return "No sensitive information was detected in this audio."
        
        # Group by category
        categories = {}
        for d in detections:
            label = d.get('label', 'OTHER')
            categories[label] = categories.get(label, 0) + 1
        
        parts = []
        for label, count in sorted(categories.items(), key=lambda x: -x[1]):
            item_word = "item" if count == 1 else "items"
            parts.append(f"{count} {label.lower()} {item_word}")
        
        summary = f"Detected and redacted {len(detections)} sensitive items: {', '.join(parts)}. "
        summary += f"All sensitive content has been replaced with {redaction_mode}s."
        return summary

    def _merge_intervals(self, intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Merges overlapping intervals."""
        if not intervals:
            return []
            
        # Sort by start time
        intervals.sort(key=lambda x: x[0])
        
        merged = [intervals[0]]
        for current in intervals[1:]:
            previous = merged[-1]
            if current[0] <= previous[1]: # Overlap
                merged[-1] = (previous[0], max(previous[1], current[1]))
            else:
                merged.append(current)
        return merged

    def redact_audio(self, audio: AudioSegment, detections: List[Dict], mode: str = "beep") -> AudioSegment:
        """
        Redacts specified time intervals in the audio.
        mode: 'beep' or 'silence'
        """
        # Extract intervals for merging
        raw_intervals = [(d['start'], d['end']) for d in detections]
        merged_intervals = self._merge_intervals(raw_intervals)
        
        redacted_audio = audio
        
        for start, end in merged_intervals:
            # Pydub works in milliseconds
            start_ms = start * 1000
            end_ms = end * 1000
            duration_ms = end_ms - start_ms
            
            if duration_ms <= 0:
                continue
                
            if mode == "beep":
                # Generate 1000Hz sine wave
                tone = Sine(1000).to_audio_segment(duration=duration_ms).apply_gain(-10) 
                redacted_audio = redacted_audio.overlay(tone, position=start_ms)
            else:
                # Silence
                silence = AudioSegment.silent(duration=duration_ms)
                redacted_audio = redacted_audio.overlay(silence, position=start_ms, gain_during_overlay=-120) 
                
        return redacted_audio, len(merged_intervals)

    def process_pipeline_stream(self, input_path: str, output_path: str, entity_list: List[str] = [], mode: str = "beep", denoise: bool = False, denoise_intensity: float = 0.8, detection_mode: str = "gemini") -> Generator[Dict, None, None]:
        """
        Runs the full redaction pipeline, YIELDING status updates.
        Final yield is the result report.
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output redacted audio file
            entity_list: Custom entities/words to redact
            mode: 'beep' or 'silence'
            denoise: Whether to apply noise reduction before transcription
            denoise_intensity: Noise reduction strength (0.0 to 1.0)
            detection_mode: 'regex', 'gemini', or 'ai' (BERT NER)
        """
        import tempfile
        import os as temp_os
        
        start_time = time.time()
        temp_denoised_path = None
        total_steps = 6 if denoise else 5
        current_step = 0
        
        # 1. Load
        current_step += 1
        yield {"status": "progress", "message": "Loading audio file...", "step": current_step, "total_steps": total_steps}
        audio = self.load_audio(input_path)
        audio_duration = len(audio) / 1000.0
        
        # 2. Denoise (optional)
        transcribe_path = input_path
        if denoise:
            current_step += 1
            yield {"status": "progress", "message": f"Applying noise reduction (intensity: {denoise_intensity})...", "step": current_step, "total_steps": total_steps}
            audio = self.denoise_audio(audio, prop_decrease=denoise_intensity)
            
            # Save denoised audio to temp file for Whisper transcription
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_denoised_path = temp_file.name
            temp_file.close()
            audio.export(temp_denoised_path, format="wav")
            transcribe_path = temp_denoised_path
            logger.info(f"Denoised audio saved to temp file: {temp_denoised_path}")
        
        try:
            # 3. Transcribe
            current_step += 1
            yield {"status": "progress", "message": "Transcribing audio (this may take a moment)...", "step": current_step, "total_steps": total_steps}
            transcript = self.transcribe(transcribe_path)
            
            # 4. Analyze - use selected detection mode
            current_step += 1
            if detection_mode == "gemini" and self.gemini_model:
                yield {"status": "progress", "message": "Analyzing with Gemini AI...", "step": current_step, "total_steps": total_steps}
                detections = self.analyze_with_gemini(transcript, entity_list)
            elif detection_mode == "ai" and self.ner_pipeline:
                yield {"status": "progress", "message": "Analyzing with BERT NER + Regex...", "step": current_step, "total_steps": total_steps}
                detections = self.analyze_text_ai(transcript, entity_list)
            else:
                yield {"status": "progress", "message": "Analyzing with Regex patterns...", "step": current_step, "total_steps": total_steps}
                detections = self.analyze_text(transcript, entity_list)
            
            # 5. Redact
            current_step += 1
            yield {"status": "progress", "message": f"Redacting {len(detections)} sensitive segments...", "step": current_step, "total_steps": total_steps}
            final_audio, count = self.redact_audio(audio, detections, mode)
            
            # 6. Save
            current_step += 1
            yield {"status": "progress", "message": "Saving output file...", "step": current_step, "total_steps": total_steps}
            final_audio.export(output_path, format="wav" if output_path.endswith(".wav") else "mp3")
            
        finally:
            # Clean up temp file
            if temp_denoised_path and temp_os.path.exists(temp_denoised_path):
                temp_os.remove(temp_denoised_path)
                logger.info(f"Cleaned up temp file: {temp_denoised_path}")
        
        end_time = time.time()
        compute_time = end_time - start_time
        
        # Generate human-readable summary with Gemini
        transcript_text = transcript.get("text", "") if transcript else ""
        summary = self.generate_summary_with_gemini(detections, transcript_text, mode)
        
        report = {
            "original_file": input_path,
            "redacted_file": output_path,
            "original_duration_seconds": audio_duration,
            "compute_time_seconds": compute_time,
            "compute_seconds_per_audio_second": compute_time / audio_duration if audio_duration > 0 else 0,
            "redaction_count": count,
            "redaction_mode": mode,
            "noise_reduction_applied": denoise,
            "noise_reduction_intensity": denoise_intensity if denoise else None,
            "summary": summary,  # Human-readable summary
            "detections": detections,
            "entities_detected": len(detections)
        }
        
        logger.info(f"Redaction Report: {json.dumps(report, indent=2)}")
        
        # Final Result Yield
        yield {"status": "complete", "data": report}

    def process_pipeline(self, input_path: str, output_path: str, entity_list: List[str] = [], mode: str = "beep", denoise: bool = False, denoise_intensity: float = 0.8, detection_mode: str = "gemini") -> Dict:
        """
        Legacy wrapper for synchronous execution.
        """
        result = None
        for update in self.process_pipeline_stream(input_path, output_path, entity_list, mode, denoise, denoise_intensity, detection_mode):
            if update["status"] == "complete":
                result = update["data"]
        return result

if __name__ == "__main__":
    # Test execution
    redactor = AudioRedactor(use_mock_transcription=True)
    # Create a dummy file for testing if it doesn't exist? 
    # Actually pydub needs a real file.
    # For now, we will just print that we are ready.
    print("Audio Redaction Service Ready.")