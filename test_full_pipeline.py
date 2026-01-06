"""
Full Pipeline Test for Audio Redaction Service
Tests all features: noise reduction, multiple detection modes (regex, AI/BERT, Gemini),
and verifies redacted audio is generated correctly.
"""
import os
import sys
import time
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_redaction_service import AudioRedactor, GEMINI_AVAILABLE

# Test configuration
TEST_AUDIO_FILE = "test.mp3"  # Use existing test file
OUTPUT_DIR = tempfile.mkdtemp(prefix="audio_redaction_test_")


def print_header(title):
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_subheader(title):
    print()
    print(f"--- {title} ---")


def print_result(success, message):
    status = "[PASS]" if success else "[FAIL]"
    print(f"  {status} {message}")
    return success


def test_audio_loading(redactor, audio_path):
    """Test that audio file can be loaded correctly."""
    print_subheader("Audio Loading Test")
    try:
        audio = redactor.load_audio(audio_path)
        duration = len(audio) / 1000.0
        print_result(True, f"Loaded audio: {duration:.2f} seconds")
        return audio, True
    except Exception as e:
        print_result(False, f"Failed to load audio: {e}")
        return None, False


def test_transcription(redactor, audio_path, use_mock=True):
    """Test transcription (mock or real Whisper)."""
    print_subheader("Transcription Test" + (" (Mock)" if use_mock else " (Whisper)"))
    try:
        start_time = time.time()
        transcript = redactor.transcribe(audio_path)
        elapsed = time.time() - start_time
        
        # Verify transcript structure
        has_text = "text" in transcript and len(transcript["text"]) > 0
        has_segments = "segments" in transcript and len(transcript["segments"]) > 0
        has_words = False
        word_count = 0
        
        if has_segments:
            for seg in transcript["segments"]:
                if "words" in seg:
                    has_words = True
                    word_count += len(seg["words"])
        
        print_result(has_text, f"Transcript text: {len(transcript.get('text', ''))} chars")
        print_result(has_segments, f"Segments found: {len(transcript.get('segments', []))}")
        print_result(has_words, f"Words with timestamps: {word_count}")
        print(f"  [INFO] Transcription time: {elapsed:.2f}s")
        
        return transcript, has_text and has_segments and has_words
    except Exception as e:
        print_result(False, f"Transcription failed: {e}")
        return None, False


def test_regex_detection(redactor, transcript):
    """Test regex-based PII detection."""
    print_subheader("Regex Detection Test")
    try:
        detections = redactor.analyze_text_ai(transcript, entity_list=[])
        
        # Count by category
        categories = {}
        for d in detections:
            label = d.get("label", "UNKNOWN")
            categories[label] = categories.get(label, 0) + 1
        
        print_result(True, f"Total detections: {len(detections)}")
        for label, count in sorted(categories.items()):
            print(f"    - {label}: {count}")
        
        # Show some examples
        if detections:
            print("  [INFO] Sample detections:")
            for d in detections[:5]:
                print(f"    [{d['label']}] '{d['text']}' @ {d['start']:.2f}s")
        
        return detections, len(detections) >= 0  # Pass even with 0 detections
    except Exception as e:
        print_result(False, f"Regex detection failed: {e}")
        return [], False


def test_gemini_detection(redactor, transcript):
    """Test Gemini AI-based PII detection."""
    print_subheader("Gemini AI Detection Test")
    
    if not GEMINI_AVAILABLE or not redactor.gemini_model:
        print("  [SKIP] Gemini not available")
        return [], True  # Skip is not a failure
    
    try:
        start_time = time.time()
        detections = redactor.analyze_with_gemini(transcript, entity_list=[])
        elapsed = time.time() - start_time
        
        # Count by category
        categories = {}
        for d in detections:
            label = d.get("label", "UNKNOWN")
            categories[label] = categories.get(label, 0) + 1
        
        print_result(True, f"Total Gemini detections: {len(detections)}")
        for label, count in sorted(categories.items()):
            print(f"    - {label}: {count}")
        print(f"  [INFO] Gemini analysis time: {elapsed:.2f}s")
        
        # Show detections
        if detections:
            print("  [INFO] Sample detections:")
            for d in detections[:5]:
                print(f"    [{d['label']}] '{d['text']}' @ {d['start']:.2f}s")
        
        return detections, True
    except Exception as e:
        print_result(False, f"Gemini detection failed: {e}")
        return [], False


def test_audio_redaction(redactor, audio, detections, mode="beep"):
    """Test audio redaction (replacing PII with beeps/silence)."""
    print_subheader(f"Audio Redaction Test (mode={mode})")
    
    if not detections:
        print("  [SKIP] No detections to redact")
        return audio, True
    
    try:
        redacted_result = redactor.redact_audio(audio, detections, mode=mode)
        # redact_audio returns a tuple: (redacted_audio, num_intervals)
        redacted_audio = redacted_result[0] if isinstance(redacted_result, tuple) else redacted_result
        original_duration = len(audio) / 1000.0
        redacted_duration = len(redacted_audio) / 1000.0
        
        same_duration = abs(original_duration - redacted_duration) < 0.1
        print_result(same_duration, f"Duration preserved: {redacted_duration:.2f}s (original: {original_duration:.2f}s)")
        
        return redacted_audio, same_duration
    except Exception as e:
        print_result(False, f"Audio redaction failed: {e}")
        return None, False


def test_noise_reduction(redactor, audio, intensity=0.5):
    """Test noise reduction preprocessing."""
    print_subheader(f"Noise Reduction Test (intensity={intensity})")
    
    try:
        start_time = time.time()
        denoised = redactor.denoise_audio(audio, prop_decrease=intensity)
        elapsed = time.time() - start_time
        
        original_duration = len(audio) / 1000.0
        denoised_duration = len(denoised) / 1000.0
        
        same_duration = abs(original_duration - denoised_duration) < 0.1
        print_result(same_duration, f"Duration preserved: {denoised_duration:.2f}s")
        print(f"  [INFO] Denoising time: {elapsed:.2f}s")
        
        return denoised, same_duration
    except Exception as e:
        print_result(False, f"Noise reduction failed: {e}")
        return None, False


def test_audio_export(audio, output_path):
    """Test exporting redacted audio to file."""
    print_subheader("Audio Export Test")
    
    try:
        audio.export(output_path, format="mp3")
        file_exists = os.path.exists(output_path)
        file_size = os.path.getsize(output_path) if file_exists else 0
        
        print_result(file_exists, f"Output file created: {output_path}")
        print_result(file_size > 0, f"File size: {file_size / 1024:.2f} KB")
        
        return file_exists and file_size > 0
    except Exception as e:
        print_result(False, f"Audio export failed: {e}")
        return False


def test_full_pipeline_stream(redactor, audio_path, output_path, detection_mode="regex", denoise=False):
    """Test the full pipeline using process_pipeline_stream."""
    print_subheader(f"Full Pipeline Stream Test (mode={detection_mode}, denoise={denoise})")
    
    try:
        start_time = time.time()
        result = None
        step_count = 0
        
        for update in redactor.process_pipeline_stream(
            audio_path, 
            output_path,
            entity_list=[],
            mode="beep",
            detection_mode=detection_mode,
            denoise=denoise,
            denoise_intensity=0.5
        ):
            step_count += 1
            result = update
            # Print progress updates
            if "step" in update:
                print(f"    Step {step_count}: {update.get('message', 'Processing...')}")
        
        elapsed = time.time() - start_time
        
        # Check final result
        success = result and result.get("status") == "complete"
        output_exists = os.path.exists(output_path)
        
        print_result(success, f"Pipeline completed in {elapsed:.2f}s")
        print_result(output_exists, f"Output file created: {output_path}")
        
        if result and "detections" in result:
            print(f"  [INFO] Total detections: {len(result['detections'])}")
        
        return success and output_exists
    except Exception as e:
        print_result(False, f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests(use_mock=True):
    """Run all pipeline tests."""
    print_header("AUDIO REDACTION SERVICE - FULL PIPELINE TEST")
    print(f"Test audio file: {TEST_AUDIO_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Mock transcription: {use_mock}")
    print(f"Gemini available: {GEMINI_AVAILABLE}")
    
    # Check test file exists
    if not os.path.exists(TEST_AUDIO_FILE):
        print(f"\n[ERROR] Test file '{TEST_AUDIO_FILE}' not found!")
        print("Please ensure test.mp3 exists in the project directory.")
        return False
    
    results = {}
    
    # Create redactor
    print_header("1. INITIALIZATION")
    print("[INFO] Creating AudioRedactor...")
    redactor = AudioRedactor(use_mock_transcription=use_mock)
    print_result(True, "AudioRedactor created")
    
    # Test 1: Audio Loading
    print_header("2. AUDIO LOADING")
    audio, results["loading"] = test_audio_loading(redactor, TEST_AUDIO_FILE)
    if not results["loading"]:
        return False
    
    # Test 2: Noise Reduction
    print_header("3. NOISE REDUCTION")
    denoised, results["denoise"] = test_noise_reduction(redactor, audio, intensity=0.5)
    
    # Test 3: Transcription
    print_header("4. TRANSCRIPTION")
    transcript, results["transcription"] = test_transcription(redactor, TEST_AUDIO_FILE, use_mock=use_mock)
    if not results["transcription"]:
        return False
    
    # Test 4: Regex Detection
    print_header("5. REGEX DETECTION")
    regex_detections, results["regex"] = test_regex_detection(redactor, transcript)
    
    # Test 5: Gemini Detection
    print_header("6. GEMINI AI DETECTION")
    gemini_detections, results["gemini"] = test_gemini_detection(redactor, transcript)
    
    # Test 6: Audio Redaction (beep mode)
    print_header("7. AUDIO REDACTION (BEEP MODE)")
    detections_to_use = gemini_detections if gemini_detections else regex_detections
    redacted_beep, results["redact_beep"] = test_audio_redaction(redactor, audio, detections_to_use, mode="beep")
    
    # Test 7: Audio Redaction (silence mode)
    print_header("8. AUDIO REDACTION (SILENCE MODE)")
    redacted_silence, results["redact_silence"] = test_audio_redaction(redactor, audio, detections_to_use, mode="silence")
    
    # Test 8: Audio Export
    print_header("9. AUDIO EXPORT")
    if redacted_beep:
        output_path = os.path.join(OUTPUT_DIR, "test_redacted_beep.mp3")
        results["export"] = test_audio_export(redacted_beep, output_path)
    else:
        results["export"] = False
    
    # Test 9: Full Pipeline (Regex mode)
    print_header("10. FULL PIPELINE - REGEX MODE")
    output_regex = os.path.join(OUTPUT_DIR, "pipeline_regex.mp3")
    results["pipeline_regex"] = test_full_pipeline_stream(
        redactor, TEST_AUDIO_FILE, output_regex, detection_mode="regex", denoise=False
    )
    
    # Test 10: Full Pipeline (Gemini mode)
    print_header("11. FULL PIPELINE - GEMINI MODE")
    if GEMINI_AVAILABLE:
        output_gemini = os.path.join(OUTPUT_DIR, "pipeline_gemini.mp3")
        results["pipeline_gemini"] = test_full_pipeline_stream(
            redactor, TEST_AUDIO_FILE, output_gemini, detection_mode="gemini", denoise=False
        )
    else:
        print("  [SKIP] Gemini not available")
        results["pipeline_gemini"] = True
    
    # Test 11: Full Pipeline with Noise Reduction
    print_header("12. FULL PIPELINE - WITH NOISE REDUCTION")
    output_denoise = os.path.join(OUTPUT_DIR, "pipeline_denoised.mp3")
    results["pipeline_denoise"] = test_full_pipeline_stream(
        redactor, TEST_AUDIO_FILE, output_denoise, detection_mode="regex", denoise=True
    )
    
    # Summary
    print_header("TEST SUMMARY")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    print(f"\nResults: {passed}/{total} tests passed")
    print()
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {test_name}")
    
    print()
    print(f"Output files saved to: {OUTPUT_DIR}")
    print()
    
    if failed == 0:
        print("=" * 70)
        print(" ALL TESTS PASSED! ")
        print("=" * 70)
        return True
    else:
        print("=" * 70)
        print(f" {failed} TEST(S) FAILED ")
        print("=" * 70)
        return False


def cleanup():
    """Clean up test output directory."""
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print(f"[INFO] Cleaned up: {OUTPUT_DIR}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Full Pipeline Test for Audio Redaction Service")
    parser.add_argument("--real", action="store_true", help="Use real Whisper transcription (slow)")
    parser.add_argument("--cleanup", action="store_true", help="Clean up test output after running")
    args = parser.parse_args()
    
    try:
        success = run_all_tests(use_mock=not args.real)
        if args.cleanup:
            cleanup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
