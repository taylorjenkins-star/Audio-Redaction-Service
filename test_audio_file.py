"""
Test script to verify price detection on test.mp3
Specifically looking for the price 1339.92 around 2:52-2:58 (172-178 seconds)
"""
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_redaction_service import AudioRedactor

def main():
    print("=" * 60)
    print("TESTING PRICE DETECTION ON test.mp3")
    print("=" * 60)
    print("Looking for price '1339.92' around 2:52-2:58 (172-178 seconds)")
    print()
    
    # Check test file exists
    test_file = "test.mp3"
    if not os.path.exists(test_file):
        print(f"[ERROR] Test file '{test_file}' not found!")
        return 1
    
    print(f"[INFO] Loading audio file: {test_file}")
    print(f"[INFO] File size: {os.path.getsize(test_file) / 1024 / 1024:.2f} MB")
    print()
    
    # Create redactor (use real transcription for actual test)
    print("[INFO] Creating AudioRedactor (this may take a moment to load models)...")
    redactor = AudioRedactor(use_mock_transcription=False)  # Use real Whisper!
    
    print("[INFO] Loading audio...")
    audio = redactor.load_audio(test_file)
    audio_duration = len(audio) / 1000.0
    print(f"[INFO] Audio duration: {audio_duration:.2f} seconds ({audio_duration/60:.2f} minutes)")
    print()
    
    print("[INFO] Transcribing audio with Whisper (this will take a while)...")
    transcript = redactor.transcribe(test_file)
    
    print("[INFO] Transcription complete!")
    print()
    
    # Show relevant transcript portion (around 2:52-2:58)
    print("=" * 60)
    print("WORDS NEAR 2:52-2:58 (172-178 seconds)")
    print("=" * 60)
    
    target_start = 170  # 2:50
    target_end = 180    # 3:00
    
    words = []
    if "segments" in transcript:
        for segment in transcript["segments"]:
            if "words" in segment:
                for word in segment["words"]:
                    if target_start <= word.get("start", 0) <= target_end:
                        words.append(word)
                        print(f"  {word['start']:.2f}s: '{word['word']}'")
    
    print()
    print("=" * 60)
    print("ANALYZING FOR PII...")
    print("=" * 60)
    
    detections = redactor.analyze_text_ai(transcript, entity_list=[])
    
    print(f"\n[INFO] Total detections: {len(detections)}")
    print()
    
    # Filter to show PRICE detections
    price_detections = [d for d in detections if d['label'] == 'PRICE']
    print(f"PRICE DETECTIONS ({len(price_detections)}):")
    for d in price_detections:
        minutes = int(d['start'] // 60)
        seconds = d['start'] % 60
        print(f"  [{minutes}:{seconds:05.2f}] '{d['text']}'")
    
    # Check specifically for the missed price
    target_price = "1339.92"
    found_target = any(target_price in d['text'] or d['text'] in target_price for d in price_detections)
    
    print()
    print("=" * 60)
    if found_target:
        print(f"[SUCCESS] Found target price '{target_price}'!")
    else:
        print(f"[FAIL] Target price '{target_price}' NOT found in detections")
        print()
        print("All detected prices:")
        for d in price_detections:
            print(f"  - {d['text']} @ {d['start']:.2f}s")
    print("=" * 60)
    
    return 0 if found_target else 1

if __name__ == "__main__":
    sys.exit(main())
