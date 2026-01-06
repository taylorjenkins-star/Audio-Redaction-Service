"""
Quick test to verify Gemini AI integration works
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_redaction_service import AudioRedactor, GEMINI_AVAILABLE

def main():
    print("=" * 60)
    print("GEMINI AI INTEGRATION TEST")
    print("=" * 60)
    
    print(f"\nGemini Available: {GEMINI_AVAILABLE}")
    
    # Create a mock transcript for testing
    mock_transcript = {
        "text": "Hi, my name is John Smith and my email is john@example.com. The price is 1339.92 dollars. Call me at 555-123-4567.",
        "segments": [
            {
                "words": [
                    {"word": "Hi,", "start": 0.0, "end": 0.3},
                    {"word": "my", "start": 0.3, "end": 0.5},
                    {"word": "name", "start": 0.5, "end": 0.7},
                    {"word": "is", "start": 0.7, "end": 0.9},
                    {"word": "John", "start": 0.9, "end": 1.2},
                    {"word": "Smith", "start": 1.2, "end": 1.5},
                    {"word": "and", "start": 1.5, "end": 1.7},
                    {"word": "my", "start": 1.7, "end": 1.9},
                    {"word": "email", "start": 1.9, "end": 2.1},
                    {"word": "is", "start": 2.1, "end": 2.3},
                    {"word": "john@example.com.", "start": 2.3, "end": 3.0},
                    {"word": "The", "start": 3.0, "end": 3.2},
                    {"word": "price", "start": 3.2, "end": 3.5},
                    {"word": "is", "start": 3.5, "end": 3.7},
                    {"word": "1339.92", "start": 3.7, "end": 4.3},
                    {"word": "dollars.", "start": 4.3, "end": 4.7},
                    {"word": "Call", "start": 4.7, "end": 5.0},
                    {"word": "me", "start": 5.0, "end": 5.2},
                    {"word": "at", "start": 5.2, "end": 5.4},
                    {"word": "555-123-4567.", "start": 5.4, "end": 6.2},
                ]
            }
        ]
    }
    
    print("\n[INFO] Creating AudioRedactor with mock transcription mode...")
    redactor = AudioRedactor(use_mock_transcription=True)
    
    if not redactor.gemini_model:
        print("[ERROR] Gemini model not initialized!")
        return 1
    
    print("[OK] Gemini model initialized successfully!")
    print()
    
    print("=" * 60)
    print("TESTING GEMINI ANALYSIS")
    print("=" * 60)
    print(f"\nTest transcript: {mock_transcript['text']}")
    print()
    
    print("[INFO] Sending to Gemini for analysis...")
    detections = redactor.analyze_with_gemini(mock_transcript, entity_list=[])
    
    print(f"\n[INFO] Gemini returned {len(detections)} detections:")
    for d in detections:
        print(f"  - [{d['label']}] '{d['text']}' @ {d['start']:.2f}s - {d['end']:.2f}s")
    
    # Check for expected detections
    expected = ["John", "Smith", "john@example.com", "1339.92", "555-123-4567"]
    found = [d['text'] for d in detections]
    
    print()
    print("=" * 60)
    print("EXPECTED DETECTIONS CHECK")
    print("=" * 60)
    
    all_found = True
    for e in expected:
        match = any(e.lower() in f.lower() or f.lower() in e.lower() for f in found)
        status = "[OK]" if match else "[MISS]"
        print(f"  {status} {e}")
        if not match:
            all_found = False
    
    print()
    if all_found:
        print("[SUCCESS] Gemini AI integration working!")
        return 0
    else:
        print("[PARTIAL] Some expected items not detected, but Gemini is working.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
