"""
Test Script for Audio Redaction Service
========================================

This script tests the PII detection patterns and audio redaction pipeline
to ensure proper functionality.

Run with: python test_redaction.py
"""

import re
import sys
import json
from typing import List, Dict

# Test data for pattern validation
TEST_CASES = {
    "EMAIL": [
        ("john.doe@example.com", True),
        ("test@test.co.uk", True),
        ("invalid-email", False),
        ("user.name+tag@domain.org", True),
    ],
    "PRICE": [
        ("$150.25", True),
        ("$1,234.56", True),
        ("1339.92", True),  # The missed case!
        ("$99", True),
        ("1,234,567.89", True),
        ("$ 50.00", True),
        ("abc", False),
        ("12", False),  # No decimal - not a price
    ],
    "PHONE": [
        ("555-123-4567", True),
        ("5551234567", True),
        ("555.123.4567", True),
        ("12345", False),
    ],
    "SSN": [
        ("123-45-6789", True),
        ("123456789", True),
        ("12345", False),
    ],
}

# Regex patterns (must match audio_redaction_service.py)
PATTERNS = {
    "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "PRICE": r'(?:\$\s?)?\d+(?:,\d{3})*\.\d{2}\b|\$\d+(?:\.\d{2})?\b',
    "PHONE": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    "SSN": r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
}


def test_pattern(pattern_name: str, pattern: str, test_cases: List[tuple]) -> Dict:
    """Test a regex pattern against test cases."""
    results = {
        "passed": 0,
        "failed": 0,
        "failures": []
    }
    
    compiled = re.compile(pattern)
    
    for test_string, should_match in test_cases:
        match = compiled.search(test_string)
        matched = match is not None
        
        if matched == should_match:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["failures"].append({
                "input": test_string,
                "expected_match": should_match,
                "actual_match": matched
            })
    
    return results


def test_all_patterns():
    """Run all pattern tests."""
    print("=" * 60)
    print("AUDIO REDACTION SERVICE - PATTERN TESTS")
    print("=" * 60)
    
    all_passed = True
    total_passed = 0
    total_failed = 0
    
    for pattern_name, test_cases in TEST_CASES.items():
        if pattern_name not in PATTERNS:
            print(f"\n[WARN]  Pattern '{pattern_name}' not defined in PATTERNS dict")
            continue
            
        pattern = PATTERNS[pattern_name]
        results = test_pattern(pattern_name, pattern, test_cases)
        
        total_passed += results["passed"]
        total_failed += results["failed"]
        
        status = "[PASS]" if results["failed"] == 0 else "[FAIL]"
        print(f"\n{status} {pattern_name}: {results['passed']}/{len(test_cases)} tests passed")
        
        if results["failures"]:
            all_passed = False
            for failure in results["failures"]:
                expected = "MATCH" if failure["expected_match"] else "NO MATCH"
                actual = "MATCH" if failure["actual_match"] else "NO MATCH"
                print(f"      [X] '{failure['input']}' - Expected: {expected}, Got: {actual}")
    
    print("\n" + "=" * 60)
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    print("=" * 60)
    
    return all_passed


def test_transcription_analysis():
    """Test the analyze_text function with mock transcription data."""
    print("\n" + "=" * 60)
    print("TRANSCRIPTION ANALYSIS TEST")
    print("=" * 60)
    
    # Mock transcription similar to what Whisper would return
    mock_transcript = {
        "text": "The price is 1339.92 dollars, my email is test@example.com and phone is 555-123-4567",
        "segments": [
            {
                "words": [
                    {"word": "The", "start": 0.0, "end": 0.2},
                    {"word": "price", "start": 0.2, "end": 0.5},
                    {"word": "is", "start": 0.5, "end": 0.6},
                    {"word": "1339.92", "start": 0.6, "end": 1.2},  # Price without $
                    {"word": "dollars,", "start": 1.2, "end": 1.5},
                    {"word": "my", "start": 1.5, "end": 1.6},
                    {"word": "email", "start": 1.6, "end": 1.8},
                    {"word": "is", "start": 1.8, "end": 1.9},
                    {"word": "test@example.com", "start": 1.9, "end": 2.5},
                    {"word": "and", "start": 2.5, "end": 2.6},
                    {"word": "phone", "start": 2.6, "end": 2.8},
                    {"word": "is", "start": 2.8, "end": 2.9},
                    {"word": "555-123-4567", "start": 2.9, "end": 3.5},
                ]
            }
        ]
    }
    
    # Expected detections
    expected_detections = [
        {"label": "PRICE", "text": "1339.92"},
        {"label": "EMAIL", "text": "test@example.com"},
        {"label": "PHONE", "text": "555-123-4567"},
    ]
    
    try:
        from audio_redaction_service import AudioRedactor
        
        # Create redactor in mock mode
        redactor = AudioRedactor(use_mock_transcription=True)
        
        # Analyze the mock transcript
        detections = redactor.analyze_text(mock_transcript, entity_list=[])
        
        print(f"\nDetections found: {len(detections)}")
        
        # Check each expected detection
        for expected in expected_detections:
            found = any(
                d["label"] == expected["label"] and expected["text"] in d["text"]
                for d in detections
            )
            status = "[OK]" if found else "[X]"
            print(f"  {status} {expected['label']}: '{expected['text']}'")
            
        # Print all actual detections
        print("\nAll detections:")
        for d in detections:
            print(f"  - [{d['label']}] '{d['text']}' @ {d['start']:.2f}s - {d['end']:.2f}s")
            
        return len(detections) >= len(expected_detections)
        
    except ImportError as e:
        print(f"[X] Could not import AudioRedactor: {e}")
        return False
    except Exception as e:
        print(f"[X] Error during analysis: {e}")
        return False


def test_price_edge_cases():
    """Test specific price edge cases that might be missed."""
    print("\n" + "=" * 60)
    print("PRICE EDGE CASE TESTS")
    print("=" * 60)
    
    price_pattern = re.compile(PATTERNS["PRICE"])
    
    edge_cases = [
        # Format: (input, should_match, description)
        ("1339.92", True, "Price without $ (the missed case)"),
        ("$1339.92", True, "Price with $"),
        ("$ 1339.92", True, "Price with $ and space"),
        ("1,339.92", True, "Price with comma"),
        ("$1,339.92", True, "Price with $ and comma"),
        ("12,345,678.99", True, "Large price with commas"),
        ("0.99", True, "Small price"),
        ("$0.99", True, "Small price with $"),
        ("99.99", True, "Two-digit price"),
        ("999.99", True, "Three-digit price"),
    ]
    
    all_passed = True
    for text, should_match, description in edge_cases:
        match = price_pattern.search(text)
        matched = match is not None
        status = "[OK]" if matched == should_match else "[X]"
        
        if matched != should_match:
            all_passed = False
            
        print(f"  {status} {description}: '{text}' - {'MATCH' if matched else 'NO MATCH'}")
        if match:
            print(f"      Matched: '{match.group()}'")
    
    return all_passed


def main():
    """Run all tests."""
    print("\n[TEST] Starting Audio Redaction Service Tests...\n")
    
    results = []
    
    # Test 1: Pattern matching
    results.append(("Pattern Tests", test_all_patterns()))
    
    # Test 2: Price edge cases
    results.append(("Price Edge Cases", test_price_edge_cases()))
    
    # Test 3: Transcription analysis
    results.append(("Transcription Analysis", test_transcription_analysis()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] ALL TESTS PASSED!")
        return 0
    else:
        print("[ERROR] SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
