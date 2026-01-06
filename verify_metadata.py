import os
import time
import requests
import subprocess
import sys
import static_ffmpeg
from pydub import AudioSegment
from pydub.generators import Sine

import shutil

static_ffmpeg.add_paths()

# Clean pycache to ensure fresh code load
shutil.rmtree('__pycache__', ignore_errors=True)

def test_metadata():
    print("Starting server...")
    process = subprocess.Popen([sys.executable, "server.py"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
    
    try:
        print("Waiting for server to startup (30s)...")
        time.sleep(30) 
        
        # Generator
        filename = "test_metadata.wav"
        Sine(440).to_audio_segment(duration=3000).export(filename, format="wav")
        
        # Send Request
        print("Sending request...")
        url = "http://localhost:8000/redact"
        files = {'file': open(filename, 'rb')}
        # 'test' should match entity list if logic works
        data = {'entities': 'test', 'mode': 'beep'}
        
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            json_resp = response.json()
            print("Response JSON keys:", json_resp.keys())
            if "detections" in json_resp:
                print("Detections found:", json_resp["detections"])
                return True
            else:
                print("MISSING 'detections' field in response!")
                # Print response for debugging
                print("Full Response:", json_resp)
                return False
        else:
            print(f"Request failed: {response.text}")
            return False

    finally:
        print("Killing server...")
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
        
        # Print server logs
        stdout, stderr = process.communicate()
        print("\n--- SERVER STDOUT ---")
        print(stdout)
        print("\n--- SERVER STDERR ---")
        print(stderr)


if __name__ == "__main__":
    if test_metadata():
        print("\n[PASS] Metadata Verification Passed")
    else:
        print("\n[FAIL] Metadata Verification Failed")
