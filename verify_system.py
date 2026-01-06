import os
import time
import requests
import subprocess
import sys
from pydub import AudioSegment
from pydub.generators import Sine

def generate_test_audio(filename="test_audio.wav"):
    print(f"Generating test audio file: {filename}")
    # Generate 10 seconds of 440Hz tone
    sound = Sine(440).to_audio_segment(duration=10000)
    sound.export(filename, format="wav")
    return filename

def test_server():
    print("Starting server...")
    # Start server in background
    # We use sys.executable to ensure we use the same python interpreter
    process = subprocess.Popen([sys.executable, "server.py"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
    
    try:
        # Wait for server to start (longer for AI model load)
        print("Waiting for server to startup (60s)...")
        time.sleep(60) 
        
        # Check if process is still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print("Server failed to start.")
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            return False

        # Generate test file
        test_file = generate_test_audio()
        
        # Send Request
        print("Sending request to /redact...")
        url = "http://localhost:8000/redact"
        files = {'file': open(test_file, 'rb')}
        data = {'entities': 'secret, confidential', 'mode': 'beep'}
        
        try:
            response = requests.post(url, files=files, data=data)
            print(f"Response Status: {response.status_code}")
            
            if response.status_code == 200:
                print("Success! Response JSON:")
                print(response.json())
                # Verify AI detection implies more sophisticated logic, but for now we check basic success
                return True
            else:
                print("Request failed.")
                print(response.text)
                return False
                
        except requests.exceptions.ConnectionError:
            print("Could not connect to server.")
            return False
            
    finally:
        print("Killing server...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    if test_server():
        print("\n[PASS] System Version Passed")
    else:
        print("\n[FAIL] Verification Failed")
