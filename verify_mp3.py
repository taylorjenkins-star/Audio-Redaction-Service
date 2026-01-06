import os
import time
import requests
import subprocess
import sys
import static_ffmpeg
from pydub import AudioSegment
from pydub.generators import Sine

# Ensure paths are added
static_ffmpeg.add_paths()

def generate_mp3(filename="test_audio.mp3"):
    print(f"Generating test audio file: {filename}")
    try:
        # Generate 5 seconds of 440Hz tone
        sound = Sine(440).to_audio_segment(duration=5000)
        sound.export(filename, format="mp3")
        print("MP3 generated successfully.")
        return filename
    except Exception as e:
        print(f"Failed to generate MP3: {e}")
        return None

def test_server():
    print("Starting server...")
    process = subprocess.Popen([sys.executable, "server.py"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
    
    try:
        # Wait for server to start (AI model load takes time)
        print("Waiting for server to startup (30s)...")
        time.sleep(30) 
        
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print("Server failed to start.")
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            return False

        # Generate test file
        test_file = generate_mp3()
        if not test_file:
            return False
        
        # Send Request
        print("Sending request to /redact with MP3...")
        url = "http://localhost:8000/redact"
        files = {'file': open(test_file, 'rb')}
        data = {'entities': 'test', 'mode': 'beep'}
        
        try:
            response = requests.post(url, files=files, data=data)
            print(f"Response Status: {response.status_code}")
            
            if response.status_code == 200:
                print("Success! Response JSON:")
                print(response.json())
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
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()

if __name__ == "__main__":
    if test_server():
        print("\n[PASS] MP3 Support Verified")
    else:
        print("\n[FAIL] MP3 Verification Failed")
