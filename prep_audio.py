import json
import os
import argparse
from urllib import response
from elevenlabs.client import ElevenLabs

greetings_path = "data/greetings/"
audio_path = "data/recordings/"

def generate_audio_file(text, name):
    global audio_path, client, voice_id
    os.makedirs(os.path.join(audio_path, name), exist_ok=True)
    filename = f"compliment-{abs(hash(text)):015x}.mp3"
    response = client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_v3",
                output_format="mp3_44100_128"
            )
    if os.path.exists(os.path.join(audio_path, name, filename)):
        print(f"Audio file already exists for {name}/{filename}: {text.strip()}")
        return
    with open(os.path.join(audio_path, name, filename), "wb") as audio_file:
        for chunk in response:
            if chunk:
                audio_file.write(chunk)
    print(f"Generated audio for {name}/{filename}: {text.strip()}")

def generate_greetings():
    # generate audio files from text files
    for f in [x for x in os.listdir(greetings_path) if x.endswith(".txt")]:
        if f.startswith('.'):
            continue
        name = os.path.splitext(f)[0]
        with open(os.path.join(greetings_path,f), "r", encoding="utf-8", errors='ignore') as file:
            lines = file.readlines()
            for index, text in enumerate(lines):
                if text.strip() == "":
                    continue
                generate_audio_file(text, name)
        
if __name__ == "__main__":
    os.makedirs(audio_path, exist_ok=True)

    with open('secrets.json') as f:
        secrets = json.load(f)
    voice_id = secrets['voice_id']

    client = ElevenLabs(api_key=secrets['api_key'])
    parser = argparse.ArgumentParser("prep_audio.py")
    parser.add_argument("--clean", action="store_true", help="Remove existing audio files before generating new ones")
    parser.add_argument("--voice", type=str, default=voice_id, help="Voice ID to use for text-to-speech")
    args = parser.parse_args()
    voice_id = args.voice
    if args.clean:
        for root, dirs, files in os.walk(audio_path):
            for file in files:
                os.remove(os.path.join(root, file))
    generate_greetings()
