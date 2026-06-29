import json
import os
import argparse
from urllib import response
from elevenlabs.client import ElevenLabs

greetings_path = "greetings/"
audio_path = "date/recordings/"

def generate_audio_file(text, name):
    global audio_path, client, voice_id
    os.makedirs(os.path.join(audio_path, name), exist_ok=True)
    filename = f"compliment-{hex(hash(text))}.mp3"
    response = client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_v3",
                output_format="mp3_44100_128"
            )
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
    subparsers = parser.add_subparsers(help='sub-command help', dest='command')
    subparsers.add_parser('names', help='generate audio files for names')
    subparsers.add_parser('greetings', help='generate audio files for greetings')
    args = parser.parse_args()
    if args.command == 'greetings':
        generate_greetings()
    elif args.command == 'all':
        generate_greetings()
    else:
        print("Please specify a command: names or greetings")