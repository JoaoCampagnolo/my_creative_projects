import pyaudio
import wave
import numpy as np
import pygame
from pydub import AudioSegment
import noisereduce as nr
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

# Initialize the PyAudio and other variables
p = pyaudio.PyAudio()
RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Load the pre-trained Wav2Vec 2.0 large model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

def record_audio(stream):
    frames = []
    for _ in range(0, int(RATE / CHUNK * 3)):  # Record for 5 seconds
        data = stream.read(CHUNK)
        frames.append(data)
    
    wf = wave.open("output.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def reduce_noise():
    with wave.open("output.wav", 'rb') as wf:
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16)
    
    # Perform noise reduction
    reduced_noise = nr.reduce_noise(y=audio_data, sr=rate)
    
    # Save the noise-reduced audio
    with wave.open("output_reduced.wav", 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(rate)
        wf.writeframes(reduced_noise.tobytes())

def transcribe_audio():
    speech, rate = torchaudio.load("output_reduced.wav")
    input_values = processor(speech.squeeze(), return_tensors="pt", sampling_rate=rate).input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription.lower()

def play_song():
    pygame.mixer.init()
    song = AudioSegment.from_mp3("all_of_the_lights.mp3")
    song.export("all_of_the_lights.wav", format="wav")
    pygame.mixer.music.load("all_of_the_lights.wav")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def main():
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Listening for the word 'light'...")
    
    try:
        while True:
            record_audio(stream)
            reduce_noise()
            transcription = transcribe_audio()
            print("You said: ", transcription)
            if "light" in transcription:
                print("Playing 'All of the Lights' by Kanye West")
                play_song()
                break  # Exit the loop after playing the song
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
