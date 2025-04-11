import os
import whisper
from pytube import YouTube


def download_audio(video_url, output_path="audio.mp3"):
    yt = YouTube(video_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_file = audio_stream.download(filename=output_path)
    print(f"Downloaded audio: {audio_file}")
    return audio_file


def transcribe_audio(audio_path):
    model = whisper.load_model("base")  # Change to "small", "medium", or "large" for better accuracy
    result = model.transcribe(audio_path)
    return result["text"]


if __name__ == "__main__":
    video_url = input("Enter YouTube video URL: ")
    audio_file = download_audio(video_url)
    transcript = transcribe_audio(audio_file)

    print("\nTranscription:\n")
    print(transcript)

    # Optional: Save to a text file
    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript)
    print("\nTranscript saved to transcript.txt")
