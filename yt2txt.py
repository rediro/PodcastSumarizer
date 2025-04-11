import os
import whisper
import yt_dlp


def download_audio(video_url, output_path="audio.mp3"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,  # yt-dlp may append the format extension
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    # Ensure correct filename
    if not os.path.exists(output_path) and os.path.exists(output_path + ".mp3"):
        os.rename(output_path + ".mp3", output_path)

    return output_path


def transcribe_audio(audio_path):
    model = whisper.load_model("base")  # Change model size if needed
    result = model.transcribe(audio_path)
    return result["text"]


if __name__ == "__main__":
    video_url = input("Enter YouTube video URL: ")
    audio_file = download_audio(video_url)

    if not os.path.exists(audio_file):
        print(f"Error: Audio file {audio_file} not found!")
    else:
        transcript = transcribe_audio(audio_file)
        print("\nTranscription:\n", transcript)

        with open("transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript)
        print("\nTranscript saved to transcript.txt")
