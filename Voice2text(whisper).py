import speech_recognition as sr
import whisper
import os


def transcribe_real_time():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Transcribing...")
        text = recognizer.recognize_whisper(audio)
        print("Transcription:", text)
        return text
    except Exception as e:
        print("Error:", str(e))
        return None


def transcribe_audio_file(file_path):
    model = whisper.load_model("base")  # You can use "tiny", "small", "medium", "large"
    result = model.transcribe(file_path)
    print("Transcription:", result["text"])
    return result["text"]


if __name__ == "__main__":
    choice = input("Choose an option:\n1. Real-time Transcription\n2. Transcribe an Audio File\nEnter (1/2): ")

    if choice == "1":
        transcribe_real_time()
    elif choice == "2":
        file_path = input("Enter the path to the audio file: ")
        if os.path.exists(file_path):
            transcribe_audio_file(file_path)
        else:
            print("File not found!")
    else:
        print("Invalid choice!")
