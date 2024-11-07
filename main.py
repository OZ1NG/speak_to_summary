from openai import OpenAI
import argparse
import sys
import os

# OpenAI API 키 설정
api_key = os.environ.get('OPEN_API_KEY')
client = OpenAI(api_key=api_key)

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format='text'
        )
    return transcript["text"]

def summarize_text(text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
          {"role": "system", "content": "Summarize the following text in the <contents> tag."},
          {"role": "user", "content": f"<contents>{text}</contents>\n\nSummary:"}
        ]
    )
    summary = response.choices[0].message
    return summary

def translate_to_korean(text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Translate the following text in the <contents> tag to Korean."},
            {"role": "user", "content": f"<contents>{text}</contents>\n\nTranslate:"}
        ]
    )
    
    translated_text = response.choices[0].message
    return translated_text

def main(file_path):
    # 1. STT (음성 인식)
    print("Transcribing audio...")
    transcript = transcribe_audio(file_path)
    print("Transcript:", transcript)
    with open(f'transcript_{os.path.basename(file_path).rsplit(".", 0)}.txt', 'w') as fp:
        fp.write(transcript)

    # 2. 텍스트 요약
    print("Summarizing text...")
    summary = summarize_text(transcript)
    print("Summary:", summary)
    with open(f'summary_{os.path.basename(file_path).rsplit(".", 0)}.txt', 'w') as fp:
        fp.write(summary)
    
    # 3. 한국어로 번역
    print("Translating summary to Korean...")
    translated_summary = translate_to_korean(summary)
    print("Translated Summary:", translated_summary)
    with open(f'translated_summary_{os.path.basename(file_path).rsplit(".", 0)}.txt', 'w') as fp:
        fp.write(translated_summary)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an audio file and provide Korean translation.")
    parser.add_argument("file_path", type=str, help="Path to the .m4a file")
    
    args = parser.parse_args()
    
    if not args.file_path.endswith(".m4a"):
        print("Error: Only .m4a files are supported.")
        sys.exit(1)
    
    main(args.file_path)
