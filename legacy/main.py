import os
import argparse
import sys
from pydub import AudioSegment
from openai import OpenAI

# OpenAI API 키 설정
api_key = os.environ.get('OPEN_API_KEY')
client = OpenAI(api_key=api_key)

# 파일을 20MB씩 나누고, 1MB context window 추가

def split_audio(file_path, segment_size=20 * 1024 * 1024, overlap_size=1 * 1024 * 1024):
    audio = AudioSegment.from_file(file_path, format="m4a")
    total_size = len(audio)
    segments = []
    start = 0
    
    while start < total_size:
        end = min(start + segment_size, total_size)
        segment = audio[start:end]
        if start > 0:
            segment = audio[max(0, start - overlap_size):end]  # 이전 세그먼트의 끝부분과 겹치게 만듦
        segments.append(segment)
        start += segment_size
    
    return segments

def transcribe_audio_segment(audio_segment):
    with open("temp_audio_segment.mp3", "wb") as f:
        audio_segment.export(f, format="mp3")
    
    with open("temp_audio_segment.mp3", "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format='text'
        )
    return transcript

def summarize_text(text, previous_summary=None):
    if previous_summary:
        prompt = (
            "Refine the following summary with new content added in a detailed and structured markdown format. "
            "Include bullet points, headings, and key details where appropriate.\n"
            "Previous Summary: \n" + previous_summary + "\n" +
            "New Content: \n" + text + "\n\nRefined Summary:"
        )
    else:
        prompt = (
            "Summarize the following text in a detailed markdown format with headings, bullet points, and key information: \n" + text + "\n\nSummary:"
        )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def translate_to_korean(text):
    prompt = (
        "Translate the following text to Korean:\n" + text + "\n\nTranslation:"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def main(file_path):
    # 1. 파일 쪼개기
    segments = split_audio(file_path)
    previous_summary = None

    for i, segment in enumerate(segments):
        print(f"Transcribing segment {i + 1}/{len(segments)}...")
        transcript = transcribe_audio_segment(segment)
        print(f"Transcript {i + 1}:", transcript)
        
        # 2. 요약 진행 및 refine
        print(f"Summarizing segment {i + 1}...")
        previous_summary = summarize_text(transcript, previous_summary)
        print(f"Updated Summary {i + 1}:", previous_summary)
    
    # 최종 요약 결과 출력
    with open(f'final_summary_{os.path.basename(file_path).rsplit(".", 1)[0]}.txt', 'w') as fp:
        fp.write(previous_summary)
    
    # 3. 요약 내용을 한국어로 번역
    print("Translating final summary to Korean...")
    translated_summary = translate_to_korean(previous_summary)
    print("Translated Summary:", translated_summary)
    with open(f'translated_summary_{os.path.basename(file_path).rsplit(".", 1)[0]}.txt', 'w') as fp:
        fp.write(translated_summary)
    print(f'Save translated_summary_{os.path.basename(file_path).rsplit(".", 1)[0]}.txt')
    print('Done')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an audio file and provide Korean translation.")
    parser.add_argument("file_path", type=str, help="Path to the .m4a file")
    
    args = parser.parse_args()
    
    if not args.file_path.endswith(".m4a"):
        print("Error: Only .m4a files are supported.")
        sys.exit(1)
    
    main(args.file_path)
