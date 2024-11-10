import os
import argparse
import sys
import json
from pydub import AudioSegment
from openai import OpenAI

# OpenAI API 키 설정
api_key = os.environ.get('OPEN_API_KEY')
if api_key == None:
    print(f"[!] Please set OPEN_API_KEY as an environment variable.")
    exit(0)
client = OpenAI(api_key=api_key)

# 파일을 23MB씩 나누고, 1MB context window 추가
def split_audio(file_path, segment_size_mb=23, overlap_size_mb=1):
    audio = AudioSegment.from_file(file_path, format="m4a")
    segment_size = segment_size_mb * 1024 * 1024  # Convert MB to bytes
    overlap_size = overlap_size_mb * 1024 * 1024  # Convert MB to bytes

    total_duration_ms = len(audio)  # Total duration in milliseconds
    bytes_per_ms = len(audio.raw_data) / total_duration_ms  # Calculate bytes per millisecond

    segment_size_ms = int(segment_size / bytes_per_ms)  # Convert segment size to milliseconds
    overlap_size_ms = int(overlap_size / bytes_per_ms)  # Convert overlap size to milliseconds

    segments = []
    start = 0

    while start < total_duration_ms:
        end = min(start + segment_size_ms, total_duration_ms)
        segment = audio[start:end]
        if start > 0:
            segment = audio[max(0, start - overlap_size_ms):end]  # Add overlap from the previous segment
        segments.append(segment)
        start += segment_size_ms

    return segments

def transcribe_audio_segment(audio_segment, segment_index, result_dir):
    # Check if transcript already exists
    transcript_filename = os.path.join(result_dir, f"transcript_segment_{segment_index}.json")
    if os.path.exists(transcript_filename):
        with open(transcript_filename, "r") as f:
            transcript_data = json.load(f)
            return transcript_data["text"]
    
    # If transcript doesn't exist, create it
    temp_audio_path = os.path.join(result_dir, "temp_audio_segment.wav")
    with open(temp_audio_path, "wb") as f:
        audio_segment.export(f, format="wav")  # Use m4a format to minimize quality loss
    
    with open(temp_audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format='text'
        )
    
    # Save transcript to file
    with open(transcript_filename, "w") as f:
        json.dump({"text": transcript}, f)
    
    return transcript

def extract_categories_and_keywords(transcripts):
    prompt = (
        "Extract the categories and keywords from the following text, focusing on identifying any professional knowledge, technical skills, and specialized terms that are mentioned:\n" + transcripts + "\n\nCategories and Keywords:"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def summarize_text(text, previous_summary=None, categories_keywords=None):
    system_prompt = (
        "너는 최고의 내용 요약 정리 전문가야. "
        "너는 최고의 내용 요약 정리 전문가로써 다음의 작업 내용 대로 항상 일을 진행해.\n\n작업1. 정리할 소주제를 생각한다.\n작업2. 소주제에서 다루는 문제점과 해결 방법을 키워드와 기술적인 관점에서 자세하고 정확히 파악한다.\n작업3. 문제점과 해결 방법이 설명된 문구를 인용구의 형식으로 남기면서 이전의 요약 내용에 추가 종합 정리한다.\n\n"
        "작업의 결과는 <workflow></workflow> 태그안에서 정리해줘. "
        "요약 정리 결과는 <res></res> 태그 안에서 마크다운 형태로 정리하며 모든 소제목의 내용은 보안 기술적인 관점으로 발표에서 다루는 문제점과 그에 대한 해결 방법을 중점으로 두고 이해하기 쉽게 길고 자세하게 정리해줘. "
        "요약 정리 결과는 제목, Bullet Point, 핵심 정보를 포함한 상세한 마크다운 형식으로 대답해줘. "
        "특정 조치가 왜 이루어졌는지, 무엇이 이를 이끌었는지, 그리고 그 결과가 무엇이었는지 설명하세요. 주요 세부사항이 강조되도록 항목, 제목, 핵심 정보를 포함하고, 제공된 범주와 키워드를 고려하여 가장 중요한 정보가 부각되도록 해줘.\n"
        # "Can you provide a comprehensive summary of the given text? The summary should cover all the key points and main ideas presented in the original text, while also condensing the information into a concise and easy-to-understand format. Please ensure that the summary includes relevant details and examples that support the main ideas, while avoiding any unnecessary information or repetition. The length of the summary should be appropriate for the length and complexity of the original text, providing a clear and accurate overview without omitting any important information. "
    )
    
    if previous_summary:
        prompt = (
            "Refine and extend the following summary with new content added. "
            "Categories and Keywords: \n" + categories_keywords + "\n"
            "Previous Summary: \n" + previous_summary + "\n" +
            "New Content(text): \n" + text + "\n\nRefined and Extended Summary:"
        )        
    else:
        prompt = (
            "Categories and Keywords: \n" + categories_keywords + "\n"
            "Content(text): \n" + text + "\n\nDetailed Summary:"
        )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content.strip()

def translate_to_korean(text, categories_keywords):
    system_prompt = "Translate the following text to Korean, and such that any technical terms are retained in their original form. 결과는 <res></res> 태그에 넣어 답변해줘. "
    
    prompt = (
        "Text: \n" + text + "\n\nKeywords to retain: \n" + categories_keywords + "\n\nTranslation:"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def main(file_path):
    # 결과 디렉토리 생성
    result_dir = f"result_{os.path.basename(file_path).rsplit('.', 1)[0]}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 1. 파일 쪼개기
    segments = split_audio(file_path)
    previous_summary = None
    all_transcripts = ""

    for i, segment in enumerate(segments):
        print(f"Transcribing segment {i + 1}/{len(segments)}...")
        transcript = transcribe_audio_segment(segment, i, result_dir)
        print(f"Transcript {i + 1}:", transcript)
        all_transcripts += transcript + "\n"
    
    # 2. 카테고리 및 키워드 추출
    print("Extracting categories and keywords...")
    categories_keywords = extract_categories_and_keywords(all_transcripts)
    print("Categories and Keywords:", categories_keywords)

    # 3. 요약 진행 및 refine
    for i, segment in enumerate(segments):
        print(f"Summarizing segment {i + 1}...")
        transcript = transcribe_audio_segment(segment, i, result_dir)
        previous_summary = summarize_text(transcript, previous_summary, categories_keywords)
        print(f"Updated Summary {i + 1}:", previous_summary)
    
    # 최종 요약 결과 출력
    final_summary_path = os.path.join(result_dir, f'final_summary.txt')
    with open(final_summary_path, 'w') as fp:
        fp.write(previous_summary)
    
    # 4. 요약 내용을 한국어로 번역
    print("Translating final summary to Korean...")
    translated_summary = translate_to_korean(previous_summary, categories_keywords)
    print("Translated Summary:", translated_summary)
    translated_summary_path = os.path.join(result_dir, f'translated_summary.txt')
    with open(translated_summary_path, 'w') as fp:
        fp.write(translated_summary)
    print(f'Save translated summary at {translated_summary_path}')
    print('Done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an audio file and provide Korean translation.")
    parser.add_argument("file_path", type=str, help="Path to the .m4a file")
    
    args = parser.parse_args()
    
    if not args.file_path.endswith(".m4a"):
        print("Error: Only .m4a files are supported.")
        sys.exit(1)
    
    main(args.file_path)
