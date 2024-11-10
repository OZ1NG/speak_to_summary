import os
import argparse
import sys
import json
from pydub import AudioSegment
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document


# OpenAI API 키 설정
api_key = os.environ.get('OPENAI_API_KEY')
if api_key is None:
    print(f"[!] Please set OPENAI_API_KEY as an environment variable.")
    exit(0)
client = OpenAI(api_key=api_key)
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

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

def summarize_text_refine(transcripts):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(transcripts)
    docs = [Document(page_content=t) for t in texts]
    
    # Use Langchain's load_summarize_chain for refining summary
    summarize_chain = load_summarize_chain(llm, chain_type="refine")
    summary = summarize_chain.run(docs)
    
    return summary

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
    all_transcripts = []

    for i, segment in enumerate(segments):
        print(f"Transcribing segment {i + 1}/{len(segments)}...")
        transcript = transcribe_audio_segment(segment, i, result_dir)
        print(f"Transcript {i + 1}:", transcript)
        all_transcripts.append(transcript)
    
    # 2. 카테고리 및 키워드 추출
    print("Extracting categories and keywords...")
    categories_keywords = extract_categories_and_keywords("\n".join(all_transcripts))
    print("Categories and Keywords:", categories_keywords)

    # 3. 요약 진행 및 refine (Langchain 사용)
    print("Summarizing transcripts using refine method...")
    refined_summary = summarize_text_refine("\n".join(all_transcripts))
    print("Refined Summary:", refined_summary)
    
    # 최종 요약 결과 출력
    final_summary_path = os.path.join(result_dir, f'final_summary.txt')
    with open(final_summary_path, 'w') as fp:
        fp.write(refined_summary)
    
    # 4. 요약 내용을 한국어로 번역
    print("Translating final summary to Korean...")
    translated_summary = translate_to_korean(refined_summary, categories_keywords)
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
