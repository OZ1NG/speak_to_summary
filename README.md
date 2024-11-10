# Speak to Summary
음성 파일의 내용을 요약하는 도구

## How it works

1. 음성 파일을 stt로 추출 (openai - whisper 사용)
    - 필요한 경우 20MB 단위로 segment 생성, 1MB context windows 생성 
2. refine 방식으로 요약 진행

## Usage
```
$ export OPEN_API_KEY="YOUR API KEY"
$ python3 main.py <m4a sound file>
```

## Requirements

- pydub
```
$ pip install pydub

$ sudo apt update
$ sudo apt install ffmpeg
```
