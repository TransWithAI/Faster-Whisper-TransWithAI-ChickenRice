@echo off
chcp 65001
set cpath=%~dp0
set cpath=%cpath:~0,-1%
"%cpath%\infer.exe" --audio_suffixes="mp3,wav,flac,m4a,aac,ogg,wma,mp4,mkv,avi,mov,webm,flv,wmv" --sub_formats="srt,vtt,lrc" --device="cpu" --compute_type="int16" %*
pause
