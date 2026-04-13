@echo off
echo Starting BIST Agentic RAG Backend...
cd /d "%~dp0"
title BIST Agentic RAG Backend (Port 8080)
python -m api.main
pause
