# OCR Document Processing Engine

A full-stack application that extracts text, table structure, bounding
boxes, and confidence scores from scanned business documents.

## Status

Active Development

## Tech Stack

- Frontend: React, Vite
- API: FastAPI
- OCR: PaddleOCR, OpenCV
- Background Processing: Celery, Redis

## Features

- Document upload and asynchronous OCR processing
- Job status polling
- Text, table, bounding-box, and confidence-score extraction
- Validation for missing and low-confidence fields
- Correction-ready OCR result structure

## Architecture

React → FastAPI → Celery → PaddleOCR
↓
Redis

## Local Development

Installation instructions will be added here.

## Privacy

The repository contains only synthetic sample documents.
