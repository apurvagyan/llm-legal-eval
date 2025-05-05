# CPSC 477/577 (NLP) Final Project — Prompting the Judge: Evaluating Legal Summaries with LLM-Based Annotation Systems

This repository contains the full implementation and evaluation pipeline for our final project in CPSC 477/577 at Yale: **Prompting the Judge**, a system that uses large language models (LLMs) as automated evaluators ("LLM judges") to assess the quality of legal case summaries.

We explore prompting strategies to elicit structured, rubric-aligned annotations from LLMs and benchmark two models (Gemini 2.0 Flash and GPT-4o-mini) across base, improved, and one-shot prompt formats.

## Overview

The goal of this project is to create a scalable evaluation framework for legal summarization that avoids traditional, surface-level metrics like ROUGE or BLEU, and instead relies on LLMs for fine-grained legal annotation. Each system-generated summary is compared to a human reference using a structured rubric of 9 attributes (e.g., Plaintiff, Holding, Judge’s Name).

## Requirements

All experiments were conducted using Python 3.11 in a virtual environment.

**Install the minimal dependencies** via:

```bash
pip install -r requirements.txt
```

The minimal requirements are numpy==1.26.4, pandas==2.0.3, matplotlib==3.8.2, google-generativeai==0.8.5, openai==1.75.0.


## Setup

This code can be run locally on any device, as it queries APIs from OpenAI and Google's Gemini. Our experiments were run on Python 3.11, macOS Sonoma, with an Apple M2 chip and no fine-tuning or quantization.

You can run the evaluation pipeline via:

```bash
python meta_eval.py
```

Ensure you have API keys for OpenAI stored in `keys/openai.key` and `keys/openai.id`, and your key for Gemini stored in the `GOOGLE_API_KEY` environment variable.