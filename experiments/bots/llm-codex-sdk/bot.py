#!/usr/bin/env python3
"""Plays every decision through the Codex SDK. See experiments/bots/llm_play.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_play import main  # noqa: E402

main("codex-sdk")
