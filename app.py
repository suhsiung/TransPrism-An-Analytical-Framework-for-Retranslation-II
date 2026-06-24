#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py - Hugging Face Spaces entry point
Imports and launches the main application from app_agent.py
"""
import app_agent

if __name__ == "__main__":
    app = app_agent.create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        allowed_paths=[str(app_agent.SESSIONS_DIR)],
    )
