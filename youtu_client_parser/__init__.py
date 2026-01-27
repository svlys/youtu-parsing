"""
ClientParser Module

Client-based OCR parser using OpenAI-compatible API endpoints.
This module provides remote inference capabilities through HTTP API calls.
"""

from .youtu_ocr_parser_client import YoutuOCRParserClient

__all__ = ['YoutuOCRParserClient']
