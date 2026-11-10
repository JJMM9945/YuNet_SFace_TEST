"""Archived lab; share the selected core without duplicating its implementation."""
from pathlib import Path
from face_lab import VERSION, PREPROCESS_VERSION, GROUPING_VERSION

__path__.append(str(Path(__file__).resolve().parents[3] / 'face_lab'))
