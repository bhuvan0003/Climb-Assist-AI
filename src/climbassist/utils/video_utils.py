"""Video I/O and temporary file management."""

import os
import tempfile


def save_temp_video(uploaded_file, filename: str = "temp_uploaded_video.mp4") -> str:
    """Save uploaded video bytes to a temporary file and return path."""
    video_bytes = uploaded_file.read()
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, filename)
    with open(temp_path, "wb") as f:
        f.write(video_bytes)
    return temp_path


def cleanup_temp_files(*filenames: str) -> None:
    """Remove temporary video files."""
    temp_dir = tempfile.gettempdir()
    for filename in filenames:
        path = os.path.join(temp_dir, filename)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
