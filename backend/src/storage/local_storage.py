"""Local filesystem implementation for image storage.

This is useful for development and testing without MinIO.
"""

import logging
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .interfaces import ImageStorage

logger = logging.getLogger(__name__)


class LocalStorage(ImageStorage):
    """Local filesystem implementation for image storage.
    
    Stores images in a local directory. Useful for development
    and testing without setting up MinIO.
    
    Example:
        >>> storage = LocalStorage("./data/images")
        >>> path = storage.save(image, "2025/11/26/frame_001.jpg")
        >>> loaded = storage.load(path)
    """

    def __init__(self, base_path: str = "./data/images"):
        """Initialize local storage.
        
        Args:
            base_path: Base directory for storing images.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalStorage initialized: {self.base_path.absolute()}")

    def save(self, image: np.ndarray, path: str) -> str:
        """Save image to local filesystem.
        
        Args:
            image: Image as numpy array (BGR format).
            path: Relative path within base directory.
            
        Returns:
            Full path to saved file.
        """
        full_path = self.base_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        success = cv2.imwrite(str(full_path), image)
        if not success:
            raise RuntimeError(f"Failed to save image: {full_path}")
        
        logger.debug(f"Saved image: {full_path}")
        return str(full_path)

    def save_bytes(self, data: bytes, path: str, content_type: str = "image/jpeg") -> str:
        """Save raw bytes to local filesystem.
        
        Args:
            data: Raw binary data.
            path: Relative path within base directory.
            content_type: MIME type (ignored for local storage).
            
        Returns:
            Full path to saved file.
        """
        full_path = self.base_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, "wb") as f:
            f.write(data)
        
        logger.debug(f"Saved file: {full_path}")
        return str(full_path)

    def load(self, path: str) -> np.ndarray:
        """Load image from local filesystem.
        
        Args:
            path: Path to image file.
            
        Returns:
            Image as numpy array (BGR format).
        """
        # Handle both absolute and relative paths
        if os.path.isabs(path):
            full_path = Path(path)
        else:
            full_path = self.base_path / path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")
        
        image = cv2.imread(str(full_path))
        if image is None:
            raise RuntimeError(f"Failed to load image: {full_path}")
        
        return image

    def load_bytes(self, path: str) -> bytes:
        """Load raw bytes from local filesystem.
        
        Args:
            path: Path to file.
            
        Returns:
            Raw binary data.
        """
        if os.path.isabs(path):
            full_path = Path(path)
        else:
            full_path = self.base_path / path
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")
        
        with open(full_path, "rb") as f:
            return f.read()

    def delete(self, path: str) -> bool:
        """Delete file from local filesystem.
        
        Args:
            path: Path to file.
            
        Returns:
            True if deleted, False if not found.
        """
        if os.path.isabs(path):
            full_path = Path(path)
        else:
            full_path = self.base_path / path
        
        if not full_path.exists():
            return False
        
        full_path.unlink()
        logger.debug(f"Deleted file: {full_path}")
        return True

    def exists(self, path: str) -> bool:
        """Check if file exists.
        
        Args:
            path: Path to file.
            
        Returns:
            True if exists.
        """
        if os.path.isabs(path):
            full_path = Path(path)
        else:
            full_path = self.base_path / path
        
        return full_path.exists()

    def list_objects(self, prefix: str = "") -> List[str]:
        """List files with given prefix.
        
        Args:
            prefix: Path prefix to filter.
            
        Returns:
            List of relative file paths.
        """
        search_path = self.base_path / prefix
        
        if not search_path.exists():
            return []
        
        if search_path.is_file():
            return [prefix]
        
        # Recursively find all files
        files = []
        for path in search_path.rglob("*"):
            if path.is_file():
                relative = path.relative_to(self.base_path)
                # Use as_posix() to ensure consistent forward slashes on all platforms
                files.append(relative.as_posix())
        
        return sorted(files)


