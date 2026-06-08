"""MinIO implementation for image storage."""

import io
import logging
from typing import List

import cv2
import numpy as np
from minio import Minio
from minio.error import S3Error

from .interfaces import ImageStorage

logger = logging.getLogger(__name__)


class MinIOStorage(ImageStorage):
    """MinIO implementation for image storage.
    
    Uses MinIO (S3-compatible) for storing images and other binary data.
    
    Example:
        >>> storage = MinIOStorage(
        ...     endpoint="localhost:9000",
        ...     access_key="minioadmin",
        ...     secret_key="minioadmin",
        ...     bucket="pluck-images"
        ... )
        >>> path = storage.save(image, "2025/11/26/frame_001.jpg")
        >>> loaded = storage.load(path)
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str = "pluck-images",
        secure: bool = False,
        auto_create_bucket: bool = True,
    ):
        """Initialize MinIO storage.
        
        Args:
            endpoint: MinIO server endpoint (host:port).
            access_key: Access key (username).
            secret_key: Secret key (password).
            bucket: Bucket name to use.
            secure: Use HTTPS connection.
            auto_create_bucket: Create bucket if not exists.
        """
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self.bucket = bucket
        
        if auto_create_bucket:
            self._ensure_bucket()

    def _ensure_bucket(self):
        """Create bucket if not exists."""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"Created bucket: {self.bucket}")
            else:
                logger.debug(f"Bucket exists: {self.bucket}")
        except S3Error as e:
            logger.error(f"Failed to create bucket: {e}")
            raise

    def save(self, image: np.ndarray, path: str) -> str:
        """Save image to MinIO.
        
        Args:
            image: Image as numpy array (BGR format).
            path: Object path in bucket.
            
        Returns:
            Full storage path (bucket/path).
        """
        # Encode image to JPEG
        success, buffer = cv2.imencode(".jpg", image)
        if not success:
            raise RuntimeError("Failed to encode image to JPEG")
        
        return self.save_bytes(buffer.tobytes(), path, "image/jpeg")

    def save_bytes(self, data: bytes, path: str, content_type: str = "image/jpeg") -> str:
        """Save raw bytes to MinIO.
        
        Args:
            data: Raw binary data.
            path: Object path in bucket.
            content_type: MIME type.
            
        Returns:
            Full storage path.
        """
        try:
            data_io = io.BytesIO(data)
            size = len(data)
            
            self.client.put_object(
                self.bucket,
                path,
                data_io,
                size,
                content_type=content_type,
            )
            
            full_path = f"{self.bucket}/{path}"
            logger.debug(f"Saved object: {full_path} ({size} bytes)")
            return full_path
            
        except S3Error as e:
            logger.error(f"Failed to save object: {e}")
            raise

    def load(self, path: str) -> np.ndarray:
        """Load image from MinIO.
        
        Args:
            path: Object path (can include bucket prefix).
            
        Returns:
            Image as numpy array (BGR format).
        """
        data = self.load_bytes(path)
        
        # Decode image
        np_arr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise RuntimeError(f"Failed to decode image: {path}")
        
        return image

    def load_bytes(self, path: str) -> bytes:
        """Load raw bytes from MinIO.
        
        Args:
            path: Object path.
            
        Returns:
            Raw binary data.
        """
        object_name = self._normalize_path(path)
        
        try:
            response = self.client.get_object(self.bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            
            logger.debug(f"Loaded object: {object_name} ({len(data)} bytes)")
            return data
            
        except S3Error as e:
            if e.code == "NoSuchKey":
                raise FileNotFoundError(f"Object not found: {path}")
            logger.error(f"Failed to load object: {e}")
            raise

    def delete(self, path: str) -> bool:
        """Delete object from MinIO.
        
        Args:
            path: Object path.
            
        Returns:
            True if deleted, False if not found.
        """
        object_name = self._normalize_path(path)
        
        try:
            self.client.remove_object(self.bucket, object_name)
            logger.debug(f"Deleted object: {object_name}")
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            logger.error(f"Failed to delete object: {e}")
            raise

    def exists(self, path: str) -> bool:
        """Check if object exists in MinIO.
        
        Args:
            path: Object path.
            
        Returns:
            True if exists.
        """
        object_name = self._normalize_path(path)
        
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            raise

    def list_objects(self, prefix: str = "") -> List[str]:
        """List objects with given prefix.
        
        Args:
            prefix: Path prefix to filter.
            
        Returns:
            List of object paths.
        """
        try:
            objects = self.client.list_objects(
                self.bucket,
                prefix=prefix,
                recursive=True
            )
            return [obj.object_name for obj in objects]
        except S3Error as e:
            logger.error(f"Failed to list objects: {e}")
            raise

    def _normalize_path(self, path: str) -> str:
        """Normalize path by removing bucket prefix if present."""
        bucket_prefix = f"{self.bucket}/"
        if path.startswith(bucket_prefix):
            return path[len(bucket_prefix):]
        return path

    def get_presigned_url(self, path: str, expires_hours: int = 24) -> str:
        """Get a presigned URL for an object.
        
        Args:
            path: Object path.
            expires_hours: URL expiration time in hours.
            
        Returns:
            Presigned URL string.
        """
        from datetime import timedelta
        
        object_name = self._normalize_path(path)
        
        try:
            url = self.client.presigned_get_object(
                self.bucket,
                object_name,
                expires=timedelta(hours=expires_hours)
            )
            return url
        except S3Error as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise


