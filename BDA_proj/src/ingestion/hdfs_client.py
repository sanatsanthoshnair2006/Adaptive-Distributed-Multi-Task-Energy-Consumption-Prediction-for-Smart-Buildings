"""
HDFS Client Module
===================

Production-quality HDFS client wrapping PyArrow's HDFS interface.
Provides all filesystem operations needed by the pipeline:
connect, create directories, upload, download, delete, copy,
move, check existence, and list contents.

Design Decisions:
    - Open/Closed Principle: All storage operations are abstracted
      behind this single class. Swapping storage backends requires
      only replacing this module.
    - All operations include retry logic for transient failures.
    - Every operation is logged for observability.
    - Connection is established lazily on first use and cached.
    - LOCAL MODE: When ``local_mode: true`` in hadoop_config.yaml,
      all HDFS paths are transparently mapped to a local directory
      (``local_base_path``) so the pipeline runs without a live
      Hadoop cluster. This is the recommended mode for development
      and single-node runs.

Migration Note (PyArrow >= 15):
    The legacy ``pyarrow.hdfs`` module has been removed.  This
    implementation uses the modern ``pyarrow.fs.HadoopFileSystem``
    API exclusively.  Key API differences:
        - ``mkdir``         → ``create_dir``
        - ``ls``            → ``get_file_info`` + ``FileSelector``
        - ``open(path,mode)``→ ``open_input_stream`` / ``open_output_stream``
        - ``info``          → ``get_file_info``
        - ``delete``        → ``delete_file`` / ``delete_dir``
        - ``rename``        → ``move``
        - No ``close()`` — the modern filesystem is not closeable.
"""

import os
import shutil
import time
from typing import Any, Dict, List, Optional

from src.logging.logger_factory import LoggerFactory
from src.utils.hdfs_utils import hdfs_path_join


class HDFSConnectionError(Exception):
    """Raised when HDFS connection cannot be established."""
    pass


class HDFSOperationError(Exception):
    """Raised when an HDFS operation fails after all retries."""
    pass


# ---------------------------------------------------------------------------
# Local filesystem backend (mirrors the HDFS interface for development)
# ---------------------------------------------------------------------------

class _LocalFSBackend:
    """
    A thin shim that maps HDFS-style paths to local filesystem paths
    so the pipeline can run without a real Hadoop cluster.

    HDFS paths like ``/user/energy_prediction/raw`` are mapped to
    ``<local_base_path>/user/energy_prediction/raw`` on disk.
    """

    def __init__(self, local_base_path: str) -> None:
        self._base = os.path.abspath(local_base_path)
        os.makedirs(self._base, exist_ok=True)

    def _local(self, hdfs_path: str) -> str:
        """Convert an HDFS-style absolute path to a local filesystem path."""
        # Strip leading slash so os.path.join works correctly
        rel = hdfs_path.lstrip("/").lstrip("\\")
        return os.path.join(self._base, rel)

    # ---- directory -------------------------------------------------------

    def create_dir(self, hdfs_path: str, recursive: bool = True) -> None:
        os.makedirs(self._local(hdfs_path), exist_ok=True)

    def delete_dir(self, hdfs_path: str) -> None:
        local = self._local(hdfs_path)
        if os.path.isdir(local):
            shutil.rmtree(local)

    # ---- file ------------------------------------------------------------

    def delete_file(self, hdfs_path: str) -> None:
        local = self._local(hdfs_path)
        if os.path.isfile(local):
            os.remove(local)

    def open_output_stream(self, hdfs_path: str):
        local = self._local(hdfs_path)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        return open(local, "wb")

    def open_input_stream(self, hdfs_path: str):
        return open(self._local(hdfs_path), "rb")

    def move(self, src: str, dst: str) -> None:
        local_src = self._local(src)
        local_dst = self._local(dst)
        os.makedirs(os.path.dirname(local_dst), exist_ok=True)
        shutil.move(local_src, local_dst)

    def copy_file(self, src: str, dst: str) -> None:
        local_src = self._local(src)
        local_dst = self._local(dst)
        os.makedirs(os.path.dirname(local_dst), exist_ok=True)
        shutil.copy2(local_src, local_dst)

    # ---- info / exists ---------------------------------------------------

    def get_file_info(self, hdfs_path):
        """Return a simple info object compatible with pyarrow FileInfo."""
        import types
        local = self._local(hdfs_path)
        info = types.SimpleNamespace()
        info.path = hdfs_path
        if os.path.isdir(local):
            info.type = _FileType.Directory
            info.size = 0
            info.mtime_ns = int(os.path.getmtime(local) * 1e9)
        elif os.path.isfile(local):
            info.type = _FileType.File
            info.size = os.path.getsize(local)
            info.mtime_ns = int(os.path.getmtime(local) * 1e9)
        else:
            info.type = _FileType.NotFound
            info.size = 0
            info.mtime_ns = 0
        return info

    def list_dir(self, hdfs_path: str) -> List[str]:
        local = self._local(hdfs_path)
        if not os.path.isdir(local):
            return []
        return [
            hdfs_path.rstrip("/") + "/" + name
            for name in os.listdir(local)
        ]


class _FileType:
    """Mirrors pyarrow.fs.FileType for local mode."""
    File = "File"
    Directory = "Directory"
    NotFound = "NotFound"


# ---------------------------------------------------------------------------
# Main HDFS Client
# ---------------------------------------------------------------------------

class HDFSClient:
    """
    Reusable HDFS client for distributed file system operations.

    Wraps PyArrow's HadoopFileSystem to provide a clean interface
    with retry logic, logging, and error handling.

    When ``local_mode: true`` is set in ``hadoop_config.yaml``, all
    operations transparently use the local filesystem instead of HDFS.

    Attributes:
        _namenode_uri: HDFS NameNode URI (e.g., hdfs://localhost:9000).
        _base_path: Base directory for all project data in HDFS.
        _replication_factor: HDFS replication factor.
        _max_retries: Maximum number of retry attempts.
        _retry_delay: Seconds between retry attempts.
        _fs: PyArrow HadoopFileSystem instance (lazy-initialized).
        _local_mode: If True, use local filesystem backend.
    """

    def __init__(self, hadoop_config: Dict[str, Any]) -> None:
        """
        Initialize HDFSClient from Hadoop configuration.

        Args:
            hadoop_config: The 'hadoop' section from the merged config.
                           Expected structure:
                           {
                               'hdfs': {
                                   'namenode_uri': 'hdfs://localhost:9000',
                                   'base_path': '/user/energy_prediction',
                                   'paths': {...},
                                   'replication_factor': 1,
                                   'max_retries': 3,
                                   'retry_delay': 5,
                                   'local_mode': False,
                                   'local_base_path': 'hdfs_local',
                               }
                           }
        """
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)

        hdfs_cfg = hadoop_config.get("hdfs", {})
        self._namenode_uri: str = hdfs_cfg.get(
            "namenode_uri", "hdfs://localhost:9000"
        )
        self._base_path: str = hdfs_cfg.get(
            "base_path", "/user/energy_prediction"
        )
        self._replication_factor: int = hdfs_cfg.get("replication_factor", 1)
        self._connection_timeout: int = hdfs_cfg.get("connection_timeout", 30)
        self._max_retries: int = hdfs_cfg.get("max_retries", 3)
        self._retry_delay: int = hdfs_cfg.get("retry_delay", 5)
        self._hdfs_paths: Dict[str, str] = hdfs_cfg.get("paths", {})

        # Local mode configuration
        self._local_mode: bool = bool(hdfs_cfg.get("local_mode", False))
        _local_base_rel: str = hdfs_cfg.get("local_base_path", "hdfs_local")

        self._fs: Optional[Any] = None  # Lazy-initialized filesystem

        if self._local_mode:
            # Resolve local_base_path relative to project_root if needed
            # The config_loader injects project_root into the merged config,
            # but here we only get the hadoop section. Use CWD as fallback.
            project_root = hadoop_config.get("_project_root", os.getcwd())
            local_abs = os.path.join(project_root, _local_base_rel)
            self._local_backend = _LocalFSBackend(local_abs)
            self._fs = self._local_backend  # Use local backend immediately
            self._logger.info(
                "HDFSClient running in LOCAL MODE. "
                "Storage: %s", os.path.abspath(local_abs)
            )
        else:
            self._local_backend = None

        self._logger.info(
            "HDFSClient initialized: namenode=%s, base_path=%s, local_mode=%s",
            self._namenode_uri,
            self._base_path,
            self._local_mode,
        )

    # ----------------------------------------------------------------
    # Connection Management
    # ----------------------------------------------------------------

    def connect(self) -> None:
        """
        Establish connection to HDFS.

        In LOCAL MODE, this is a no-op — the local filesystem backend
        is already available.  In HDFS mode, uses PyArrow's
        ``pyarrow.fs.HadoopFileSystem`` with retries.

        Raises:
            HDFSConnectionError: If connection fails after all retries.
        """
        if self._local_mode:
            self._logger.debug("LOCAL MODE: connect() is a no-op.")
            return

        if self._fs is not None:
            self._logger.debug("HDFS connection already established.")
            return

        import pyarrow.fs as pafs

        for attempt in range(1, self._max_retries + 1):
            try:
                self._logger.info(
                    "Connecting to HDFS at %s (attempt %d/%d)...",
                    self._namenode_uri,
                    attempt,
                    self._max_retries,
                )

                # Parse host and port from URI
                host, port = self._parse_namenode_uri()

                self._fs = pafs.HadoopFileSystem(
                    host=host,
                    port=port,
                )

                self._logger.info(
                    "Successfully connected to HDFS at %s:%d", host, port
                )
                return

            except Exception as e:
                self._logger.warning(
                    "HDFS connection attempt %d/%d failed: %s",
                    attempt,
                    self._max_retries,
                    str(e),
                )
                if attempt < self._max_retries:
                    self._logger.info(
                        "Retrying in %d seconds...", self._retry_delay
                    )
                    time.sleep(self._retry_delay)

        error_msg = (
            f"Failed to connect to HDFS at {self._namenode_uri} "
            f"after {self._max_retries} attempts."
        )
        self._logger.error(error_msg)
        raise HDFSConnectionError(error_msg)

    def disconnect(self) -> None:
        """Close the HDFS connection if open.

        Note: The modern ``pyarrow.fs.HadoopFileSystem`` does not have
        an explicit ``close()`` method.  We simply discard the reference.
        In LOCAL MODE this is a no-op.
        """
        if self._local_mode:
            return
        if self._fs is not None:
            self._logger.info("HDFS connection released.")
            self._fs = None

    def _ensure_connected(self) -> None:
        """Ensure filesystem is available; connect if not."""
        if self._local_mode:
            return  # always available
        if self._fs is None:
            self.connect()

    def _parse_namenode_uri(self) -> tuple:
        """
        Parse host and port from the NameNode URI.

        Returns:
            Tuple of (host: str, port: int).
        """
        uri = self._namenode_uri
        # Remove hdfs:// prefix
        if uri.startswith("hdfs://"):
            uri = uri[7:]
        # Split host:port
        if ":" in uri:
            host, port_str = uri.rsplit(":", 1)
            port = int(port_str)
        else:
            host = uri
            port = 9000  # Default HDFS port
        return host, port

    def _get_file_type_notfound(self):
        """Return the NotFound sentinel compatible with the current backend."""
        if self._local_mode:
            return _FileType.NotFound
        import pyarrow.fs as pafs
        return pafs.FileType.NotFound

    def _get_file_type_directory(self):
        """Return the Directory sentinel compatible with the current backend."""
        if self._local_mode:
            return _FileType.Directory
        import pyarrow.fs as pafs
        return pafs.FileType.Directory

    def _get_file_type_file(self):
        """Return the File sentinel compatible with the current backend."""
        if self._local_mode:
            return _FileType.File
        import pyarrow.fs as pafs
        return pafs.FileType.File

    # ----------------------------------------------------------------
    # Directory Operations
    # ----------------------------------------------------------------

    def create_directory(self, hdfs_path: str) -> bool:
        """
        Create a directory in HDFS (recursive, like mkdir -p).

        Args:
            hdfs_path: HDFS directory path to create.

        Returns:
            True if directory was created or already exists.

        Raises:
            HDFSOperationError: If directory creation fails.
        """
        self._ensure_connected()
        return self._retry_operation(
            operation_name=f"create_directory({hdfs_path})",
            func=self._do_create_directory,
            hdfs_path=hdfs_path,
        )

    def _do_create_directory(self, hdfs_path: str) -> bool:
        """Internal: create directory in HDFS."""
        if self.exists(hdfs_path):
            self._logger.debug(
                "Directory already exists: %s", hdfs_path
            )
            return True

        self._fs.create_dir(hdfs_path, recursive=True)
        self._logger.info("Created directory: %s", hdfs_path)
        return True

    def list_directory(self, hdfs_path: str) -> List[str]:
        """
        List contents of an HDFS directory.

        Args:
            hdfs_path: HDFS directory path to list.

        Returns:
            List of file/directory paths in the specified path.

        Raises:
            HDFSOperationError: If listing fails.
        """
        self._ensure_connected()
        return self._retry_operation(
            operation_name=f"list_directory({hdfs_path})",
            func=self._do_list_directory,
            hdfs_path=hdfs_path,
        )

    def _do_list_directory(self, hdfs_path: str) -> List[str]:
        """Internal: list directory contents."""
        if self._local_mode:
            return self._fs.list_dir(hdfs_path)
        import pyarrow.fs as pafs
        selector = pafs.FileSelector(hdfs_path, recursive=False)
        file_infos = self._fs.get_file_info(selector)
        contents = [fi.path for fi in file_infos]
        self._logger.debug(
            "Listed directory %s: %d items", hdfs_path, len(contents)
        )
        return contents

    # ----------------------------------------------------------------
    # File Operations
    # ----------------------------------------------------------------

    def upload_file(
        self, local_path: str, hdfs_path: str, overwrite: bool = False
    ) -> bool:
        """
        Upload a local file to HDFS (or local storage in local_mode).

        Args:
            local_path: Absolute path to the local file.
            hdfs_path: Destination path in HDFS.
            overwrite: If True, overwrite existing file.

        Returns:
            True if upload succeeded.

        Raises:
            FileNotFoundError: If local file does not exist.
            HDFSOperationError: If upload fails after retries.
        """
        if not os.path.isfile(local_path):
            raise FileNotFoundError(
                f"Local file not found: {local_path}"
            )

        if not overwrite and self.exists(hdfs_path):
            self._logger.info(
                "File already exists (skipping upload): %s", hdfs_path
            )
            return True

        self._ensure_connected()
        return self._retry_operation(
            operation_name=f"upload_file({local_path} → {hdfs_path})",
            func=self._do_upload_file,
            local_path=local_path,
            hdfs_path=hdfs_path,
        )

    def _do_upload_file(self, local_path: str, hdfs_path: str) -> bool:
        """Internal: upload file to storage."""
        # Ensure parent directory exists
        parent_dir = os.path.dirname(hdfs_path)
        if parent_dir and parent_dir != "/":
            self._do_create_directory(parent_dir)

        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        self._logger.info(
            "Uploading file: %s → %s (%.2f MB)",
            local_path,
            hdfs_path,
            file_size_mb,
        )

        with open(local_path, "rb") as local_file:
            with self._fs.open_output_stream(hdfs_path) as hdfs_file:
                # Read and write in chunks for large files
                chunk_size = 64 * 1024 * 1024  # 64 MB chunks
                while True:
                    chunk = local_file.read(chunk_size)
                    if not chunk:
                        break
                    hdfs_file.write(chunk)

        self._logger.info(
            "Successfully uploaded: %s (%.2f MB)",
            hdfs_path,
            file_size_mb,
        )
        return True

    def download_file(self, hdfs_path: str, local_path: str) -> bool:
        """
        Download a file from HDFS to local filesystem.

        Args:
            hdfs_path: Source path in HDFS.
            local_path: Destination path on local filesystem.

        Returns:
            True if download succeeded.

        Raises:
            HDFSOperationError: If the file does not exist or download fails.
        """
        self._ensure_connected()
        return self._retry_operation(
            operation_name=f"download_file({hdfs_path} → {local_path})",
            func=self._do_download_file,
            hdfs_path=hdfs_path,
            local_path=local_path,
        )

    def _do_download_file(self, hdfs_path: str, local_path: str) -> bool:
        """Internal: download file from storage."""
        if not self.exists(hdfs_path):
            raise HDFSOperationError(
                f"File does not exist: {hdfs_path}"
            )

        # Ensure local parent directory exists
        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)

        self._logger.info(
            "Downloading: %s → %s", hdfs_path, local_path
        )

        with self._fs.open_input_stream(hdfs_path) as hdfs_file:
            with open(local_path, "wb") as local_file:
                chunk_size = 64 * 1024 * 1024  # 64 MB chunks
                while True:
                    chunk = hdfs_file.read(chunk_size)
                    if not chunk:
                        break
                    local_file.write(chunk)

        self._logger.info("Successfully downloaded from: %s", hdfs_path)
        return True

    def delete(self, hdfs_path: str, recursive: bool = False) -> bool:
        """
        Delete a file or directory from HDFS.

        Args:
            hdfs_path: Path in HDFS to delete.
            recursive: If True, delete directory contents recursively.

        Returns:
            True if deletion succeeded.

        Raises:
            HDFSOperationError: If deletion fails.
        """
        self._ensure_connected()
        return self._retry_operation(
            operation_name=f"delete({hdfs_path}, recursive={recursive})",
            func=self._do_delete,
            hdfs_path=hdfs_path,
            recursive=recursive,
        )

    def _do_delete(self, hdfs_path: str, recursive: bool = False) -> bool:
        """Internal: delete file or directory from storage."""
        if not self.exists(hdfs_path):
            self._logger.warning(
                "Path does not exist (nothing to delete): %s", hdfs_path
            )
            return True

        file_info = self._fs.get_file_info(hdfs_path)

        if file_info.type == self._get_file_type_directory():
            self._fs.delete_dir(hdfs_path)
        else:
            self._fs.delete_file(hdfs_path)

        self._logger.info(
            "Deleted: %s (recursive=%s)", hdfs_path, recursive
        )
        return True

    def exists(self, hdfs_path: str) -> bool:
        """
        Check if a path exists in HDFS (or local storage).

        Args:
            hdfs_path: Path to check.

        Returns:
            True if the path exists.
        """
        self._ensure_connected()
        try:
            info = self._fs.get_file_info(hdfs_path)
            return info.type != self._get_file_type_notfound()
        except Exception:
            return False

    def copy_file(
        self, source_path: str, dest_path: str, overwrite: bool = False
    ) -> bool:
        """
        Copy a file within HDFS.

        Uses ``pyarrow.fs.copy_files`` for HDFS mode, or shutil for local.

        Args:
            source_path: Source HDFS path.
            dest_path: Destination HDFS path.
            overwrite: If True, overwrite existing destination.

        Returns:
            True if copy succeeded.

        Raises:
            HDFSOperationError: If copy fails.
        """
        self._ensure_connected()
        return self._retry_operation(
            operation_name=f"copy_file({source_path} → {dest_path})",
            func=self._do_copy_file,
            source_path=source_path,
            dest_path=dest_path,
            overwrite=overwrite,
        )

    def _do_copy_file(
        self, source_path: str, dest_path: str, overwrite: bool = False
    ) -> bool:
        """Internal: copy file within storage."""
        if not self.exists(source_path):
            raise HDFSOperationError(
                f"Source path does not exist: {source_path}"
            )

        if not overwrite and self.exists(dest_path):
            self._logger.info(
                "Destination already exists (skipping copy): %s", dest_path
            )
            return True

        # Ensure destination parent directory exists
        parent_dir = os.path.dirname(dest_path)
        if parent_dir and parent_dir != "/":
            self._do_create_directory(parent_dir)

        self._logger.info(
            "Copying: %s → %s", source_path, dest_path
        )

        self._fs.copy_file(source_path, dest_path)

        self._logger.info("Successfully copied: %s → %s", source_path, dest_path)
        return True

    def move_file(self, source_path: str, dest_path: str) -> bool:
        """
        Move (rename) a file or directory within HDFS.

        Args:
            source_path: Source HDFS path.
            dest_path: Destination HDFS path.

        Returns:
            True if move succeeded.

        Raises:
            HDFSOperationError: If move fails.
        """
        self._ensure_connected()
        return self._retry_operation(
            operation_name=f"move_file({source_path} → {dest_path})",
            func=self._do_move_file,
            source_path=source_path,
            dest_path=dest_path,
        )

    def _do_move_file(self, source_path: str, dest_path: str) -> bool:
        """Internal: move file within storage."""
        if not self.exists(source_path):
            raise HDFSOperationError(
                f"Source path does not exist: {source_path}"
            )

        # Ensure destination parent directory exists
        parent_dir = os.path.dirname(dest_path)
        if parent_dir and parent_dir != "/":
            self._do_create_directory(parent_dir)

        self._logger.info(
            "Moving: %s → %s", source_path, dest_path
        )
        self._fs.move(source_path, dest_path)
        self._logger.info("Successfully moved: %s → %s", source_path, dest_path)
        return True

    def get_file_info(self, hdfs_path: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a file or directory.

        Args:
            hdfs_path: Path to inspect.

        Returns:
            Dictionary with file metadata (size, type, path,
            mtime_ns) or None if path does not exist.
        """
        self._ensure_connected()
        try:
            info = self._fs.get_file_info(hdfs_path)
            if info.type == self._get_file_type_notfound():
                return None
            return {
                "path": info.path,
                "size": info.size,
                "type": "file" if info.type == self._get_file_type_file() else "directory",
                "mtime_ns": info.mtime_ns,
            }
        except Exception:
            return None

    # ----------------------------------------------------------------
    # Path Helpers
    # ----------------------------------------------------------------

    def get_hdfs_path(self, path_key: str) -> str:
        """
        Get a configured HDFS path by its key name.

        Args:
            path_key: Key from hadoop_config.yaml paths section
                      (e.g., 'raw_data', 'processed_data').

        Returns:
            The full HDFS path.

        Raises:
            KeyError: If path_key is not found in configuration.
        """
        if path_key not in self._hdfs_paths:
            raise KeyError(
                f"HDFS path key '{path_key}' not found in configuration. "
                f"Available keys: {list(self._hdfs_paths.keys())}"
            )
        return self._hdfs_paths[path_key]

    @property
    def base_path(self) -> str:
        """Return the HDFS base path."""
        return self._base_path

    @property
    def namenode_uri(self) -> str:
        """Return the HDFS NameNode URI."""
        return self._namenode_uri

    def spark_path(self, hdfs_path: str) -> str:
        """
        Convert an HDFS-style logical path to a Spark-readable URI.

        In LOCAL MODE:  ``/user/energy_prediction/features/v1``
                        → ``file:///C:/project/hdfs_local/user/energy_prediction/features/v1``

        In HDFS MODE:   ``/user/energy_prediction/features/v1``
                        → ``hdfs://localhost:9000/user/energy_prediction/features/v1``

        Args:
            hdfs_path: Logical HDFS-style path (starts with /).

        Returns:
            Spark-readable URI string.
        """
        if self._local_mode:
            rel = hdfs_path.lstrip("/").lstrip("\\")
            local = os.path.join(self._local_backend._base, rel)
            # Windows: use forward slashes for Spark file:// URI
            return "file:///" + local.replace("\\", "/")
        else:
            namenode = self._namenode_uri.rstrip("/")
            return f"{namenode}/{hdfs_path.lstrip('/')}"

    # ----------------------------------------------------------------
    # HDFS Directory Setup
    # ----------------------------------------------------------------

    def setup_hdfs_directories(self) -> None:
        """
        Create all configured HDFS directories.

        Reads paths from hadoop_config.yaml and ensures each exists.
        """
        self._logger.info("Setting up directory structure...")

        # Create base path
        self.create_directory(self._base_path)

        # Create all configured sub-paths
        for path_key, path_value in self._hdfs_paths.items():
            self.create_directory(path_value)
            self._logger.info(
                "Directory ready: %s → %s", path_key, path_value
            )

        self._logger.info("Directory structure setup complete.")

    # ----------------------------------------------------------------
    # Retry Logic
    # ----------------------------------------------------------------

    def _retry_operation(
        self, operation_name: str, func: callable, **kwargs: Any
    ) -> Any:
        """
        Execute an HDFS operation with retry logic.

        Args:
            operation_name: Human-readable name for logging.
            func: The function to execute.
            **kwargs: Arguments to pass to the function.

        Returns:
            The return value of the function.

        Raises:
            HDFSOperationError: If operation fails after all retries.
        """
        last_exception = None

        for attempt in range(1, self._max_retries + 1):
            try:
                return func(**kwargs)
            except HDFSOperationError:
                # Re-raise domain errors immediately (no retry)
                raise
            except Exception as e:
                last_exception = e
                self._logger.warning(
                    "Operation '%s' failed (attempt %d/%d): %s",
                    operation_name,
                    attempt,
                    self._max_retries,
                    str(e),
                )
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay)

        error_msg = (
            f"Operation '{operation_name}' failed after "
            f"{self._max_retries} attempts. Last error: {last_exception}"
        )
        self._logger.error(error_msg)
        raise HDFSOperationError(error_msg)

    # ----------------------------------------------------------------
    # Context Manager
    # ----------------------------------------------------------------

    def __enter__(self) -> "HDFSClient":
        """Support usage as context manager."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Disconnect on context manager exit."""
        self.disconnect()

    def __repr__(self) -> str:
        return (
            f"HDFSClient(namenode='{self._namenode_uri}', "
            f"base_path='{self._base_path}', "
            f"local_mode={self._local_mode})"
        )
