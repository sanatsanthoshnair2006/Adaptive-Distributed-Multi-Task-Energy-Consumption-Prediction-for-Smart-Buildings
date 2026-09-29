"""
HDFS Utilities Module
======================

Helper functions for safe HDFS path construction and validation.
"""

from src.logging.logger_factory import LoggerFactory


_logger = LoggerFactory.get_logger("utils.hdfs_utils")


def hdfs_path_join(*parts: str) -> str:
    """
    Join HDFS path components using forward slashes.
    Handles trailing/leading slashes correctly.

    Args:
        *parts: Path components to join.

    Returns:
        Joined HDFS path string.
    """
    cleaned = []
    for i, part in enumerate(parts):
        part = part.strip()
        if i > 0:
            part = part.lstrip("/")
        if i < len(parts) - 1:
            part = part.rstrip("/")
        if part:
            cleaned.append(part)
    return "/".join(cleaned)


def ensure_hdfs_path(hdfs_client: object, path: str) -> None:
    """
    Ensure that an HDFS directory exists by creating it if necessary.

    Args:
        hdfs_client: An HDFSClient instance with a create_directory method.
        path: HDFS directory path to ensure.
    """
    # The HDFSClient.create_directory method handles existence checks
    hdfs_client.create_directory(path)  # type: ignore
    _logger.debug("Ensured HDFS directory exists: %s", path)
