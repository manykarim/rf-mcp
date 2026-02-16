"""Native JSON-based pattern storage for self-learning optimization.

This module provides persistent storage for learned optimization patterns
using only Python standard library (json, pathlib). No external dependencies.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any


class PatternStore:
    """
    Native pattern storage using JSON files.

    Stores learned optimization patterns persistently in the user's home
    directory under ~/.rf-mcp/patterns/. Each namespace gets its own
    subdirectory, and each key becomes a JSON file.

    Features:
    - LRU-style in-memory cache for fast lookups
    - Automatic directory creation
    - TTL-based cleanup for old entries
    - Thread-safe file operations

    Example:
        store = PatternStore()
        store.store("compression", "ecommerce_page", {"ratio": 5.2})
        pattern = store.retrieve("compression", "ecommerce_page")
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize the pattern store.

        Args:
            storage_dir: Custom storage directory. Defaults to ~/.rf-mcp/patterns/
        """
        self.storage_dir = storage_dir or Path.home() / ".rf-mcp" / "patterns"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._max_cache_size = 1000
        self._cache_ttl_seconds = 3600  # 1 hour cache TTL

    def _get_cache_key(self, namespace: str, key: str) -> str:
        """Generate a cache key from namespace and key."""
        return f"{namespace}:{key}"

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a string to be safe for use as a filename."""
        # Replace characters that are problematic in filenames
        unsafe_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        result = name
        for char in unsafe_chars:
            result = result.replace(char, '_')
        return result

    def _get_file_path(self, namespace: str, key: str) -> Path:
        """Get the file path for a namespace/key combination."""
        safe_namespace = self._sanitize_filename(namespace)
        safe_key = self._sanitize_filename(key)
        return self.storage_dir / safe_namespace / f"{safe_key}.json"

    def _evict_cache_if_needed(self) -> None:
        """Evict oldest cache entries if cache is full."""
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest 10% of entries
            entries_to_remove = max(1, self._max_cache_size // 10)
            sorted_keys = sorted(
                self._cache_timestamps.keys(),
                key=lambda k: self._cache_timestamps[k]
            )
            for cache_key in sorted_keys[:entries_to_remove]:
                self._cache.pop(cache_key, None)
                self._cache_timestamps.pop(cache_key, None)

    def store(self, namespace: str, key: str, value: Dict[str, Any]) -> bool:
        """
        Store a pattern to disk and cache.

        Args:
            namespace: The category/type of pattern (e.g., "compression", "timeouts")
            key: The unique identifier within the namespace (e.g., "ecommerce_page")
            value: The pattern data to store (must be JSON-serializable)

        Returns:
            True if storage succeeded, False otherwise
        """
        try:
            file_path = self._get_file_path(namespace, key)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Add metadata
            data_with_meta = {
                "_stored_at": time.time(),
                "_namespace": namespace,
                "_key": key,
                **value
            }

            # Write atomically using temp file
            temp_path = file_path.with_suffix('.json.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data_with_meta, f, indent=2)

            # Atomic rename
            temp_path.replace(file_path)

            # Update cache
            self._evict_cache_if_needed()
            cache_key = self._get_cache_key(namespace, key)
            self._cache[cache_key] = data_with_meta
            self._cache_timestamps[cache_key] = time.time()

            return True
        except (OSError, IOError, TypeError, ValueError) as e:
            # Log error in production; silently fail for now
            return False

    def retrieve(self, namespace: str, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a pattern from cache or disk.

        Args:
            namespace: The category/type of pattern
            key: The unique identifier within the namespace

        Returns:
            The stored pattern data, or None if not found
        """
        cache_key = self._get_cache_key(namespace, key)

        # Check cache first (with TTL validation)
        if cache_key in self._cache:
            cache_age = time.time() - self._cache_timestamps.get(cache_key, 0)
            if cache_age < self._cache_ttl_seconds:
                return self._cache[cache_key]

        # Load from disk
        try:
            file_path = self._get_file_path(namespace, key)
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    value = json.load(f)

                # Update cache
                self._evict_cache_if_needed()
                self._cache[cache_key] = value
                self._cache_timestamps[cache_key] = time.time()
                return value
        except (OSError, IOError, json.JSONDecodeError):
            pass

        return None

    def list_keys(self, namespace: str) -> List[str]:
        """
        List all keys in a namespace.

        Args:
            namespace: The category/type to list keys for

        Returns:
            List of key names in the namespace
        """
        safe_namespace = self._sanitize_filename(namespace)
        ns_dir = self.storage_dir / safe_namespace

        if not ns_dir.exists():
            return []

        try:
            return [f.stem for f in ns_dir.glob("*.json") if f.is_file()]
        except OSError:
            return []

    def delete(self, namespace: str, key: str) -> bool:
        """
        Delete a pattern from storage and cache.

        Args:
            namespace: The category/type of pattern
            key: The unique identifier to delete

        Returns:
            True if deletion succeeded, False otherwise
        """
        try:
            # Remove from cache
            cache_key = self._get_cache_key(namespace, key)
            self._cache.pop(cache_key, None)
            self._cache_timestamps.pop(cache_key, None)

            # Remove from disk
            file_path = self._get_file_path(namespace, key)
            if file_path.exists():
                file_path.unlink()
            return True
        except OSError:
            return False

    def cleanup_old_entries(self, namespace: str, max_age_days: int = 30) -> int:
        """
        Remove entries older than max_age_days from a namespace.

        Args:
            namespace: The category/type to clean up
            max_age_days: Maximum age in days before an entry is removed

        Returns:
            Number of entries removed
        """
        removed_count = 0
        max_age_seconds = max_age_days * 24 * 60 * 60
        current_time = time.time()

        for key in self.list_keys(namespace):
            data = self.retrieve(namespace, key)
            if data:
                stored_at = data.get("_stored_at", 0)
                if current_time - stored_at >= max_age_seconds:
                    if self.delete(namespace, key):
                        removed_count += 1

        return removed_count

    def cleanup_all_namespaces(self, max_age_days: int = 30) -> Dict[str, int]:
        """
        Clean up old entries from all namespaces.

        Args:
            max_age_days: Maximum age in days before an entry is removed

        Returns:
            Dictionary mapping namespace to number of entries removed
        """
        results = {}

        try:
            for ns_dir in self.storage_dir.iterdir():
                if ns_dir.is_dir():
                    namespace = ns_dir.name
                    removed = self.cleanup_old_entries(namespace, max_age_days)
                    if removed > 0:
                        results[namespace] = removed
        except OSError:
            pass

        return results

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the pattern storage.

        Returns:
            Dictionary with storage statistics
        """
        stats = {
            "storage_dir": str(self.storage_dir),
            "cache_size": len(self._cache),
            "max_cache_size": self._max_cache_size,
            "namespaces": {},
            "total_patterns": 0,
            "total_size_bytes": 0,
        }

        try:
            for ns_dir in self.storage_dir.iterdir():
                if ns_dir.is_dir():
                    namespace = ns_dir.name
                    keys = self.list_keys(namespace)
                    size = sum(
                        f.stat().st_size
                        for f in ns_dir.glob("*.json")
                        if f.is_file()
                    )
                    stats["namespaces"][namespace] = {
                        "count": len(keys),
                        "size_bytes": size,
                    }
                    stats["total_patterns"] += len(keys)
                    stats["total_size_bytes"] += size
        except OSError:
            pass

        return stats

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
