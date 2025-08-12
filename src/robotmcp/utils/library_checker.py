"""Utility for checking library availability before installation."""

import importlib
import subprocess
import sys
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class LibraryAvailabilityChecker:
    """Checks if libraries are available before attempting installation."""
    
    def __init__(self):
        self.checked_libraries = {}  # Cache for checked libraries
        
    def is_library_available(self, library_name: str, import_name: str = None) -> bool:
        """
        Check if a library is available for import.
        
        Args:
            library_name: The pip package name (e.g., 'robotframework-browser')
            import_name: The import name (e.g., 'Browser'). If None, uses library_name
            
        Returns:
            bool: True if library can be imported, False otherwise
        """
        if not import_name:
            import_name = library_name
            
        # Check cache first
        cache_key = f"{library_name}:{import_name}"
        if cache_key in self.checked_libraries:
            return self.checked_libraries[cache_key]
        
        try:
            importlib.import_module(import_name)
            logger.info(f"Library '{library_name}' is available (import: {import_name})")
            self.checked_libraries[cache_key] = True
            return True
        except ImportError:
            logger.info(f"Library '{library_name}' is not available (import: {import_name})")
            self.checked_libraries[cache_key] = False
            return False
    
    def is_robot_library_available(self, library_name: str) -> bool:
        """Check if a Robot Framework library is available."""
        try:
            # Try importing as Robot Framework library
            importlib.import_module(library_name)
            return True
        except ImportError:
            # Try common Robot Framework library patterns
            common_patterns = [
                f"{library_name}Library",
                f"robot.libraries.{library_name}",
                f"robotframework_{library_name.lower()}",
            ]
            
            for pattern in common_patterns:
                try:
                    importlib.import_module(pattern)
                    return True
                except ImportError:
                    continue
            
            return False
    
    def check_multiple_libraries(self, libraries: List[Dict[str, str]]) -> Dict[str, bool]:
        """
        Check availability of multiple libraries.
        
        Args:
            libraries: List of dicts with 'package' and 'import' keys
            
        Returns:
            Dict mapping library names to availability status
        """
        results = {}
        for lib in libraries:
            package = lib.get('package')
            import_name = lib.get('import', package)
            results[package] = self.is_library_available(package, import_name)
        
        return results
    
    def get_missing_libraries(self, libraries: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Get list of libraries that are not available."""
        missing = []
        for lib in libraries:
            package = lib.get('package')
            import_name = lib.get('import', package)
            if not self.is_library_available(package, import_name):
                missing.append(lib)
        return missing
    
    def check_pip_package_installed(self, package_name: str) -> bool:
        """Check if a pip package is installed using pip list."""
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'list', '--format=freeze'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                installed_packages = result.stdout.lower()
                return package_name.lower() in installed_packages
            else:
                logger.warning(f"Failed to check pip packages: {result.stderr}")
                return False
                
        except Exception as e:
            logger.warning(f"Error checking pip packages: {e}")
            return False
    
    def get_installation_command(self, library_info: Dict[str, str]) -> List[str]:
        """Get the pip install command for a library."""
        package = library_info.get('package')
        version = library_info.get('version', '')
        
        if version:
            package_spec = f"{package}=={version}"
        else:
            package_spec = package
            
        return [sys.executable, '-m', 'pip', 'install', package_spec]
    
    def suggest_installation(self, library_info: Dict[str, str]) -> str:
        """Generate installation suggestion message."""
        package = library_info.get('package')
        import_name = library_info.get('import', package)
        description = library_info.get('description', '')
        
        cmd = ' '.join(self.get_installation_command(library_info))
        
        msg = f"Library '{import_name}' is not available"
        if description:
            msg += f" ({description})"
        msg += f".\nTo install it, run:\n  {cmd}"
        
        return msg

# Import library configurations from centralized registry
from robotmcp.config.library_registry import get_installation_info

# Get library configurations from central registry
COMMON_ROBOT_LIBRARIES = get_installation_info()

def check_and_suggest_libraries(required_libraries: List[str]) -> Tuple[List[str], List[str]]:
    """
    Check which libraries are available and suggest installations for missing ones.
    
    Args:
        required_libraries: List of library names to check
        
    Returns:
        Tuple of (available_libraries, installation_suggestions)
    """
    checker = LibraryAvailabilityChecker()
    available = []
    suggestions = []
    
    for lib_name in required_libraries:
        if lib_name in COMMON_ROBOT_LIBRARIES:
            lib_info = COMMON_ROBOT_LIBRARIES[lib_name]
            if checker.is_library_available(lib_info['package'], lib_info['import']):
                available.append(lib_name)
            else:
                suggestion = checker.suggest_installation(lib_info)
                if lib_info.get('post_install'):
                    suggestion += f"\nAfter installation, run: {lib_info['post_install']}"
                suggestions.append(suggestion)
        else:
            # Try to check the library directly
            if checker.is_robot_library_available(lib_name):
                available.append(lib_name)
            else:
                suggestions.append(f"Library '{lib_name}' not found. Manual installation may be required.")
    
    return available, suggestions