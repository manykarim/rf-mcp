"""Mobile capability service for Appium testing.

This service validates and prepares mobile capabilities for Appium,
checks server connectivity, and manages device configurations.
"""

import logging
import json
from typing import Dict, Any, Optional, List
import requests
from robotmcp.models.session_models import MobileConfig

logger = logging.getLogger(__name__)


class MobileCapabilityService:
    """Service for managing mobile testing capabilities and Appium configuration."""
    
    def __init__(self):
        """Initialize the mobile capability service."""
        self.default_timeout = 60
        self.default_appium_port = 4723
        
    def validate_capabilities(self, caps: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate Appium capabilities.
        
        Args:
            caps: Capability dictionary to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required capabilities
        if 'platformName' not in caps and 'appium:platformName' not in caps:
            errors.append("platformName is required")
            
        platform = caps.get('platformName') or caps.get('appium:platformName')
        
        if platform not in ['iOS', 'Android', None]:
            errors.append(f"Invalid platformName: {platform}. Must be 'iOS' or 'Android'")
            
        # Check for device identifier
        if not any(k in caps for k in ['deviceName', 'appium:deviceName', 
                                        'udid', 'appium:udid']):
            errors.append("deviceName or udid is required")
            
        # Platform-specific validation
        if platform == 'Android':
            # For Android, either app or appPackage/appActivity is needed
            has_app = any(k in caps for k in ['app', 'appium:app'])
            has_package = any(k in caps for k in ['appPackage', 'appium:appPackage'])
            
            if not has_app and not has_package:
                errors.append("Android requires either 'app' or 'appPackage/appActivity'")
                
        elif platform == 'iOS':
            # For iOS, either app or bundleId is needed
            has_app = any(k in caps for k in ['app', 'appium:app'])
            has_bundle = any(k in caps for k in ['bundleId', 'appium:bundleId'])
            
            if not has_app and not has_bundle:
                errors.append("iOS requires either 'app' or 'bundleId'")
                
        return len(errors) == 0, errors
    
    def prepare_android_caps(self, config: MobileConfig) -> Dict[str, Any]:
        """
        Prepare Android-specific capabilities.
        
        Args:
            config: Mobile configuration
            
        Returns:
            Dictionary of Android capabilities
        """
        caps = {
            'platformName': 'Android',
            'appium:automationName': config.automation_name or 'UiAutomator2',
            'appium:deviceName': config.device_name or 'Android Emulator'
        }
        
        # Add optional Android capabilities
        if config.app_path:
            caps['appium:app'] = config.app_path
        if config.app_package:
            caps['appium:appPackage'] = config.app_package
        if config.app_activity:
            caps['appium:appActivity'] = config.app_activity
        if config.device_udid:
            caps['appium:udid'] = config.device_udid
        if config.platform_version:
            caps['appium:platformVersion'] = config.platform_version
            
        # Reset options
        if config.no_reset:
            caps['appium:noReset'] = True
        if config.full_reset:
            caps['appium:fullReset'] = True
            
        # Additional Android options
        caps['appium:newCommandTimeout'] = self.default_timeout
        caps['appium:autoGrantPermissions'] = True
        
        return caps
    
    def prepare_ios_caps(self, config: MobileConfig) -> Dict[str, Any]:
        """
        Prepare iOS-specific capabilities.
        
        Args:
            config: Mobile configuration
            
        Returns:
            Dictionary of iOS capabilities
        """
        caps = {
            'platformName': 'iOS',
            'appium:automationName': config.automation_name or 'XCUITest',
            'appium:deviceName': config.device_name or 'iPhone Simulator'
        }
        
        # Add optional iOS capabilities
        if config.app_path:
            caps['appium:app'] = config.app_path
        if config.bundle_id:
            caps['appium:bundleId'] = config.bundle_id
        if config.device_udid:
            caps['appium:udid'] = config.device_udid
        if config.platform_version:
            caps['appium:platformVersion'] = config.platform_version
            
        # Reset options
        if config.no_reset:
            caps['appium:noReset'] = True
        if config.full_reset:
            caps['appium:fullReset'] = True
            
        # Additional iOS options
        caps['appium:newCommandTimeout'] = self.default_timeout
        caps['appium:autoAcceptAlerts'] = False
        
        return caps
    
    def check_appium_server(self, url: str = "http://127.0.0.1:4723") -> tuple[bool, Optional[Dict]]:
        """
        Check if Appium server is accessible.
        
        Args:
            url: Appium server URL
            
        Returns:
            Tuple of (is_accessible, server_info)
        """
        try:
            # Normalize URL
            if not url.startswith('http'):
                url = f'http://{url}'
            if not url.endswith('/'):
                url = f'{url}/'
                
            # Check server status
            response = requests.get(f'{url}status', timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                server_info = {
                    'ready': data.get('value', {}).get('ready', False),
                    'message': data.get('value', {}).get('message', ''),
                    'version': data.get('value', {}).get('build', {}).get('version', 'unknown')
                }
                logger.info(f"Appium server accessible at {url}: {server_info}")
                return True, server_info
            else:
                logger.warning(f"Appium server returned status {response.status_code}")
                return False, None
                
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Appium server at {url}")
            return False, None
        except requests.exceptions.Timeout:
            logger.error(f"Timeout connecting to Appium server at {url}")
            return False, None
        except Exception as e:
            logger.error(f"Error checking Appium server: {e}")
            return False, None
    
    def get_device_list(self, platform: str) -> List[Dict[str, str]]:
        """
        Get list of available devices/emulators.
        
        Args:
            platform: 'Android' or 'iOS'
            
        Returns:
            List of available devices
        """
        devices = []
        
        if platform == 'Android':
            # This would normally use ADB to list devices
            # For now, return common emulator names
            devices = [
                {'name': 'emulator-5554', 'type': 'emulator'},
                {'name': 'emulator-5556', 'type': 'emulator'}
            ]
        elif platform == 'iOS':
            # This would normally use xcrun simctl to list simulators
            # For now, return common simulator names
            devices = [
                {'name': 'iPhone 14', 'type': 'simulator'},
                {'name': 'iPhone 15', 'type': 'simulator'},
                {'name': 'iPad Pro', 'type': 'simulator'}
            ]
            
        return devices
    
    def parse_capabilities_string(self, caps_string: str) -> Dict[str, Any]:
        """
        Parse capabilities from a string format.
        
        Supports formats like:
        - "platformName=Android deviceName=emulator"
        - JSON string
        
        Args:
            caps_string: String containing capabilities
            
        Returns:
            Dictionary of capabilities
        """
        # Try JSON first
        try:
            return json.loads(caps_string)
        except json.JSONDecodeError:
            pass
            
        # Parse key=value format
        caps = {}
        parts = caps_string.split()
        
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                # Convert boolean strings
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                # Add appium: prefix if needed
                if not key.startswith('appium:') and key not in ['platformName', 'deviceName']:
                    key = f'appium:{key}'
                caps[key] = value
                
        return caps
    
    def merge_capabilities(self, *cap_dicts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple capability dictionaries.
        
        Later dictionaries override earlier ones.
        
        Args:
            *cap_dicts: Variable number of capability dictionaries
            
        Returns:
            Merged capability dictionary
        """
        merged = {}
        for caps in cap_dicts:
            if caps:
                merged.update(caps)
        return merged
    
    def get_default_capabilities(self, platform: str) -> Dict[str, Any]:
        """
        Get default capabilities for a platform.
        
        Args:
            platform: 'Android' or 'iOS'
            
        Returns:
            Dictionary of default capabilities
        """
        if platform == 'Android':
            return {
                'platformName': 'Android',
                'appium:automationName': 'UiAutomator2',
                'appium:deviceName': 'Android Emulator',
                'appium:newCommandTimeout': self.default_timeout,
                'appium:autoGrantPermissions': True
            }
        elif platform == 'iOS':
            return {
                'platformName': 'iOS',
                'appium:automationName': 'XCUITest',
                'appium:deviceName': 'iPhone Simulator',
                'appium:newCommandTimeout': self.default_timeout,
                'appium:autoAcceptAlerts': False
            }
        else:
            return {}