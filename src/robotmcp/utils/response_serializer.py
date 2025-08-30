"""MCP response serialization utilities for complex objects."""

import base64
import json
import logging
from typing import Any, Dict, List, Union
from xml.etree.ElementTree import Element as XMLElement

logger = logging.getLogger(__name__)


class MCPResponseSerializer:
    """
    Handle serialization of complex objects for MCP responses.
    
    This ensures that complex objects like XML Elements, custom objects, etc.
    can be safely included in MCP tool responses without causing serialization errors.
    """

    def __init__(self):
        """Initialize the serializer."""
        self.max_string_length = 1000  # Truncate very long strings
        self.max_list_items = 50  # Limit list serialization for performance
        self.max_dict_items = 50  # Limit dict serialization for performance

    def serialize_for_response(self, obj: Any) -> Any:
        """
        Convert an object to MCP-serializable format.
        
        Args:
            obj: Object to serialize
            
        Returns:
            MCP-serializable representation of the object
        """
        try:
            return self._serialize_object(obj)
        except Exception as e:
            logger.warning(f"Failed to serialize object {type(obj)}: {e}")
            # Fallback to string representation
            return {
                "_type": "serialization_fallback",
                "_original_type": type(obj).__name__,
                "_string_repr": str(obj)[:self.max_string_length],
                "_error": str(e)
            }

    def serialize_assigned_variables(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize assigned variables dictionary for MCP response.
        
        Args:
            variables: Dictionary of variable names to values
            
        Returns:
            Dictionary with serialized values safe for MCP response
        """
        if not variables:
            return {}
        
        serialized = {}
        for var_name, value in variables.items():
            try:
                serialized[var_name] = self.serialize_for_response(value)
            except Exception as e:
                logger.error(f"Failed to serialize variable '{var_name}': {e}")
                # Include error information but don't fail the whole response
                serialized[var_name] = {
                    "_type": "serialization_error",
                    "_original_type": type(value).__name__,
                    "_error": str(e),
                    "_fallback": str(value)[:100]
                }
        
        return serialized

    def _serialize_object(self, obj: Any, depth: int = 0) -> Any:
        """
        Internal recursive serialization method.
        
        Args:
            obj: Object to serialize
            depth: Current recursion depth (prevents infinite recursion)
            
        Returns:
            Serialized object
        """
        # Prevent infinite recursion
        if depth > 10:
            return {
                "_type": "max_depth_exceeded",
                "_string_repr": str(obj)[:100]
            }

        # Handle None
        if obj is None:
            return None

        # Handle basic types (already JSON serializable)
        if isinstance(obj, (str, int, float, bool)):
            if isinstance(obj, str) and len(obj) > self.max_string_length:
                return {
                    "_type": "truncated_string",
                    "_value": obj[:self.max_string_length],
                    "_original_length": len(obj),
                    "_truncated": True
                }
            return obj

        # Handle XML Elements
        if isinstance(obj, XMLElement):
            return self._serialize_xml_element(obj)

        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            return self._serialize_sequence(obj, depth)

        # Handle dictionaries
        if isinstance(obj, dict):
            return self._serialize_dict(obj, depth)

        # Handle sets
        if isinstance(obj, set):
            return {
                "_type": "set",
                "_items": self._serialize_sequence(list(obj), depth)
            }

        # Handle bytes
        if isinstance(obj, bytes):
            return self._serialize_bytes(obj)

        # Handle requests.Response objects specially - create a serialized representation
        # but preserve key information for method calls
        try:
            import requests
            if isinstance(obj, requests.models.Response):
                return self._serialize_requests_response(obj)
        except ImportError:
            pass  # requests not available, continue with normal processing

        # Handle other iterables (but not strings)
        if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            try:
                return {
                    "_type": "iterable",
                    "_original_type": type(obj).__name__,
                    "_items": self._serialize_sequence(list(obj), depth)
                }
            except Exception:
                pass  # Fall through to generic handling

        # Handle objects with useful attributes
        if hasattr(obj, '__dict__'):
            return self._serialize_object_with_dict(obj, depth)

        # Handle callable objects
        if callable(obj):
            return {
                "_type": "callable",
                "_name": getattr(obj, '__name__', str(obj)),
                "_module": getattr(obj, '__module__', None),
                "_string_repr": str(obj)
            }

        # Generic fallback for other objects
        return {
            "_type": "object",
            "_class": type(obj).__name__,
            "_module": getattr(type(obj), '__module__', None),
            "_string_repr": str(obj)[:self.max_string_length],
            "_has_dict": hasattr(obj, '__dict__'),
            "_dir": [attr for attr in dir(obj) if not attr.startswith('_')][:20]  # First 20 public attributes
        }

    def _serialize_xml_element(self, element: XMLElement) -> Dict[str, Any]:
        """Serialize XML Element to a descriptive dictionary."""
        try:
            # Basic element information
            result = {
                "_type": "xml_element",
                "_tag": element.tag,
                "_text": element.text,
                "_tail": element.tail,
                "_attrib": dict(element.attrib) if element.attrib else {},
            }

            # Add child information (but don't recurse deeply for performance)
            children = list(element)
            if children:
                result["_children"] = []
                for child in children[:10]:  # Limit to first 10 children
                    child_info = {
                        "tag": child.tag,
                        "text": child.text,
                        "attrib": dict(child.attrib) if child.attrib else {},
                        "child_count": len(list(child))
                    }
                    result["_children"].append(child_info)
                
                if len(children) > 10:
                    result["_children_truncated"] = True
                    result["_total_children"] = len(children)

            # Add XML string representation (truncated)
            try:
                import xml.etree.ElementTree as ET
                xml_str = ET.tostring(element, encoding='unicode', method='xml')
                if len(xml_str) > 500:
                    result["_xml_preview"] = xml_str[:500] + "..."
                    result["_xml_truncated"] = True
                else:
                    result["_xml_preview"] = xml_str
            except Exception as e:
                result["_xml_preview"] = f"<serialization_error: {e}>"

            return result

        except Exception as e:
            logger.warning(f"Failed to serialize XML element: {e}")
            return {
                "_type": "xml_element_error",
                "_error": str(e),
                "_string_repr": str(element)
            }

    def _serialize_sequence(self, seq: Union[List, tuple], depth: int) -> List[Any]:
        """Serialize a sequence (list or tuple)."""
        if len(seq) > self.max_list_items:
            # Serialize first N items and add truncation info
            serialized_items = [
                self._serialize_object(item, depth + 1) 
                for item in seq[:self.max_list_items]
            ]
            return {
                "_type": "truncated_list",
                "_items": serialized_items,
                "_original_length": len(seq),
                "_truncated": True,
                "_shown_items": self.max_list_items
            }
        else:
            return [self._serialize_object(item, depth + 1) for item in seq]

    def _serialize_dict(self, d: Dict[Any, Any], depth: int) -> Dict[str, Any]:
        """Serialize a dictionary."""
        if len(d) > self.max_dict_items:
            # Take first N items and add truncation info
            items = list(d.items())[:self.max_dict_items]
            serialized_dict = {
                str(k): self._serialize_object(v, depth + 1) 
                for k, v in items
            }
            return {
                "_type": "truncated_dict",
                "_items": serialized_dict,
                "_original_length": len(d),
                "_truncated": True,
                "_shown_items": self.max_dict_items
            }
        else:
            return {
                str(k): self._serialize_object(v, depth + 1) 
                for k, v in d.items()
            }

    def _serialize_bytes(self, data: bytes) -> Dict[str, Any]:
        """Serialize bytes data."""
        if len(data) <= 100:
            # Small bytes - include as base64
            return {
                "_type": "bytes",
                "_base64": base64.b64encode(data).decode('ascii'),
                "_length": len(data)
            }
        else:
            # Large bytes - just metadata
            return {
                "_type": "bytes",
                "_length": len(data),
                "_preview": base64.b64encode(data[:50]).decode('ascii') + "...",
                "_truncated": True
            }

    def _serialize_requests_response(self, response) -> Dict[str, Any]:
        """
        Serialize requests.Response object with key information preserved.
        
        This provides a serialized representation suitable for MCP responses
        while preserving important attributes for debugging and analysis.
        """
        try:
            # Extract key information from the Response object
            result = {
                "_type": "requests_response",
                "_original_type": "Response",
                "status_code": response.status_code,
                "reason": response.reason,
                "url": str(response.url),
                "headers": dict(response.headers),
                "encoding": response.encoding,
                "apparent_encoding": response.apparent_encoding,
            }
            
            # Try to get JSON content safely
            try:
                json_content = response.json()
                # Limit JSON size for performance
                json_str = json.dumps(json_content)
                if len(json_str) > 1000:
                    result["json_preview"] = json_str[:1000] + "..."
                    result["json_truncated"] = True
                else:
                    result["json"] = json_content
            except Exception as e:
                result["json_error"] = f"Could not parse JSON: {e}"
            
            # Add text content preview
            try:
                text_content = response.text
                if len(text_content) > 500:
                    result["text_preview"] = text_content[:500] + "..."
                    result["text_truncated"] = True
                    result["text_length"] = len(text_content)
                else:
                    result["text"] = text_content
            except Exception as e:
                result["text_error"] = f"Could not get text content: {e}"
                
            # Add content info
            if hasattr(response, 'content'):
                result["content_length"] = len(response.content) if response.content else 0
                
            # Add timing info if available
            if hasattr(response, 'elapsed'):
                result["elapsed_seconds"] = response.elapsed.total_seconds()
                
            # Add cookies info if available
            if hasattr(response, 'cookies') and response.cookies:
                result["cookies"] = dict(response.cookies)
                
            return result
            
        except Exception as e:
            logger.warning(f"Failed to serialize requests.Response: {e}")
            # Fallback to basic info
            return {
                "_type": "requests_response_error",
                "_original_type": "Response", 
                "_error": str(e),
                "_string_repr": str(response)[:100],
                "status_code": getattr(response, 'status_code', 'unknown'),
                "url": str(getattr(response, 'url', 'unknown'))
            }

    def _serialize_object_with_dict(self, obj: Any, depth: int) -> Dict[str, Any]:
        """Serialize an object that has a __dict__ attribute."""
        try:
            result = {
                "_type": "object_with_dict",
                "_class": type(obj).__name__,
                "_module": getattr(type(obj), '__module__', None),
            }

            # Add the object's __dict__ (limited)
            obj_dict = obj.__dict__
            if len(obj_dict) > self.max_dict_items:
                # Limit the number of attributes
                items = list(obj_dict.items())[:self.max_dict_items]
                result["_attributes"] = {
                    k: self._serialize_object(v, depth + 1)
                    for k, v in items
                }
                result["_attributes_truncated"] = True
                result["_total_attributes"] = len(obj_dict)
            else:
                result["_attributes"] = {
                    k: self._serialize_object(v, depth + 1)
                    for k, v in obj_dict.items()
                }

            # Add string representation
            result["_string_repr"] = str(obj)[:self.max_string_length]

            return result

        except Exception as e:
            logger.warning(f"Failed to serialize object with __dict__: {e}")
            return {
                "_type": "object_dict_error",
                "_class": type(obj).__name__,
                "_error": str(e),
                "_string_repr": str(obj)[:100]
            }

    def is_serializable_natively(self, obj: Any) -> bool:
        """
        Check if an object is natively JSON serializable.
        
        Args:
            obj: Object to check
            
        Returns:
            True if object can be serialized by json.dumps()
        """
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False

    def get_serialization_info(self, obj: Any) -> Dict[str, Any]:
        """
        Get information about how an object would be serialized.
        
        Args:
            obj: Object to analyze
            
        Returns:
            Dictionary with serialization analysis
        """
        return {
            "original_type": type(obj).__name__,
            "is_natively_serializable": self.is_serializable_natively(obj),
            "has_dict": hasattr(obj, '__dict__'),
            "is_callable": callable(obj),
            "is_iterable": hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)),
            "string_length": len(str(obj)),
            "special_handling": self._get_special_handling_type(obj)
        }

    def _get_special_handling_type(self, obj: Any) -> str:
        """Determine what special handling an object would receive."""
        if obj is None:
            return "none"
        elif isinstance(obj, (str, int, float, bool)):
            return "basic_type"
        elif isinstance(obj, XMLElement):
            return "xml_element"
        elif isinstance(obj, (list, tuple)):
            return "sequence"
        elif isinstance(obj, dict):
            return "dictionary"
        elif isinstance(obj, set):
            return "set"
        elif isinstance(obj, bytes):
            return "bytes"
        elif callable(obj):
            return "callable"
        elif hasattr(obj, '__dict__'):
            return "object_with_dict"
        else:
            return "generic_object"