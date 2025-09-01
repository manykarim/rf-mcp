"""Enhanced MCP response serialization utilities for complex objects with intelligent data reduction."""

import base64
import inspect
import logging
from typing import Any, Dict, List, Union
from xml.etree.ElementTree import Element as XMLElement

logger = logging.getLogger(__name__)


class MCPResponseSerializer:
    """
    Handle serialization of complex objects for MCP responses with intelligent data reduction.

    This ensures that complex objects like XML Elements, custom objects, etc.
    can be safely included in MCP tool responses without causing serialization errors
    or excessive response size.

    Features:
    - Type-based intelligent serialization (simple types serialized fully, complex types summarized)
    - Detail level control ('minimal', 'standard', 'full')
    - Special handling for common RF library objects
    - Configurable limits for strings, lists, and dictionaries
    """

    def __init__(self):
        """Initialize the serializer with configurable limits."""
        self.max_string_length = 1000  # Truncate very long strings
        self.max_list_items = 50  # Limit list serialization for performance
        self.max_dict_items = 50  # Limit dict serialization for performance
        self.max_depth = 10  # Maximum recursion depth

    def serialize_for_response(self, obj: Any, detail_level: str = "minimal") -> Any:
        """
        Convert an object to MCP-serializable format with detail level control.

        Args:
            obj: Object to serialize
            detail_level: Controls serialization depth ('minimal', 'standard', 'full')

        Returns:
            MCP-serializable representation of the object
        """
        try:
            return self._serialize_object(obj, depth=0, detail_level=detail_level)
        except Exception as e:
            logger.warning(f"Failed to serialize object {type(obj)}: {e}")
            # Fallback to string representation
            return self._create_serialization_fallback(obj, str(e))

    def serialize_assigned_variables(
        self, variables: Dict[str, Any], detail_level: str = "minimal"
    ) -> Dict[str, Any]:
        """
        Serialize assigned variables dictionary for MCP response.

        Args:
            variables: Dictionary of variable names to values
            detail_level: Controls serialization depth

        Returns:
            Dictionary with serialized values safe for MCP response
        """
        if not variables:
            return {}

        serialized = {}
        for var_name, value in variables.items():
            try:
                serialized[var_name] = self.serialize_for_response(
                    value, detail_level=detail_level
                )
            except Exception as e:
                logger.error(f"Failed to serialize variable '{var_name}': {e}")
                # Include error information but don't fail the whole response
                serialized[var_name] = self._create_serialization_error(value, str(e))

        return serialized

    def _serialize_object(
        self, obj: Any, depth: int = 0, detail_level: str = "minimal"
    ) -> Any:
        """
        Internal recursive serialization method with intelligent type-based reduction.

        Args:
            obj: Object to serialize
            depth: Current recursion depth (prevents infinite recursion)
            detail_level: Controls serialization depth ('minimal', 'standard', 'full')

        Returns:
            Serialized object with appropriate level of detail
        """
        # Prevent infinite recursion
        if depth > self.max_depth:
            return self._create_max_depth_exceeded(obj)

        # Handle None
        if obj is None:
            return None

        # Always fully serialize basic primitive types
        if isinstance(obj, (str, int, float, bool)):
            if isinstance(obj, str) and len(obj) > self.max_string_length:
                return self._create_truncated_string(obj)
            return obj

        # Check for special object types that need custom handling FIRST
        # to ensure proper type detection before general rules are applied

        # Handle XML Elements
        if isinstance(obj, XMLElement):
            return self._serialize_xml_element(obj, detail_level)

        # Try to detect Selenium WebElement
        if self._is_selenium_webelement(obj):
            return self._serialize_selenium_webelement(obj)

        # Try to detect Browser Library ElementHandle
        if self._is_browser_element_handle(obj):
            return self._serialize_browser_element_handle(obj)

        # Handle requests.Response objects specially
        try:
            import requests

            if (
                isinstance(obj, requests.models.Response)
                or type(obj).__name__ == "MockResponse"
            ):
                return self._serialize_requests_response(obj, detail_level)
        except ImportError:
            pass  # requests not available, continue with normal processing

        # For minimal detail level, summarize complex objects after first level
        if detail_level == "minimal" and depth > 0:
            # Always fully serialize small lists and dictionaries for usability
            if isinstance(obj, (list, tuple)) and len(obj) <= 5:
                pass  # Continue to list handling below
            elif isinstance(obj, dict) and len(obj) <= 5:
                pass  # Continue to dict handling below
            else:
                # For other complex objects, return type summary
                return self._create_type_summary(obj)

        # For standard detail level, summarize deeply nested complex objects
        if detail_level == "standard" and depth > 2:
            if not isinstance(obj, (list, tuple, dict)) or len(obj) > 10:
                return self._create_type_summary(obj)

        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            return self._serialize_sequence(obj, depth, detail_level)

        # Handle dictionaries
        if isinstance(obj, dict):
            return self._serialize_dict(obj, depth, detail_level)

        # Handle sets
        if isinstance(obj, set):
            return {
                "_type": "set",
                "_items": self._serialize_sequence(list(obj), depth, detail_level),
            }

        # Handle bytes
        if isinstance(obj, bytes):
            return self._serialize_bytes(obj)

        # Handle other iterables (but not strings)
        if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
            try:
                return {
                    "_type": "iterable",
                    "_original_type": type(obj).__name__,
                    "_items": self._serialize_sequence(
                        list(obj)[:20], depth, detail_level
                    ),
                }
            except Exception:
                pass  # Fall through to generic handling

        # For full detail level or shallow depth, provide more information
        if detail_level == "full" or depth <= 1:
            # Handle objects with useful attributes
            if hasattr(obj, "__dict__"):
                return self._serialize_object_with_dict(obj, depth, detail_level)

        # For minimal detail level or deep nesting, just return a type summary
        return self._create_type_summary(obj)

    def _create_type_summary(self, obj: Any) -> Dict[str, Any]:
        """Create a concise type summary for complex objects."""
        try:
            # Get string representation safely
            str_repr = str(obj)
            if len(str_repr) > 100:
                str_repr = str_repr[:100] + "..."
        except Exception:
            str_repr = "<Error getting string representation>"

        return {
            "_type": "object_summary",
            "_class": type(obj).__name__,
            "_module": getattr(type(obj), "__module__", None),
            "_string_preview": str_repr,
        }

    def _create_truncated_string(self, s: str) -> Dict[str, Any]:
        """Create a truncated string representation."""
        return {
            "_type": "truncated_string",
            "_value": s[: self.max_string_length],
            "_original_length": len(s),
            "_truncated": True,
        }

    def _create_serialization_fallback(self, obj: Any, error: str) -> Dict[str, Any]:
        """Create a fallback for failed serialization."""
        return {
            "_type": "serialization_fallback",
            "_original_type": type(obj).__name__,
            "_string_repr": str(obj)[: self.max_string_length],
            "_error": error,
        }

    def _create_serialization_error(self, obj: Any, error: str) -> Dict[str, Any]:
        """Create an error entry for failed variable serialization."""
        return {
            "_type": "serialization_error",
            "_original_type": type(obj).__name__,
            "_error": error,
            "_fallback": str(obj)[:100],
        }

    def _create_max_depth_exceeded(self, obj: Any) -> Dict[str, Any]:
        """Create an entry for objects that exceed maximum recursion depth."""
        return {
            "_type": "max_depth_exceeded",
            "_original_type": type(obj).__name__,
            "_string_repr": str(obj)[:100],
        }

    def _is_selenium_webelement(self, obj: Any) -> bool:
        """Check if object is a Selenium WebElement."""
        # Check class name and common attributes/methods
        class_name = type(obj).__name__
        if class_name in ["WebElement", "RemoteWebElement", "MockWebElement"]:
            return hasattr(obj, "click") and hasattr(obj, "find_element")
        return False

    def _serialize_selenium_webelement(self, obj: Any) -> Dict[str, Any]:
        """Create a summarized representation of Selenium WebElement."""
        element_id = "unknown"
        tag_name = "unknown"

        # Safely try to get element details
        try:
            element_id = obj.id
        except Exception:
            pass

        try:
            tag_name = obj.tag_name
        except Exception:
            pass

        return {
            "_type": "selenium_webelement",
            "_class": type(obj).__name__,
            "_id": element_id,
            "_tag": tag_name,
        }

    def _is_browser_element_handle(self, obj: Any) -> bool:
        """Check if object is a Browser Library ElementHandle."""
        class_name = type(obj).__name__
        module_name = getattr(type(obj), "__module__", "")

        # Check for Browser Library's ElementHandle
        return (
            class_name == "ElementHandle"
            or "ElementHandle" in class_name
            or "playwright" in module_name
        )

    def _serialize_browser_element_handle(self, obj: Any) -> Dict[str, Any]:
        """Create a summarized representation of Browser Library ElementHandle."""
        return {
            "_type": "browser_element_handle",
            "_class": type(obj).__name__,
            "_module": getattr(type(obj), "__module__", None),
        }

    def _serialize_sequence(
        self, seq: Union[List, tuple], depth: int, detail_level: str
    ) -> Any:
        """
        Serialize a sequence (list or tuple) with detail level control.

        For minimal detail level, only show summary for large lists.
        """
        # For minimal detail level and large lists, return a summary
        if detail_level == "minimal" and len(seq) > 10 and depth > 0:
            return {
                "_type": "list_summary",
                "_length": len(seq),
                "_sample": [
                    self._serialize_object(item, depth + 1, detail_level)
                    for item in seq[:3]
                ],
                "_class": type(seq).__name__,
            }

        # For standard detail and very large lists, return a summary
        if detail_level == "standard" and len(seq) > 20 and depth > 1:
            return {
                "_type": "list_summary",
                "_length": len(seq),
                "_sample": [
                    self._serialize_object(item, depth + 1, detail_level)
                    for item in seq[:5]
                ],
                "_class": type(seq).__name__,
            }

        # Handle truncation for large lists
        if len(seq) > self.max_list_items:
            # Serialize first N items and add truncation info
            serialized_items = [
                self._serialize_object(item, depth + 1, detail_level)
                for item in seq[: self.max_list_items]
            ]
            return {
                "_type": "truncated_list",
                "_items": serialized_items,
                "_original_length": len(seq),
                "_truncated": True,
                "_shown_items": self.max_list_items,
            }
        else:
            return [
                self._serialize_object(item, depth + 1, detail_level) for item in seq
            ]

    def _serialize_dict(self, d: Dict[Any, Any], depth: int, detail_level: str) -> Any:
        """
        Serialize a dictionary with detail level control.

        For minimal detail level, only show summary for large dicts.
        """
        # For minimal detail level and large dicts, return a summary
        if detail_level == "minimal" and len(d) > 10 and depth > 0:
            # Get a sample of keys
            sample_keys = list(d.keys())[:3]
            sample_dict = {k: d[k] for k in sample_keys}

            return {
                "_type": "dict_summary",
                "_length": len(d),
                "_sample": {
                    k: self._serialize_object(v, depth + 1, detail_level)
                    for k, v in sample_dict.items()
                },
                "_keys": list(d.keys())[:10] if len(d) > 10 else list(d.keys()),
            }

        # For standard detail and very large dicts, return a summary
        if detail_level == "standard" and len(d) > 20 and depth > 1:
            # Get a sample of keys
            sample_keys = list(d.keys())[:5]
            sample_dict = {k: d[k] for k in sample_keys}

            return {
                "_type": "dict_summary",
                "_length": len(d),
                "_sample": {
                    k: self._serialize_object(v, depth + 1, detail_level)
                    for k, v in sample_dict.items()
                },
                "_keys": list(d.keys())[:15] if len(d) > 15 else list(d.keys()),
            }

        # Handle truncation for large dictionaries
        if len(d) > self.max_dict_items:
            # Take first N items and add truncation info
            items = list(d.items())[: self.max_dict_items]
            serialized_dict = {}

            for k, v in items:
                # Convert non-string keys to strings
                key = str(k) if not isinstance(k, str) else k
                serialized_dict[key] = self._serialize_object(
                    v, depth + 1, detail_level
                )

            return {
                "_type": "truncated_dict",
                "_items": serialized_dict,
                "_original_length": len(d),
                "_truncated": True,
                "_shown_items": self.max_dict_items,
                "_keys": [str(k) for k in d.keys()][:20],  # Show first 20 keys
            }
        else:
            result = {}
            for k, v in d.items():
                # Convert non-string keys to strings
                key = str(k) if not isinstance(k, str) else k
                result[key] = self._serialize_object(v, depth + 1, detail_level)
            return result

    def _serialize_bytes(self, b: bytes) -> Dict[str, Any]:
        """Serialize bytes to a descriptive dictionary."""
        # For large byte arrays, just return a summary
        if len(b) > 100:
            return {
                "_type": "bytes",
                "_length": len(b),
                "_preview": base64.b64encode(b[:50]).decode("ascii"),
                "_truncated": True,
            }

        # For small byte arrays, include full content
        return {
            "_type": "bytes",
            "_length": len(b),
            "_base64": base64.b64encode(b).decode("ascii"),
        }

    def _serialize_requests_response(
        self, response, detail_level: str
    ) -> Dict[str, Any]:
        """
        Serialize a requests.Response object with appropriate detail level.

        For minimal detail level, only include basic info like status code and URL.
        For standard detail level, include headers but not content.
        For full detail level, include all details.
        """
        # Basic info for all detail levels
        result = {
            "_type": "requests_response",
            "status_code": response.status_code,
            "url": response.url,
            "reason": response.reason,
            "elapsed": str(response.elapsed),
        }

        # Add headers for standard and full detail levels
        if detail_level in ["standard", "full"]:
            result["headers"] = dict(response.headers)

        # Add content for full detail level only
        if detail_level == "full":
            try:
                if "application/json" in response.headers.get("Content-Type", ""):
                    result["json"] = response.json()
                else:
                    # Try to decode text
                    try:
                        content = response.text
                        if len(content) > 1000:
                            result["text"] = content[:1000] + "..."
                            result["text_truncated"] = True
                            result["content_length"] = len(content)
                        else:
                            result["text"] = content
                    except Exception:
                        # Fallback to content length only
                        result["content_length"] = len(response.content)
            except Exception as e:
                result["content_error"] = str(e)

        return result

    def _serialize_xml_element(
        self, element: XMLElement, detail_level: str
    ) -> Dict[str, Any]:
        """Serialize XML Element to a descriptive dictionary with detail level control."""
        try:
            # Basic element information for all detail levels
            result = {
                "_type": "xml_element",
                "_tag": element.tag,
            }

            # For minimal detail level, just return basic info
            if detail_level == "minimal":
                # Add attribute count and child count for context
                result["_attrib_count"] = len(element.attrib) if element.attrib else 0
                result["_child_count"] = len(list(element))
                return result

            # For standard and full detail levels, include more information
            result.update(
                {
                    "_text": element.text,
                    "_tail": element.tail,
                    "_attrib": dict(element.attrib) if element.attrib else {},
                }
            )

            # Add child information for standard and full detail levels
            children = list(element)
            if children:
                if detail_level == "standard":
                    # For standard, just summary info about children
                    result["child_tags"] = [child.tag for child in children[:10]]
                    if len(children) > 10:
                        result["_children_truncated"] = True
                else:  # full detail level
                    # For full, include detailed child info
                    result["_children"] = []
                    for child in children[:10]:  # Limit to first 10 children
                        child_info = {
                            "tag": child.tag,
                            "text": child.text,
                            "attrib": dict(child.attrib) if child.attrib else {},
                            "child_count": len(list(child)),
                        }
                        result["_children"].append(child_info)

                    if len(children) > 10:
                        result["_children_truncated"] = True
                        result["_total_children"] = len(children)

            # Add XML string representation for full detail level only
            if detail_level == "full":
                try:
                    import xml.etree.ElementTree as ET

                    xml_str = ET.tostring(element, encoding="unicode", method="xml")
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
                "_string_repr": str(element),
            }

    def _serialize_object_with_dict(
        self, obj: Any, depth: int, detail_level: str
    ) -> Dict[str, Any]:
        """Serialize an object with a __dict__ attribute."""
        try:
            # For minimal detail, just return type info
            if detail_level == "minimal":
                return self._create_type_summary(obj)

            # For standard detail or shallow depth, include basic dict info
            obj_dict = obj.__dict__

            # Create basic result with object info
            result = {
                "_type": "object",
                "_class": type(obj).__name__,
                "_module": getattr(type(obj), "__module__", None),
            }

            # For standard detail level, include attributes but with possible summarization
            if detail_level == "standard":
                # Limit attributes for standard detail level
                if len(obj_dict) > 10:
                    # Take a subset of attributes
                    result["_attributes"] = {
                        k: self._serialize_object(v, depth + 1, detail_level)
                        for k, v in list(obj_dict.items())[:10]
                    }
                    result["_attribute_count"] = len(obj_dict)
                    result["_attributes_truncated"] = True
                else:
                    # Include all attributes for smaller objects
                    result["_attributes"] = {
                        k: self._serialize_object(v, depth + 1, detail_level)
                        for k, v in obj_dict.items()
                    }

            # For full detail level, include all attributes and methods
            elif detail_level == "full":
                # Include full attribute information
                result["_attributes"] = {
                    k: self._serialize_object(v, depth + 1, detail_level)
                    for k, v in obj_dict.items()
                }

                # Include method information for full detail
                result["_methods"] = [
                    {
                        "name": name,
                        "signature": str(inspect.signature(method))
                        if callable(method)
                        else None,
                    }
                    for name, method in inspect.getmembers(
                        obj, predicate=inspect.ismethod
                    )
                    if not name.startswith("_")
                ][:20]  # Limit to 20 methods

            return result

        except Exception as e:
            logger.warning(f"Failed to serialize object with dict: {e}")
            return self._create_type_summary(obj)
