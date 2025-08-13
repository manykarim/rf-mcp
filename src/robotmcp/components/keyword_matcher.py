"""Keyword Matcher component for semantic matching of Robot Framework keywords."""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import asyncio
from difflib import SequenceMatcher

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from scipy.spatial.distance import cosine
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    from robot.libraries import STDLIBS
except ImportError:
    STDLIBS = {}

try:
    from robot.api import get_model
except ImportError:
    get_model = None

try:
    from robot.libdoc import LibraryDocumentation
    LIBDOC_AVAILABLE = True
except ImportError:
    LIBDOC_AVAILABLE = False
    LibraryDocumentation = None

logger = logging.getLogger(__name__)

@dataclass
class KeywordMatch:
    """Represents a matched keyword with metadata."""
    keyword_name: str
    library: str
    confidence: float
    arguments: List[str]
    argument_types: List[str]
    documentation: str
    usage_example: Optional[str] = None

@dataclass
class KeywordInfo:
    """Information about a Robot Framework keyword."""
    name: str
    library: str
    arguments: List[str]
    argument_types: List[str]
    documentation: str
    tags: List[str]
    source: Optional[str] = None
    lineno: Optional[int] = None
    deprecated: bool = False
    private: bool = False

class KeywordMatcher:
    """Matches natural language actions to Robot Framework keywords using semantic similarity."""
    
    def __init__(self):
        self.keyword_registry: Dict[str, List[KeywordInfo]] = {}
        self.embeddings_model = None
        self.keyword_embeddings: Dict[str, np.ndarray] = {}
        self._initialized = False
        
        # Initialize embeddings model if available
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded sentence transformer model for semantic matching")
            except Exception as e:
                logger.warning(f"Could not load embeddings model: {e}")
                self.embeddings_model = None
        
        # Common keyword patterns for different actions
        self.action_keyword_mapping = {
            'setup_browser': {
                'patterns': ['create browser', 'new browser', 'start browser'],
                'libraries': ['Browser'],
                'keywords': ['New Browser', 'New Context', 'New Page']
            },
            'navigate': {
                'patterns': ['go to', 'navigate', 'visit', 'new page'],
                'libraries': ['Browser', 'SeleniumLibrary'],
                'keywords': ['New Page', 'Navigate To', 'Go To']
            },
            'open_browser': {
                'patterns': ['open browser', 'start selenium'],
                'libraries': ['SeleniumLibrary'],
                'keywords': ['Open Browser']
            },
            'click': {
                'patterns': ['click', 'press', 'select', 'tap'],
                'libraries': ['Browser', 'SeleniumLibrary', 'AppiumLibrary'],
                'keywords': ['Click', 'Click Element', 'Click Button', 'Click Link', 'Tap']
            },
            'input': {
                'patterns': ['type', 'enter', 'input', 'fill', 'set'],
                'libraries': ['Browser', 'SeleniumLibrary', 'AppiumLibrary'],
                'keywords': ['Fill', 'Fill Text', 'Type Text', 'Input Text', 'Set Text']
            },
            'verify': {
                'patterns': ['verify', 'check', 'assert', 'should', 'expect', 'get text'],
                'libraries': ['Browser', 'SeleniumLibrary', 'BuiltIn'],
                'keywords': ['Get Text', 'Wait For Elements State', 'Page Should Contain', 'Element Should Be Visible', 'Should Be Equal']
            },
            'wait': {
                'patterns': ['wait', 'pause', 'sleep', 'delay'],
                'libraries': ['Browser', 'SeleniumLibrary', 'BuiltIn'],
                'keywords': ['Wait For Elements State', 'Wait For Condition', 'Wait Until Element Is Visible', 'Sleep']
            },
            'search': {
                'patterns': ['search', 'find', 'look for', 'locate', 'get element'],
                'libraries': ['Browser', 'SeleniumLibrary'],
                'keywords': ['Get Element', 'Get Elements', 'Find Element', 'Locate Element']
            },
            'property': {
                'patterns': ['get property', 'property', 'attribute'],
                'libraries': ['Browser'],
                'keywords': ['Get Property', 'Get Attribute', 'Get Element Attribute']
            },
            'cleanup': {
                'patterns': ['close', 'cleanup', 'teardown', 'quit'],
                'libraries': ['Browser', 'SeleniumLibrary'],
                'keywords': ['Close Browser', 'Close All Browsers', 'Quit']
            }
        }

    async def _ensure_initialized(self) -> None:
        """Ensure the keyword registry is initialized."""
        if not self._initialized:
            await self._initialize_keyword_registry()
            self._initialized = True

    async def _initialize_keyword_registry(self) -> None:
        """Initialize the keyword registry with standard libraries."""
        try:
            # Load library list from centralized registry
            from robotmcp.config.library_registry import get_library_names_for_loading
            all_libraries = get_library_names_for_loading()
            
            for lib_name in all_libraries:
                try:
                    await self._load_library_keywords(lib_name)
                except Exception as e:
                    logger.debug(f"Could not load {lib_name}: {e}")
            
            logger.info(f"Loaded {len(self.keyword_registry)} libraries into keyword registry")
            
        except Exception as e:
            logger.error(f"Error initializing keyword registry: {e}")

    async def _load_library_keywords(self, library_name: str) -> None:
        """Load keywords from a specific library using robot.libdoc."""
        try:
            if not LIBDOC_AVAILABLE:
                logger.warning("robot.libdoc not available, falling back to manual loading")
                await self._load_library_keywords_fallback(library_name)
                return
            
            # Use LibraryDocumentation for comprehensive keyword extraction
            try:
                lib_doc = LibraryDocumentation(library_name)
            except Exception as e:
                logger.debug(f"LibraryDocumentation failed for {library_name}, trying fallback: {e}")
                await self._load_library_keywords_fallback(library_name)
                return
            
            keywords = []
            
            # Extract keywords with full metadata
            for kw in lib_doc.keywords:
                try:
                    # Parse arguments with proper types
                    arguments = []
                    argument_types = []
                    
                    if hasattr(kw, 'args') and kw.args:
                        for arg in kw.args:
                            if isinstance(arg, str):
                                # Simple string argument
                                arguments.append(arg)
                                argument_types.append('str')
                            elif hasattr(arg, 'name'):
                                # Argument object with metadata
                                arguments.append(arg.name)
                                arg_type = getattr(arg, 'type', 'str') or 'str'
                                argument_types.append(str(arg_type))
                            else:
                                arguments.append(str(arg))
                                argument_types.append('str')
                    
                    # Extract tags
                    tags = []
                    if hasattr(kw, 'tags') and kw.tags:
                        tags = list(kw.tags)
                    else:
                        # Fallback to documentation-based tag extraction
                        tags = self._extract_tags_from_doc(kw.doc or "")
                    
                    # Check for deprecated/private status
                    deprecated = 'robot:deprecated' in tags or 'deprecated' in (kw.doc or "").lower()
                    private = 'robot:private' in tags or kw.name.startswith('_')
                    
                    keyword_info = KeywordInfo(
                        name=kw.name,
                        library=library_name,
                        arguments=arguments,
                        argument_types=argument_types,
                        documentation=kw.doc or "",
                        tags=tags,
                        source=getattr(kw, 'source', None),
                        lineno=getattr(kw, 'lineno', None),
                        deprecated=deprecated,
                        private=private
                    )
                    
                    # Skip private keywords unless explicitly requested
                    if not private:
                        keywords.append(keyword_info)
                        
                except Exception as e:
                    logger.debug(f"Could not process keyword {kw.name}: {e}")
            
            self.keyword_registry[library_name] = keywords
            logger.debug(f"Loaded {len(keywords)} keywords from {library_name} using LibraryDocumentation")
            
            # Generate embeddings for keywords if model is available
            if self.embeddings_model and keywords:
                await self._generate_keyword_embeddings(library_name, keywords)
                
        except Exception as e:
            logger.warning(f"Could not load library {library_name}: {e}")
            # Try fallback method
            await self._load_library_keywords_fallback(library_name)

    async def _generate_keyword_embeddings(self, library_name: str, keywords: List[KeywordInfo]) -> None:
        """Generate embeddings for keywords for semantic matching."""
        try:
            for keyword in keywords:
                # Create text for embedding: keyword name + documentation
                embedding_text = f"{keyword.name} {keyword.documentation}"
                embedding = self.embeddings_model.encode(embedding_text)
                self.keyword_embeddings[f"{library_name}.{keyword.name}"] = embedding
                
        except Exception as e:
            logger.warning(f"Could not generate embeddings for {library_name}: {e}")

    async def discover_keywords(
        self,
        action_description: str,
        context: str = "web",
        current_state: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Discover matching Robot Framework keywords for an action description.
        
        Args:
            action_description: Natural language description of the action
            context: Application context (web, mobile, api, database)
            current_state: Current application state
            
        Returns:
            Dictionary containing ranked keyword matches
        """
        try:
            # Ensure initialization is complete
            await self._ensure_initialized()
            
            if current_state is None:
                current_state = {}
            
            # Normalize action description
            normalized_action = self._normalize_action(action_description)
            
            # Extract action type from description
            action_type = self._classify_action(normalized_action)
            
            # Get keyword matches using multiple strategies
            matches = []
            
            # Strategy 1: Pattern-based matching
            pattern_matches = await self._pattern_based_matching(normalized_action, action_type, context)
            matches.extend(pattern_matches)
            
            # Strategy 2: Semantic similarity matching (if embeddings available)
            if self.embeddings_model:
                semantic_matches = await self._semantic_matching(normalized_action, context)
                matches.extend(semantic_matches)
            
            # Strategy 3: Context-aware matching
            context_matches = await self._context_aware_matching(
                normalized_action, context, current_state
            )
            matches.extend(context_matches)
            
            # Remove duplicates and rank by confidence
            unique_matches = self._deduplicate_matches(matches)
            ranked_matches = self._rank_matches(unique_matches, normalized_action, context)
            
            # Limit to top 10 matches
            top_matches = ranked_matches[:10]
            
            return {
                "success": True,
                "action_description": action_description,
                "action_type": action_type,
                "matches": [
                    {
                        "keyword_name": match.keyword_name,
                        "library": match.library,
                        "confidence": match.confidence,
                        "arguments": match.arguments,
                        "argument_types": match.argument_types,
                        "documentation": match.documentation[:200] + "..." if len(match.documentation) > 200 else match.documentation,
                        "usage_example": match.usage_example
                    } for match in top_matches
                ],
                "total_matches": len(unique_matches),
                "recommendations": self._generate_usage_recommendations(top_matches, normalized_action)
            }
            
        except Exception as e:
            logger.error(f"Error discovering keywords: {e}")
            return {
                "success": False,
                "error": str(e),
                "matches": []
            }

    def _normalize_action(self, action: str) -> str:
        """Normalize action description for matching."""
        # Convert to lowercase and remove extra whitespace
        normalized = re.sub(r'\s+', ' ', action.lower().strip())
        
        # Remove quotes and common filler words
        normalized = re.sub(r'["\']', '', normalized)
        normalized = re.sub(r'\b(the|a|an|on|in|at|to|for|with|by)\b', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized

    def _classify_action(self, action: str) -> str:
        """Classify the type of action based on description."""
        action_lower = action.lower()
        
        # Check against known action patterns
        for action_type, config in self.action_keyword_mapping.items():
            for pattern in config['patterns']:
                if pattern in action_lower:
                    return action_type
        
        # Default classification based on common verbs
        if any(word in action_lower for word in ['open', 'go', 'navigate', 'visit']):
            return 'navigate'
        elif any(word in action_lower for word in ['click', 'press', 'select', 'tap']):
            return 'click'
        elif any(word in action_lower for word in ['type', 'enter', 'input', 'fill']):
            return 'input'
        elif any(word in action_lower for word in ['verify', 'check', 'assert', 'should']):
            return 'verify'
        elif any(word in action_lower for word in ['wait', 'pause', 'sleep']):
            return 'wait'
        else:
            return 'unknown'

    async def _pattern_based_matching(
        self,
        action: str,
        action_type: str,
        context: str
    ) -> List[KeywordMatch]:
        """Match keywords using predefined patterns."""
        matches = []
        
        if action_type in self.action_keyword_mapping:
            config = self.action_keyword_mapping[action_type]
            
            # Look for keywords in relevant libraries
            for library_name in config['libraries']:
                if library_name in self.keyword_registry:
                    for keyword_info in self.keyword_registry[library_name]:
                        # Check if keyword name matches expected patterns
                        for expected_keyword in config['keywords']:
                            similarity = self._calculate_string_similarity(
                                keyword_info.name, expected_keyword
                            )
                            
                            if similarity > 0.6:  # Threshold for pattern matching
                                matches.append(KeywordMatch(
                                    keyword_name=keyword_info.name,
                                    library=keyword_info.library,
                                    confidence=similarity * 0.8,  # Pattern matching gets lower confidence
                                    arguments=keyword_info.arguments,
                                    argument_types=keyword_info.argument_types,
                                    documentation=keyword_info.documentation,
                                    usage_example=self._generate_usage_example(keyword_info, action)
                                ))
        
        return matches

    async def _semantic_matching(self, action: str, context: str) -> List[KeywordMatch]:
        """Match keywords using semantic similarity."""
        matches = []
        
        if not self.embeddings_model or not self.keyword_embeddings:
            return matches
        
        try:
            # Generate embedding for the action
            action_embedding = self.embeddings_model.encode(action)
            
            # Calculate similarity with all keyword embeddings
            for keyword_key, keyword_embedding in self.keyword_embeddings.items():
                similarity = 1 - cosine(action_embedding, keyword_embedding)
                
                if similarity > 0.3:  # Minimum similarity threshold
                    library_name, keyword_name = keyword_key.split('.', 1)
                    
                    # Find keyword info
                    keyword_info = None
                    if library_name in self.keyword_registry:
                        for kw in self.keyword_registry[library_name]:
                            if kw.name == keyword_name:
                                keyword_info = kw
                                break
                    
                    if keyword_info:
                        matches.append(KeywordMatch(
                            keyword_name=keyword_info.name,
                            library=keyword_info.library,
                            confidence=similarity,
                            arguments=keyword_info.arguments,
                            argument_types=keyword_info.argument_types,
                            documentation=keyword_info.documentation,
                            usage_example=self._generate_usage_example(keyword_info, action)
                        ))
        
        except Exception as e:
            logger.warning(f"Error in semantic matching: {e}")
        
        return matches

    async def _context_aware_matching(
        self,
        action: str,
        context: str,
        current_state: Dict[str, Any]
    ) -> List[KeywordMatch]:
        """Match keywords based on current context and state using tags."""
        matches = []
        
        # Priority libraries based on context - use centralized registry for consistency
        from robotmcp.config.library_registry import get_libraries_by_category, LibraryCategory
        context_libraries = {
            'web': list(get_libraries_by_category(LibraryCategory.WEB).keys()),
            'mobile': list(get_libraries_by_category(LibraryCategory.MOBILE).keys()), 
            'api': list(get_libraries_by_category(LibraryCategory.API).keys()),
            'database': list(get_libraries_by_category(LibraryCategory.DATABASE).keys())
        }
        
        # Context-to-tag mapping for better filtering
        context_tags = {
            'web': ['web', 'browser', 'selenium', 'html'],
            'mobile': ['mobile', 'app', 'appium', 'touch'],
            'api': ['api', 'http', 'request', 'rest'],
            'database': ['database', 'sql', 'db', 'query']
        }
        
        priority_libraries = context_libraries.get(context, ['BuiltIn'])
        relevant_tags = context_tags.get(context, [])
        
        # Look for keywords in all libraries, with priority weighting
        for library_name in self.keyword_registry:
            library_priority = 1.0 if library_name in priority_libraries else 0.7
            
            for keyword_info in self.keyword_registry[library_name]:
                # Skip deprecated keywords unless specifically requested
                if keyword_info.deprecated:
                    continue
                    
                # Calculate relevance based on documentation, context, and tags
                relevance = self._calculate_context_relevance(
                    keyword_info, action, context, current_state
                )
                
                # Boost relevance for keywords with matching tags
                tag_boost = 0.0
                if keyword_info.tags:
                    matching_tags = set(keyword_info.tags).intersection(set(relevant_tags))
                    if matching_tags:
                        tag_boost = min(len(matching_tags) * 0.15, 0.3)
                
                # Apply library priority and tag boost
                final_relevance = (relevance + tag_boost) * library_priority
                
                if final_relevance > 0.3:  # Lower threshold due to better matching
                    matches.append(KeywordMatch(
                        keyword_name=keyword_info.name,
                        library=keyword_info.library,
                        confidence=final_relevance,
                        arguments=keyword_info.arguments,
                        argument_types=keyword_info.argument_types,
                        documentation=keyword_info.documentation,
                        usage_example=self._generate_usage_example(keyword_info, action)
                    ))
        
        return matches

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings."""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def _calculate_context_relevance(
        self,
        keyword_info: KeywordInfo,
        action: str,
        context: str,
        current_state: Dict[str, Any]
    ) -> float:
        """Calculate how relevant a keyword is for the current context."""
        relevance = 0.0
        
        # Base similarity with action
        name_similarity = self._calculate_string_similarity(keyword_info.name, action)
        doc_similarity = self._calculate_string_similarity(keyword_info.documentation, action)
        relevance += max(name_similarity, doc_similarity * 0.5)
        
        # Context bonus
        if context == 'web' and any(term in keyword_info.documentation.lower() 
                                   for term in ['browser', 'element', 'page', 'web']):
            relevance += 0.2
        elif context == 'api' and any(term in keyword_info.documentation.lower()
                                     for term in ['request', 'response', 'http', 'api']):
            relevance += 0.2
        elif context == 'mobile' and any(term in keyword_info.documentation.lower()
                                        for term in ['mobile', 'app', 'touch', 'device']):
            relevance += 0.2
        
        # State-based relevance
        if current_state.get('dom') and 'element' in keyword_info.name.lower():
            relevance += 0.1
        
        return min(relevance, 1.0)

    def _deduplicate_matches(self, matches: List[KeywordMatch]) -> List[KeywordMatch]:
        """Remove duplicate matches, keeping the highest confidence."""
        unique_matches = {}
        
        for match in matches:
            key = f"{match.library}.{match.keyword_name}"
            if key not in unique_matches or match.confidence > unique_matches[key].confidence:
                unique_matches[key] = match
        
        return list(unique_matches.values())

    def _rank_matches(
        self,
        matches: List[KeywordMatch],
        action: str,
        context: str
    ) -> List[KeywordMatch]:
        """Rank matches by confidence and relevance."""
        return sorted(matches, key=lambda x: x.confidence, reverse=True)

    async def _load_library_keywords_fallback(self, library_name: str) -> None:
        """Fallback method for loading keywords without LibraryDocumentation."""
        try:
            # Import the library to get its keywords
            if library_name in STDLIBS:
                # Handle standard libraries
                lib_module = STDLIBS[library_name]
            else:
                # Try to import external library
                import importlib
                lib_module = importlib.import_module(library_name)
            
            keywords = []
            
            # Extract keywords from library
            if hasattr(lib_module, 'get_keyword_names'):
                keyword_names = lib_module.get_keyword_names()
                for kw_name in keyword_names:
                    try:
                        # Get keyword documentation and arguments
                        doc = ""
                        args = []
                        arg_types = []
                        tags = []
                        
                        if hasattr(lib_module, 'get_keyword_documentation'):
                            doc = lib_module.get_keyword_documentation(kw_name) or ""
                        
                        if hasattr(lib_module, 'get_keyword_arguments'):
                            args = lib_module.get_keyword_arguments(kw_name) or []
                            arg_types = ['str'] * len(args)  # Default to string
                        
                        if hasattr(lib_module, 'get_keyword_tags'):
                            tags = lib_module.get_keyword_tags(kw_name) or []
                        else:
                            tags = self._extract_tags_from_doc(doc)
                        
                        keyword_info = KeywordInfo(
                            name=kw_name,
                            library=library_name,
                            arguments=args,
                            argument_types=arg_types,
                            documentation=doc,
                            tags=tags
                        )
                        keywords.append(keyword_info)
                        
                    except Exception as e:
                        logger.debug(f"Could not process keyword {kw_name}: {e}")
            
            self.keyword_registry[library_name] = keywords
            logger.debug(f"Loaded {len(keywords)} keywords from {library_name} using fallback method")
            
            # Generate embeddings for keywords if model is available
            if self.embeddings_model and keywords:
                await self._generate_keyword_embeddings(library_name, keywords)
                
        except Exception as e:
            logger.warning(f"Fallback loading failed for library {library_name}: {e}")
    
    def _generate_usage_example(self, keyword_info: KeywordInfo, action: str) -> str:
        """Generate a usage example for a keyword."""
        if not keyword_info.arguments:
            return f"{keyword_info.name}"
        
        # Generate placeholder arguments based on action
        example_args = []
        for i, arg in enumerate(keyword_info.arguments):
            if 'locator' in arg.lower() or 'element' in arg.lower():
                example_args.append("id=my-element")
            elif 'text' in arg.lower() or 'value' in arg.lower():
                example_args.append("example text")
            elif 'url' in arg.lower():
                example_args.append("https://example.com")
            elif 'timeout' in arg.lower():
                example_args.append("10s")
            else:
                example_args.append(f"arg{i+1}")
        
        args_str = "    ".join(example_args)
        return f"{keyword_info.name}    {args_str}"

    def _generate_usage_recommendations(
        self,
        matches: List[KeywordMatch],
        action: str
    ) -> List[str]:
        """Generate usage recommendations based on matches."""
        recommendations = []
        
        if not matches:
            recommendations.append("No matching keywords found. Consider:")
            recommendations.append("- Check if required libraries are imported")
            recommendations.append("- Rephrase the action description")
            recommendations.append("- Use more specific terms")
        else:
            top_match = matches[0]
            recommendations.append(f"Best match: {top_match.keyword_name} (confidence: {top_match.confidence:.2f})")
            
            if top_match.arguments:
                recommendations.append(f"Required arguments: {', '.join(top_match.arguments)}")
            
            if len(matches) > 1:
                recommendations.append(f"Alternative options: {', '.join([m.keyword_name for m in matches[1:4]])}")

        return recommendations

    def _extract_tags_from_doc(self, documentation: str) -> List[str]:
        """Extract tags from keyword documentation."""
        tags = []
        
        # Look for common patterns in documentation
        doc_lower = documentation.lower()
        
        if any(term in doc_lower for term in ['browser', 'web', 'html', 'dom']):
            tags.append('web')
        if any(term in doc_lower for term in ['mobile', 'app', 'touch']):
            tags.append('mobile')
        if any(term in doc_lower for term in ['api', 'http', 'request', 'response']):
            tags.append('api')
        if any(term in doc_lower for term in ['database', 'sql', 'query']):
            tags.append('database')
        if any(term in doc_lower for term in ['click', 'button', 'link']):
            tags.append('interaction')
        if any(term in doc_lower for term in ['verify', 'assert', 'check', 'should']):
            tags.append('verification')
        
        return tags