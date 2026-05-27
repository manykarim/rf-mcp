"""Keyword Matcher component for semantic matching of Robot Framework keywords."""

import os
import re
import logging
from typing import Any, Dict, List, Optional, Tuple
import dataclasses
from dataclasses import dataclass, field
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
    # OBS-18B — RF tags carried through the matcher pipeline so the
    # action-class reranker can classify keywords (per OBS-18A v2
    # design). Default empty list to keep existing call-sites
    # backward-compatible.
    tags: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# OBS-18B — Action-class reranker (per OBS-18A v2 design)
# ----------------------------------------------------------------------------
#
# The matcher's pre-rerank confidence reflects per-strategy local optima
# (SequenceMatcher overlap + tag-boost + pattern similarity). High
# confidence doesn't always mean the keyword performs the queried action
# — e.g., S02 ("select dropdown option by visible label") ranks
# `Element Should Be Visible` at 0.87 because "visible" matches both
# query and keyword name.
#
# The reranker classifies BOTH the query and each keyword into one of
# 9 action classes, then down-weights mismatches. A separate confidence
# cap fires when (A) top-3 spans ≥3 distinct classes (uncertain
# matcher) OR (B) query is `unknown` and the top match has an
# opinionated class above the cap threshold.
#
# Feature flag: ROBOTMCP_MATCHER_RERANK (off → no-op; on → reranker
# active). Defaults to on in v0.34+ per OBS-18A v2 rollback plan.


_QUERY_TRIGGERS: Dict[str, Tuple[str, ...]] = {
    # Order matters when triggers overlap; first match wins.
    "wait":     ("wait until", "wait for", "wait ", "sleep", "pause",
                 "delay", "timeout"),
    "navigate": ("go to", "navigate", "visit", "open page", "open url",
                 "load url", "new page"),
    "select":   ("select dropdown", "select option", "select from list",
                 "choose dropdown", "pick from", "dropdown option"),
    "fill":     ("fill ", "type ", "enter ", "input ", "set value",
                 "set text", "write "),
    "assert":   ("should ", "verify ", "must be", "ensure ", "expect ",
                 "page contains", "page should"),
    "query":    ("get ", "read ", "fetch ", "retrieve ", "current ",
                 "value of", "count of"),
    "control":  ("iterate", "loop ", "repeat ", "for each",
                 "conditionally", "run keyword"),
    "click":    ("click", "press", "tap", "push button", "hit"),
}


def _classify_query_action_class(query: str) -> str:
    """Return action class for a natural-language query.

    Determines intent from trigger phrases. Order matters — more
    specific intents (wait, navigate) checked before broad ones
    (click). Returns 'unknown' when no trigger matches.
    """
    q = (query or "").lower().strip()
    if not q:
        return "unknown"
    for cls, triggers in _QUERY_TRIGGERS.items():
        for trig in triggers:
            if trig in q:
                return cls
    return "unknown"


def classify_keyword_action(name: str, tags: List[str]) -> str:
    """Return action class for a keyword. Deterministic; same inputs
    always produce same output.

    Precedence (per OBS-18A v2 design):
      Wait → BrowserControl → Getter → Assertion → Setter+name → name-pattern

    Tag values are case-normalised + space-stripped, so both
    ``PageContent`` and ``Page Content`` map to the same set entry.
    None entries in tags are filtered defensively.
    """
    name_lower = (name or "").lower()
    tag_set = {
        (t or "").lower().replace(" ", "")
        for t in (tags or [])
        if t
    }

    # Priority 1: Wait wins (most specific intent)
    if "wait" in tag_set:
        return "wait"
    # Priority 2: BrowserControl wins over Setter (Go To, New Page tagged BOTH)
    if "browsercontrol" in tag_set:
        return "navigate"
    # Priority 3: Getter wins over Assertion when co-tagged
    if "getter" in tag_set:
        return "query"
    # Priority 4: Pure Assertion (no Getter) — Should*, Page Should*
    if "assertion" in tag_set:
        return "assert"
    # Priority 5: Setter — context-dependent via name pattern
    if "setter" in tag_set:
        if name_lower.startswith((
            "select options", "select option", "select from",
            "select checkbox", "deselect options", "deselect checkbox",
        )):
            return "select"
        if name_lower.startswith((
            "fill text", "fill secret", "type text", "type secret",
            "input text", "input password", "input secret",
            "set text", "press keys",
        )):
            return "fill"
        return "click"  # default Setter

    # Priority 6: name-pattern fallback (SL, BuiltIn, resource kws)
    if name_lower.startswith("wait") or name_lower == "sleep":
        return "wait"
    if name_lower.startswith((
        "go to", "navigate", "new page", "new browser",
        "new context", "new persistent context",
        "open browser", "open page",
    )):
        return "navigate"
    if name_lower.startswith(("select from", "select options",
                              "deselect", "select checkbox")):
        return "select"
    if name_lower.startswith(("click", "tap", "press", "double click")):
        return "click"
    if name_lower.startswith(("fill", "type", "input text", "type text",
                              "input password", "set text", "press keys",
                              "send keys")):
        return "fill"
    # Pure-assertion check before pure-query so ``Element Should Be
    # Visible`` etc. don't get mis-classified as control.
    if (name_lower.startswith(("should ", "page should",
                               "element should"))
            or " should be " in f" {name_lower} "):
        return "assert"
    if name_lower.startswith(("get ", "fetch ", "read ")):
        return "query"
    if name_lower.startswith(("run keyword", "repeat keyword",
                              "for each", "evaluate")):
        return "control"

    return "unknown"


def _reranker_enabled() -> bool:
    """Feature flag — ROBOTMCP_MATCHER_RERANK env var. Defaults to
    ON (1) per OBS-18A v2 rollback plan. Set to '0' / 'false' / 'off'
    to disable.
    """
    val = os.getenv("ROBOTMCP_MATCHER_RERANK", "1").strip().lower()
    return val not in ("0", "false", "off", "no", "")


def apply_action_class_reranker(
    matches: List[KeywordMatch],
    query_action_class: str,
) -> List[KeywordMatch]:
    """Down-weight matches whose action class doesn't match the
    query's class. Caller-supplied confidence cap is applied
    separately by ``apply_confidence_cap``.

    Uses ``dataclasses.replace`` for the down-weight rebuild (per
    Codex/Claude round-1 perf review — ~5x faster than rebuilding
    7 explicit kwargs).
    """
    if query_action_class == "unknown" or not matches:
        # Abstain — preserve matcher's original ranking.
        return matches
    downweight = float(os.getenv("ROBOTMCP_RERANK_DOWNWEIGHT", "0.6"))
    reranked: List[KeywordMatch] = []
    for m in matches:
        kw_class = classify_keyword_action(m.keyword_name, m.tags)
        if kw_class == query_action_class:
            reranked.append(m)  # confidence unchanged
        else:
            reranked.append(dataclasses.replace(
                m, confidence=m.confidence * downweight,
            ))
    # Re-sort by the new confidence
    return sorted(reranked, key=lambda x: x.confidence, reverse=True)


def apply_confidence_cap_dict(
    matches: List[Dict[str, Any]],
    query_action_class: str,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Variant of ``apply_confidence_cap`` operating on the dict shape
    used post-library-filter in ``server.find_keywords``. Used by
    OBS-18B to re-apply the cap to the actual user-visible top match
    after library filtering may have shuffled the matcher's top entry.

    Returns (matches, low_confidence_top_match_flag).
    """
    if not matches:
        return matches, False
    cap = float(os.getenv("ROBOTMCP_RERANK_CAP", "0.5"))
    top = matches[0]
    top_name = top.get("keyword_name") or top.get("name") or ""
    top_tags = top.get("tags") or []
    top_conf = float(top.get("confidence", 0.0))
    top_class = classify_keyword_action(top_name, top_tags)

    trigger_b = (
        query_action_class == "unknown"
        and top_class != "unknown"
    )

    trigger_a = False
    if len(matches) >= 3:
        top3_classes = set()
        for m in matches[:3]:
            nm = m.get("keyword_name") or m.get("name") or ""
            tg = m.get("tags") or []
            top3_classes.add(classify_keyword_action(nm, tg))
        trigger_a = len(top3_classes) >= 3

    if not (trigger_a or trigger_b):
        return matches, False

    if top_conf > cap:
        capped = dict(top)
        capped["confidence"] = cap
        return [capped] + list(matches[1:]), True
    return matches, True


def apply_confidence_cap(
    matches: List[KeywordMatch],
    query_action_class: str,
) -> Tuple[List[KeywordMatch], bool]:
    """Cap top-match confidence under two trigger conditions:

    - Trigger A: top-3 spans ≥3 distinct action classes (divergent
      ranking — matcher is uncertain).
    - Trigger B (OBS-18A v2 — closes the S10 gap): query class is
      ``unknown`` AND top match has an opinionated class AND its
      confidence > cap threshold.

    Returns (matches, low_confidence_top_match_flag).
    """
    if not matches:
        return matches, False
    cap = float(os.getenv("ROBOTMCP_RERANK_CAP", "0.5"))
    top = matches[0]
    top_class = classify_keyword_action(top.keyword_name, top.tags)

    # Trigger B — unknown query + opinionated top → matcher reaches
    # for a class the query doesn't ask for. Flag fires regardless of
    # whether confidence happens to be above the cap (the agent
    # benefits from knowing the matcher is uncertain). The actual
    # confidence cap only applies when conf > cap.
    trigger_b = (
        query_action_class == "unknown"
        and top_class != "unknown"
    )

    # Trigger A — divergent top-3
    trigger_a = False
    if len(matches) >= 3:
        top3_classes = {
            classify_keyword_action(m.keyword_name, m.tags)
            for m in matches[:3]
        }
        trigger_a = len(top3_classes) >= 3

    if not (trigger_a or trigger_b):
        return matches, False

    if top.confidence > cap:
        capped_top = dataclasses.replace(top, confidence=cap)
        return [capped_top] + list(matches[1:]), True
    # Top is already at/below cap; still surface the flag so agents
    # know the matcher is uncertain.
    return matches, True

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

        # OBS-30 — Initialize embeddings model if the optional
        # ``[semantic]`` extra is installed. Log the mode clearly so
        # operators can tell whether semantic ranking is using
        # embeddings or the difflib + tag-based fallback.
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info(
                    "find_keywords semantic strategy: embedding similarity "
                    "ACTIVE (sentence-transformers installed; model "
                    "all-MiniLM-L6-v2)"
                )
            except Exception as e:
                logger.warning(f"Could not load embeddings model: {e}")
                self.embeddings_model = None
                logger.info(
                    "find_keywords semantic strategy: embedding similarity "
                    "DISABLED (load error). Falling back to pattern + tag "
                    "+ difflib SequenceMatcher ranking."
                )
        else:
            logger.info(
                "find_keywords semantic strategy: embedding similarity "
                "DISABLED (sentence-transformers not installed). Falling "
                "back to pattern + tag + difflib SequenceMatcher ranking. "
                "Install via ``uv add robotmcp[semantic]`` for embedding-"
                "based ranking."
            )
        
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
        current_state: Dict[str, Any] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Discover matching Robot Framework keywords for an action description.

        Args:
            action_description: Natural language description of the action
            context: Application context (web, mobile, api, database)
            current_state: Current application state
            limit: Maximum number of ranked matches to return. Defaults
                   to 10 when not provided. OBS-22 — without this
                   parameter, semantic strategy silently capped at 10
                   regardless of caller intent.

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

            # OBS-18B — Action-class reranker + confidence cap.
            # Gated by ROBOTMCP_MATCHER_RERANK env var (default ON).
            # Pipeline order:
            #   5. _rank_matches (sort by raw confidence) — existing
            #   6. apply_action_class_reranker (NEW) — penalise mismatch
            #   7. apply_confidence_cap (NEW) — surface uncertainty
            #   8. limit slice (OBS-22) — caller-supplied trim
            low_confidence_top_match = False
            if _reranker_enabled() and ranked_matches:
                query_class = _classify_query_action_class(action_description)
                ranked_matches = apply_action_class_reranker(
                    ranked_matches, query_class,
                )
                ranked_matches, low_confidence_top_match = apply_confidence_cap(
                    ranked_matches, query_class,
                )

            # OBS-22 — honour caller-supplied limit (default 10).
            effective_limit = limit if (isinstance(limit, int) and limit > 0) else 10
            top_matches = ranked_matches[:effective_limit]

            response = {
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
                        "usage_example": match.usage_example,
                        # OBS-18B — propagate tags so the find_keywords
                        # post-library-filter cap can re-classify the
                        # actual surviving top match.
                        "tags": list(match.tags or []),
                    } for match in top_matches
                ],
                "total_matches": len(unique_matches),
                "recommendations": self._generate_usage_recommendations(top_matches, normalized_action)
            }
            # OBS-18B — surface the cap flag at the matcher-response
            # level. find_keywords wraps under `result` so agents see
            # ``result.low_confidence_top_match: true`` when the cap
            # fires.
            if low_confidence_top_match:
                response["low_confidence_top_match"] = True
            return response

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
                                    usage_example=self._generate_usage_example(keyword_info, action),
                                    tags=list(keyword_info.tags or []),
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
                            usage_example=self._generate_usage_example(keyword_info, action),
                            tags=list(keyword_info.tags or []),
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
                        usage_example=self._generate_usage_example(keyword_info, action),
                        tags=list(keyword_info.tags or []),
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