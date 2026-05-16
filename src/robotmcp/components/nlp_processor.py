"""Natural Language Processing component for scenario analysis."""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class TestAction:
    """Represents a single test action."""
    action_type: str  # navigate, click, input, verify, etc.
    description: str
    target: Optional[str] = None
    value: Optional[str] = None
    verification: Optional[str] = None

@dataclass
class TestScenario:
    """Structured representation of a test scenario."""
    title: str
    description: str
    context: str
    actions: List[TestAction]
    preconditions: List[str]
    expected_outcomes: List[str]
    required_capabilities: List[str]

class NaturalLanguageProcessor:
    """Processes natural language test descriptions into structured formats."""

    # Stemming map for common verb forms
    STEM_MAP = {
        # Click variations
        'clicking': 'click', 'clicks': 'click', 'clicked': 'click',
        # Press variations
        'pressing': 'press', 'presses': 'press', 'pressed': 'press',
        # Navigate variations
        'navigating': 'navigate', 'navigates': 'navigate', 'navigated': 'navigate',
        # Open variations
        'opening': 'open', 'opens': 'open', 'opened': 'open',
        # Close variations
        'closing': 'close', 'closes': 'close', 'closed': 'close',
        # Fill variations
        'filling': 'fill', 'fills': 'fill', 'filled': 'fill',
        # Type variations
        'typing': 'type', 'types': 'type', 'typed': 'type',
        # Submit variations
        'submitting': 'submit', 'submits': 'submit', 'submitted': 'submit',
        # Verify variations
        'verifying': 'verify', 'verifies': 'verify', 'verified': 'verify',
        # Check variations
        'checking': 'check', 'checks': 'check', 'checked': 'check',
        # Wait variations
        'waiting': 'wait', 'waits': 'wait', 'waited': 'wait',
        # Select variations
        'selecting': 'select', 'selects': 'select', 'selected': 'select',
        # Enter variations
        'entering': 'enter', 'enters': 'enter', 'entered': 'enter',
        # Scroll variations
        'scrolling': 'scroll', 'scrolls': 'scroll', 'scrolled': 'scroll',
        # Test variations
        'testing': 'test', 'tests': 'test', 'tested': 'test',
        # Assert variations
        'asserting': 'assert', 'asserts': 'assert', 'asserted': 'assert',
        # Validate variations
        'validating': 'validate', 'validates': 'validate', 'validated': 'validate',
    }

    # Synonym groups - first word is canonical
    SYNONYMS = {
        'click': ['click', 'press', 'tap', 'select', 'hit'],
        'input': ['input', 'type', 'enter', 'fill', 'write'],
        'verify': ['verify', 'check', 'assert', 'validate', 'confirm', 'ensure'],
        'navigate': ['navigate', 'go', 'open', 'visit', 'browse', 'access'],
        'wait': ['wait', 'pause', 'delay', 'sleep'],
        'close': ['close', 'quit', 'exit', 'terminate', 'end'],
        'submit': ['submit', 'send', 'post', 'confirm'],
        'scroll': ['scroll', 'swipe', 'drag'],
        'search': ['search', 'find', 'locate', 'query', 'lookup'],
        'login': ['login', 'signin', 'authenticate', 'logon'],
        'logout': ['logout', 'signout', 'logoff'],
    }

    def __init__(self):
        # Proposal-A (A5/A7 helper): action patterns rewritten so the captured
        # target is anchored on the right by either a real suffix word OR end of
        # string. The previous patterns ended with optional groups like
        # ``(?:\s+button)?`` after a non-greedy ``(.+?)`` capture, which causes
        # ``.+?`` to collapse to a single character (it prefers the shortest
        # match satisfying the zero-width optional). New patterns require
        # at least 2 characters in the captured target.
        self.action_patterns = {
            'navigate': [
                # URL first — capture only the URL token (no trailing prose).
                r'(?:go\s+to|navigate\s+to|open|visit|browse\s+to)\s+(https?://\S+)',
                # Fallback (no URL): capture a target up to first comma/period
                # or "in <browser>" prepositional phrase, keeping it bounded.
                r'open\s+(?:the\s+)?(\S{2,}[^,.]*?)(?:\s+page|\s+url|\s+in\s+|$|\.\s+|,)',
            ],
            'click': [
                # Anchor on suffix-word-or-EOL so the non-greedy capture has a
                # real reason to grow past 1 char.
                r'click\s+(?:on\s+)?(?:the\s+)?(\S{2,}.*?)(?:\s+(?:button|link|element|icon|tab|item))?(?:\s*$|\s*\.\s+)',
                r'press\s+(?:the\s+)?(\S{2,}.*?)(?:\s+button)?(?:\s*$|\s*\.\s+)',
                r'select\s+(?:the\s+)?(\S{2,}.*?)(?:\s+option)?(?:\s*$|\s*\.\s+)',
            ],
            'input': [
                r'(?:enter|type|input)\s+["\'](.+?)["\'](?:\s+into|\s+in)?\s+(?:the\s+)?(\S{2,}.*?)(?:\s+(?:field|box))?(?:\s*$|\s*\.\s+)',
                r'fill\s+(?:in\s+)?(?:the\s+)?(\S{2,}.*?)(?:\s+(?:field|box))?\s+with\s+["\'](.+?)["\']',
                r'set\s+(?:the\s+)?(\S{2,}.*?)\s+to\s+["\'](.+?)["\']',
                # Bare-target form ("Fill in vehicle details"). Target only,
                # no value — value is None.
                r'fill\s+(?:in\s+)?(?:the\s+)?(\S{2,}[^.]*?)(?:\s*$|\s*\.\s+)',
            ],
            'submit': [
                r'(?:submit|send)\s+(?:the\s+)?(\S{2,}[^.]*?)(?:\s*$|\s*\.\s+)',
            ],
            'choose': [
                r'(?:choose|pick)\s+(?:the\s+)?(\S{2,}[^.]*?)(?:\s+(?:option|item))?(?:\s*$|\s*\.\s+)',
            ],
            'verify': [
                r'(?:verify|check|ensure|confirm)\s+(?:that\s+)?(.+)',
                r'(?:should\s+see|should\s+contain|should\s+display)\s+(.+)',
                r'expect\s+(.+)',
                r'assert\s+(.+)',
            ],
            'wait': [
                r'wait\s+(?:for\s+)?(.+?)(?:\s+to\s+(?:appear|be\s+visible|load))?(?:\s*$|\s*\.\s+)',
                r'pause\s+(?:for\s+)?(\d+)\s*(?:seconds?|ms|milliseconds?)?',
            ],
            'search': [
                r'search\s+for\s+["\'](.+?)["\']',
                r'find\s+["\'](.+?)["\']',
                r'look\s+for\s+["\'](.+?)["\']',
            ]
        }
        
        self.context_keywords = {
            'web': ['browser', 'website', 'page', 'url', 'dom', 'html', 'css'],
            'mobile': ['app', 'mobile', 'android', 'ios', 'touch', 'swipe'],
            'api': ['api', 'endpoint', 'request', 'response', 'json', 'rest', 'graphql'],
            'database': ['database', 'db', 'table', 'query', 'sql', 'record']
        }
        
        # Proposal-A (A2/A3): tightened capability keywords.
        # - RequestsLibrary: 'request' and 'http' alone are too ambiguous
        #   ("insurance request"; "http://..." matches the protocol of any URL),
        #   so they are removed.  Specific API tokens only.
        # - AppiumLibrary: 'app' is removed (matches sampleapp.tricentis.com
        #   when tokenised). 'mobile'/'android'/'ios'/'appium' remain.
        self.capability_keywords = {
            'SeleniumLibrary': ['selenium', 'webdriver'],
            'RequestsLibrary': ['api', 'rest api', 'endpoint', 'graphql', 'webhook'],
            'DatabaseLibrary': ['database', 'sql', 'mysql', 'postgresql'],
            'AppiumLibrary': ['mobile', 'android', 'ios', 'appium', 'iphone', 'ipad']
        }

    def _stem_word(self, word: str) -> str:
        """Apply simple stemming to a word.

        Args:
            word: Word to stem

        Returns:
            Stemmed word
        """
        return self.STEM_MAP.get(word.lower(), word.lower())

    def _normalize_keywords(self, keywords: List[str]) -> List[str]:
        """Normalize a list of keywords by applying stemming.

        Args:
            keywords: List of keywords to normalize

        Returns:
            Normalized keywords
        """
        return [self._stem_word(kw) for kw in keywords]

    def _expand_synonyms(self, keyword: str) -> List[str]:
        """Expand a keyword to include its synonyms.

        Args:
            keyword: Keyword to expand

        Returns:
            List containing keyword and all synonyms
        """
        keyword_lower = keyword.lower()

        # Check each synonym group
        for canonical, synonyms in self.SYNONYMS.items():
            if keyword_lower in synonyms:
                return synonyms

        return [keyword_lower]

    def _expand_all_synonyms(self, keywords: List[str]) -> List[str]:
        """Expand all keywords to include synonyms.

        Args:
            keywords: List of keywords to expand

        Returns:
            Expanded list with synonyms included
        """
        expanded = set()
        for kw in keywords:
            expanded.update(self._expand_synonyms(kw))
        return list(expanded)

    def _fuzzy_match(self, text: str, pattern: str, threshold: float = 0.85) -> bool:
        """Check if text fuzzy matches pattern.

        Args:
            text: Text to check
            pattern: Pattern to match
            threshold: Similarity threshold (0-1)

        Returns:
            True if similarity >= threshold
        """
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, text.lower(), pattern.lower()).ratio()
        return ratio >= threshold

    async def analyze_scenario(self, scenario: str, context: str = "web") -> Dict[str, Any]:
        """
        Analyze a natural language scenario and extract structured test information.
        
        Args:
            scenario: Natural language test description
            context: Application context (web, mobile, api, database)
            
        Returns:
            Dictionary containing structured test scenario
        """
        try:
            # Clean and normalize the scenario text
            normalized_scenario = self._normalize_text(scenario)
            
            # Extract title from first sentence or create one
            title = self._extract_title(normalized_scenario)
            
            # Split scenario into sentences for action extraction
            sentences = self._split_sentences(normalized_scenario)
            
            # Extract actions from sentences
            actions = []
            for sentence in sentences:
                action = self._extract_action(sentence)
                if action:
                    actions.append(action)
            
            # Extract preconditions and expected outcomes
            preconditions = self._extract_preconditions(normalized_scenario)
            expected_outcomes = self._extract_expected_outcomes(normalized_scenario)
            
            # Determine required capabilities
            required_capabilities = self._determine_capabilities(normalized_scenario, context)
            
            # Detect explicit library preferences
            explicit_library_preference = self._detect_explicit_library_preference(normalized_scenario)
            session_type = self._detect_session_type(normalized_scenario, context)
            
            # Build structured scenario
            structured_scenario = TestScenario(
                title=title,
                description=scenario.strip(),
                context=context,
                actions=actions,
                preconditions=preconditions,
                expected_outcomes=expected_outcomes,
                required_capabilities=required_capabilities
            )
            
            return {
                "success": True,
                "scenario": {
                    "title": structured_scenario.title,
                    "description": structured_scenario.description,
                    "context": structured_scenario.context,
                    "actions": [
                        {
                            "action_type": action.action_type,
                            "description": action.description,
                            "target": action.target,
                            "value": action.value,
                            "verification": action.verification
                        } for action in structured_scenario.actions
                    ],
                    "preconditions": structured_scenario.preconditions,
                    "expected_outcomes": structured_scenario.expected_outcomes,
                    "required_capabilities": structured_scenario.required_capabilities
                },
                "analysis": {
                    "action_count": len(actions),
                    "complexity": self._assess_complexity(actions),
                    "estimated_steps": len(actions) * 2,  # Rough estimate
                    "suggested_libraries": required_capabilities,
                    "explicit_library_preference": explicit_library_preference,
                    "detected_session_type": session_type
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing scenario: {e}")
            return {
                "success": False,
                "error": str(e),
                "scenario": None
            }

    async def suggest_next_step(
        self,
        current_state: Dict[str, Any],
        test_objective: str,
        executed_steps: List[Dict[str, Any]],
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Suggest the next test step based on current state and objective.
        
        Args:
            current_state: Current application state
            test_objective: Overall test objective
            executed_steps: Previously executed steps
            session_id: Session identifier
            
        Returns:
            Suggested next steps and recommendations
        """
        try:
            # Analyze current progress
            progress = self._analyze_progress(executed_steps, test_objective)
            
            # Determine what's available in current state
            available_elements = self._extract_available_elements(current_state)
            
            # Generate suggestions based on objective and state
            suggestions = self._generate_step_suggestions(
                test_objective, current_state, executed_steps, available_elements
            )
            
            # Rank suggestions by confidence
            ranked_suggestions = self._rank_suggestions(suggestions, current_state)
            
            return {
                "success": True,
                "suggestions": ranked_suggestions,
                "progress": progress,
                "available_elements": available_elements,
                "recommended_verifications": self._suggest_verifications(current_state)
            }
            
        except Exception as e:
            logger.error(f"Error suggesting next step: {e}")
            return {
                "success": False,
                "error": str(e),
                "suggestions": []
            }

    async def validate_scenario(
        self,
        parsed_scenario: Dict[str, Any],
        available_libraries: List[str] = None
    ) -> Dict[str, Any]:
        """
        Validate scenario feasibility and suggest missing capabilities.
        
        Args:
            parsed_scenario: Parsed scenario from analyze_scenario
            available_libraries: List of available Robot Framework libraries
            
        Returns:
            Validation results and recommendations
        """
        try:
            if available_libraries is None:
                available_libraries = []
            
            scenario = parsed_scenario.get("scenario", {})
            required_capabilities = scenario.get("required_capabilities", [])
            actions = scenario.get("actions", [])
            
            # Check capability availability
            missing_capabilities = [
                cap for cap in required_capabilities 
                if cap not in available_libraries
            ]
            
            # Validate actions
            validation_issues = []
            for i, action in enumerate(actions):
                issues = self._validate_action(action, available_libraries)
                if issues:
                    validation_issues.extend([f"Action {i+1}: {issue}" for issue in issues])
            
            # Assess overall feasibility
            feasibility_score = self._calculate_feasibility_score(
                actions, available_libraries, missing_capabilities
            )
            
            return {
                "success": True,
                "feasible": feasibility_score > 0.7,
                "feasibility_score": feasibility_score,
                "missing_capabilities": missing_capabilities,
                "validation_issues": validation_issues,
                "recommendations": self._generate_recommendations(
                    missing_capabilities, validation_issues
                )
            }
            
        except Exception as e:
            logger.error(f"Error validating scenario: {e}")
            return {
                "success": False,
                "error": str(e),
                "feasible": False
            }

    def _normalize_text(self, text: str) -> str:
        """Normalize text for processing."""
        # Remove extra whitespace and normalize quotes
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        return text

    def _extract_title(self, scenario: str) -> str:
        """Extract or generate a title for the test scenario.

        Proposal-A (A6): a leading "Open https://X/..." used to be truncated at
        the first dot in the URL (yielding "Open https://sampleapp" — a
        meaningless title). Instead, if the scenario opens with a URL, use the
        URL's host as the title, optionally prefixed by the verb.
        """
        # A6: leading-URL handling — title becomes the host
        leading_url_match = re.match(
            r'^\s*(?:open|navigate\s+to|go\s+to|visit|browse\s+to)\s+(https?://([^/\s]+))',
            scenario,
            re.IGNORECASE,
        )
        if leading_url_match:
            host = leading_url_match.group(2)
            return host.capitalize()

        # Try to find a title pattern. Use a sentence terminator that
        # ignores periods INSIDE a URL (e.g. sampleapp.tricentis.com/101/).
        # "(?:\.\s+|$)" only matches a period followed by whitespace or EOL.
        title_patterns = [
            r'^(?:test|verify|check|ensure)\s+(?:that\s+)?(.+?)(?:\.\s+|$)',
            r'^(.+?)(?:\s+test|\s+scenario|\.\s+|$)',
        ]

        for pattern in title_patterns:
            match = re.search(pattern, scenario.lower())
            if match:
                title = match.group(1).strip()
                if title:
                    return title.capitalize()

        # Default title based on first few words
        words = scenario.split()[:6]
        return ' '.join(words) + ('...' if len(scenario.split()) > 6 else '')

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences for action extraction.

        Proposal-A (A7): the previous implementation split only on .!?,
        leaving long compound sentences ("Click X, then fill Y, then click Z")
        as a single sentence and dropping all but one action. We additionally
        split on sequencing words (then/after/next/finally), semicolons, and
        commas that precede a verb. Periods inside URLs are preserved by
        splitting only on a period followed by whitespace.
        """
        # Split on real sentence terminators (period must be followed by
        # whitespace so dots in URLs do not split).
        parts = re.split(r'(?:[!?]+|\.\s+)', text)

        # Further split each part on explicit sequencing words and semicolons.
        # Note: "next" as a sequencer must be preceded by a comma/space-only
        # context, NOT used as the object of a click ("Click Next"). We require
        # ", next" or "; next" or "then next" — but a bare "Click Next" is left
        # intact.
        sequencing = re.compile(
            r'\s*(?:;|\bthen\b|\bafter\s+that\b|\bafter\b|\bfinally,?\b|\band\s+then\b)\s*',
            re.IGNORECASE,
        )
        action_verb_at_start = re.compile(
            r'^\s*(?:click|open|navigate|go\s+to|visit|browse|fill|enter|type|input|set|select|press|verify|check|ensure|confirm|expect|assert|wait|search|find|hover|drag|drop|upload|download|submit|send|choose|pick|complete)\b',
            re.IGNORECASE,
        )
        out: List[str] = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            sub_parts = sequencing.split(p)
            for sp in sub_parts:
                sp = sp.strip()
                if not sp:
                    continue
                # Commas before an action verb also split.
                comma_parts = re.split(r',\s+(?=\w)', sp)
                kept_any = False
                for cp in comma_parts:
                    cp = cp.strip().rstrip(',')
                    if cp and action_verb_at_start.search(cp):
                        out.append(cp)
                        kept_any = True
                if not kept_any:
                    out.append(sp)
        return out

    def _extract_action(self, sentence: str) -> Optional[TestAction]:
        """Extract action from a sentence."""
        sentence_lower = sentence.lower().strip()
        
        for action_type, patterns in self.action_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, sentence_lower)
                if match:
                    groups = match.groups()
                    
                    if action_type == 'input' and len(groups) >= 2:
                        # Handle input patterns with value and target
                        if 'fill' in pattern or 'set' in pattern:
                            target, value = groups[0], groups[1]
                        else:
                            value, target = groups[0], groups[1]
                        
                        return TestAction(
                            action_type=action_type,
                            description=sentence,
                            target=target.strip(),
                            value=value.strip()
                        )
                    else:
                        target = groups[0] if groups else None
                        if target:
                            # Strip trailing punctuation that leaked past the
                            # boundary regex (e.g. "quote." -> "quote").
                            target = target.rstrip('.,;: \t')
                        return TestAction(
                            action_type=action_type,
                            description=sentence,
                            target=target.strip() if target else None
                        )
        
        return None

    def _extract_preconditions(self, scenario: str) -> List[str]:
        """Extract preconditions from scenario."""
        precondition_patterns = [
            r'(?:given|assuming|provided)\s+(.+?)(?:\.|,|$)',
            r'(?:before|first|initially)\s+(.+?)(?:\.|,|$)',
            r'(?:prerequisite|requirement):\s*(.+?)(?:\.|,|$)'
        ]
        
        preconditions = []
        for pattern in precondition_patterns:
            matches = re.findall(pattern, scenario.lower())
            preconditions.extend([match.strip() for match in matches])
        
        return preconditions

    def _extract_expected_outcomes(self, scenario: str) -> List[str]:
        """Extract expected outcomes from scenario."""
        outcome_patterns = [
            r'(?:should|must|will|expect)\s+(.+?)(?:\.|,|$)',
            r'(?:result|outcome):\s*(.+?)(?:\.|,|$)',
            r'(?:then|finally)\s+(.+?)(?:\.|,|$)'
        ]
        
        outcomes = []
        for pattern in outcome_patterns:
            matches = re.findall(pattern, scenario.lower())
            outcomes.extend([match.strip() for match in matches])
        
        return outcomes

    def _determine_capabilities(self, scenario: str, context: str) -> List[str]:
        """Determine required Robot Framework libraries."""
        scenario_lower = scenario.lower()
        required = set()
        
        # Add based on context - FIXED: Default to Browser Library for modern web automation
        if context == "web":
            # Check for explicit library preference first
            if any(pattern in scenario_lower for pattern in [
                "selenium", "seleniumlibrary", "webdriver"
            ]):
                required.add("SeleniumLibrary")
            else:
                # Default to Browser Library for modern web automation (matches recommend_libraries logic)
                required.add("Browser")
        elif context == "api":
            required.add("RequestsLibrary")
        elif context == "mobile":
            required.add("AppiumLibrary")
        elif context == "database":
            required.add("DatabaseLibrary")
        
        # Proposal-A (A3): switch from substring containment to whole-word match.
        # The previous logic matched 'app' inside 'application' and 'rest' inside
        # 'restroom', polluting required_capabilities with mobile/api libs on
        # plain web scenarios.
        for library, keywords in self.capability_keywords.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', scenario_lower):
                    required.add(library)
                    break

        return list(required)

    def _assess_complexity(self, actions: List[TestAction]) -> str:
        """Assess the complexity of the test scenario."""
        if len(actions) <= 3:
            return "simple"
        elif len(actions) <= 7:
            return "medium"
        else:
            return "complex"

    def _analyze_progress(self, executed_steps: List[Dict[str, Any]], objective: str) -> Dict[str, Any]:
        """Analyze progress towards test objective."""
        total_steps = len(executed_steps)
        successful_steps = sum(1 for step in executed_steps if step.get("status") == "pass")
        
        return {
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "completion_ratio": successful_steps / total_steps if total_steps > 0 else 0,
            "current_phase": self._determine_current_phase(executed_steps, objective)
        }

    def _extract_available_elements(self, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract available UI elements from current state."""
        elements = []
        
        # Extract from DOM state if available
        dom_state = current_state.get("dom", {})
        if "elements" in dom_state:
            for element in dom_state["elements"]:
                elements.append({
                    "type": element.get("tag", "unknown"),
                    "text": element.get("text", ""),
                    "id": element.get("id"),
                    "class": element.get("class"),
                    "clickable": element.get("clickable", False),
                    "visible": element.get("visible", True)
                })
        
        return elements

    def _generate_step_suggestions(
        self,
        objective: str,
        current_state: Dict[str, Any],
        executed_steps: List[Dict[str, Any]],
        available_elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate step suggestions based on current context."""
        suggestions = []
        
        # Analyze objective for action hints
        objective_lower = objective.lower()
        
        # Look for clickable elements that match objective
        for element in available_elements:
            if element.get("clickable") and element.get("visible"):
                element_text = element.get("text", "").lower()
                if any(word in element_text for word in objective_lower.split()):
                    suggestions.append({
                        "action": "click",
                        "target": element_text or f"element with id '{element.get('id')}'",
                        "confidence": 0.8,
                        "reason": f"Clickable element '{element_text}' matches objective"
                    })
        
        # Suggest common verification steps
        if not any(step.get("keyword", "").startswith("Page Should") for step in executed_steps):
            suggestions.append({
                "action": "verify",
                "target": "page content",
                "confidence": 0.6,
                "reason": "Verify page loaded correctly"
            })
        
        return suggestions

    def _rank_suggestions(
        self,
        suggestions: List[Dict[str, Any]],
        current_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Rank suggestions by confidence and relevance."""
        return sorted(suggestions, key=lambda x: x.get("confidence", 0), reverse=True)

    def _suggest_verifications(self, current_state: Dict[str, Any]) -> List[str]:
        """Suggest verification steps based on current state."""
        verifications = []
        
        # Basic page verifications
        if "dom" in current_state:
            verifications.extend([
                "Verify page title",
                "Verify page URL",
                "Verify key elements are visible"
            ])
        
        # API response verifications
        if "api" in current_state:
            verifications.extend([
                "Verify response status code",
                "Verify response structure",
                "Verify response data"
            ])
        
        return verifications
    
    def _detect_explicit_library_preference(self, scenario_text: str) -> Optional[str]:
        """Detect explicit library preference using centralized LibraryDetector."""
        try:
            from robotmcp.utils.library_detection import detect_library_preference
            detected = detect_library_preference(scenario_text, min_score=5)
            if detected:
                return detected
        except ImportError:
            pass

        # Fallback to local patterns for backward compatibility
        return self._fallback_detect_library_preference(scenario_text)

    def _fallback_detect_library_preference(self, scenario_text: str) -> Optional[str]:
        """Fallback library preference detection using local patterns.

        Proposal-A negative-evidence rule (A2): the bare word ``request`` is too
        ambiguous to imply API testing on its own (it appears in product/business
        text like "insurance request"). RequestsLibrary is only returned when API
        vocabulary is present AND a generic web URL is not the dominant signal.
        """
        if not scenario_text:
            return None

        text_lower = scenario_text.lower()

        # Selenium patterns (highest priority for explicit mentions)
        selenium_patterns = [
            r'\b(use|using|with)\s+(selenium|seleniumlibrary|selenium\s*library)\b',
            r'\bselenium\b(?!.*browser)',  # Selenium mentioned but not "selenium browser"
            r'\bseleniumlibrary\b',
        ]

        # Browser Library patterns
        browser_patterns = [
            r'\b(use|using|with)\s+(browser|browserlibrary|browser\s*library|playwright)\b',
            r'\bbrowser\s*library\b',
            r'\bplaywright\b',
        ]

        # Check for explicit Selenium preference first
        for pattern in selenium_patterns:
            if re.search(pattern, text_lower):
                logger.info(f"NLP: Detected explicit SeleniumLibrary preference: {pattern}")
                return "SeleniumLibrary"

        # Check for explicit Browser Library preference
        for pattern in browser_patterns:
            if re.search(pattern, text_lower):
                logger.info(f"NLP: Detected explicit Browser Library preference: {pattern}")
                return "Browser"

        # Check for other library preferences
        if re.search(r'\b(xml|xpath)\b', text_lower):
            return "XML"

        # A1+A2: A web URL is strong evidence of web automation. The bare word
        # "request" alone (e.g. "insurance request") is not sufficient to imply
        # API testing. Require a real API vocabulary signal AND tolerate URL
        # presence only when an API word is also present.
        has_web_url = bool(re.search(r'https?://\S+', text_lower))
        api_word_match = re.search(
            r'\b(api|http|rest|endpoint|graphql|webhook|oauth|jwt|webservice|web\s+service|microservice)\b',
            text_lower,
        )
        if api_word_match and not has_web_url:
            return "RequestsLibrary"
        if api_word_match and has_web_url:
            # Both signals present: API word wins only if it's stronger than a
            # single ambiguous mention (e.g. "http" or "rest"). Without further
            # disambiguation we keep "RequestsLibrary" only when the API word
            # is unambiguous (api/rest/endpoint/graphql/webhook/oauth/jwt).
            unambiguous = re.search(
                r'\b(api|rest|endpoint|graphql|webhook|oauth|jwt|webservice|web\s+service|microservice)\b',
                text_lower,
            )
            if unambiguous:
                return "RequestsLibrary"
        return None
    
    def _detect_session_type(self, scenario: str, context: str) -> str:
        """Detect session type using centralized session models detection."""
        try:
            from robotmcp.models.session_models import ExecutionSession
            temp_session = ExecutionSession(session_id="__nlp_detect__")
            session_type = temp_session.detect_session_type_from_scenario(scenario)
            if session_type and session_type.value != "unknown":
                return session_type.value
        except (ImportError, Exception):
            pass

        # Fallback to local patterns
        return self._fallback_detect_session_type(scenario, context)

    def _fallback_detect_session_type(self, scenario_text: str, context: str) -> str:
        """Fallback session type detection using local patterns."""
        if not scenario_text:
            return "unknown"

        text_lower = scenario_text.lower()

        # Web automation patterns
        web_patterns = [
            r'\b(click|fill|navigate|browser|page|element|locator)\b',
            r'\b(new page|go to|wait for|screenshot)\b',
            r'\b(get text|get attribute|should contain)\b'
        ]

        # API testing patterns
        api_patterns = [
            r'\b(get request|post|put|delete|api|http)\b',
            r'\b(create session|request|response|status)\b',
            r'\b(json|rest|endpoint)\b'
        ]

        # XML processing patterns
        xml_patterns = [
            r'\b(parse|xml|xpath|element|attribute)\b',
            r'\b(get element|set element|xml)\b'
        ]

        # Count matches for each type
        web_score = sum(len(re.findall(pattern, text_lower)) for pattern in web_patterns)
        api_score = sum(len(re.findall(pattern, text_lower)) for pattern in api_patterns)
        xml_score = sum(len(re.findall(pattern, text_lower)) for pattern in xml_patterns)

        # Determine session type based on highest score
        scores = {"web_automation": web_score, "api_testing": api_score, "xml_processing": xml_score}

        # Consider context as a tie-breaker
        if context == "web":
            scores["web_automation"] += 1
        elif context == "api":
            scores["api_testing"] += 1

        if max(scores.values()) == 0:
            return "unknown"

        return max(scores, key=scores.get)

    def _validate_action(self, action: Dict[str, Any], available_libraries: List[str]) -> List[str]:
        """Validate a single action for feasibility."""
        issues = []
        action_type = action.get("action_type")
        
        # Check if required library is available
        if action_type in ["click", "input", "navigate"] and "SeleniumLibrary" not in available_libraries:
            issues.append("SeleniumLibrary required for web actions")
        
        # Check for missing target
        if action_type in ["click", "input"] and not action.get("target"):
            issues.append("Target element not specified")
        
        # Check for missing value in input actions
        if action_type == "input" and not action.get("value"):
            issues.append("Input value not specified")
        
        return issues

    def _calculate_feasibility_score(
        self,
        actions: List[Dict[str, Any]],
        available_libraries: List[str],
        missing_capabilities: List[str]
    ) -> float:
        """Calculate overall feasibility score."""
        if not actions:
            return 0.0
        
        # Base score
        score = 1.0
        
        # Reduce score for missing capabilities
        if missing_capabilities:
            score -= len(missing_capabilities) * 0.2
        
        # Reduce score for validation issues
        total_issues = 0
        for action in actions:
            issues = self._validate_action(action, available_libraries)
            total_issues += len(issues)
        
        if total_issues > 0:
            score -= min(total_issues * 0.1, 0.5)
        
        return max(score, 0.0)

    def _generate_recommendations(
        self,
        missing_capabilities: List[str],
        validation_issues: List[str]
    ) -> List[str]:
        """Generate recommendations for improving scenario feasibility."""
        recommendations = []
        
        if missing_capabilities:
            recommendations.append(
                f"Install missing libraries: {', '.join(missing_capabilities)}"
            )
        
        if validation_issues:
            recommendations.append("Review and fix validation issues:")
            recommendations.extend([f"  - {issue}" for issue in validation_issues])
        
        if not missing_capabilities and not validation_issues:
            recommendations.append("Scenario appears feasible - proceed with execution")
        
        return recommendations

    def _determine_current_phase(self, executed_steps: List[Dict[str, Any]], objective: str) -> str:
        """Determine the current phase of test execution."""
        if not executed_steps:
            return "initialization"
        
        last_step = executed_steps[-1]
        keyword = last_step.get("keyword", "").lower()
        
        if "open" in keyword or "navigate" in keyword:
            return "navigation"
        elif "click" in keyword or "input" in keyword:
            return "interaction"
        elif "should" in keyword or "verify" in keyword:
            return "verification"
        else:
            return "execution"