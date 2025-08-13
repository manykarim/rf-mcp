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
    
    def __init__(self):
        self.action_patterns = {
            'navigate': [
                r'(?:go to|navigate to|open|visit)\s+(.+)',
                r'open\s+(?:the\s+)?(.+?)(?:\s+page|\s+url)?',
            ],
            'click': [
                r'click\s+(?:on\s+)?(?:the\s+)?(.+?)(?:\s+button|\s+link|\s+element)?',
                r'press\s+(?:the\s+)?(.+?)(?:\s+button)?',
                r'select\s+(?:the\s+)?(.+?)(?:\s+option)?',
            ],
            'input': [
                r'(?:enter|type|input)\s+["\'](.+?)["\'](?:\s+into|\s+in)?\s+(?:the\s+)?(.+?)(?:\s+field|\s+box)?',
                r'fill\s+(?:in\s+)?(?:the\s+)?(.+?)(?:\s+field|\s+box)?\s+with\s+["\'](.+?)["\']',
                r'set\s+(?:the\s+)?(.+?)\s+to\s+["\'](.+?)["\']',
            ],
            'verify': [
                r'(?:verify|check|ensure|confirm)\s+(?:that\s+)?(.+)',
                r'(?:should\s+see|should\s+contain|should\s+display)\s+(.+)',
                r'expect\s+(.+)',
                r'assert\s+(.+)',
            ],
            'wait': [
                r'wait\s+(?:for\s+)?(.+?)(?:\s+to\s+(?:appear|be\s+visible|load))?',
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
        
        self.capability_keywords = {
            'SeleniumLibrary': ['browser', 'web', 'selenium', 'chrome', 'firefox'],
            'RequestsLibrary': ['api', 'http', 'request', 'endpoint', 'rest'],
            'DatabaseLibrary': ['database', 'db', 'sql', 'table', 'query'],
            'AppiumLibrary': ['mobile', 'app', 'android', 'ios', 'appium']
        }

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
        """Extract or generate a title for the test scenario."""
        # Try to find a title pattern
        title_patterns = [
            r'^(?:test|verify|check|ensure)\s+(?:that\s+)?(.+?)(?:\.|$)',
            r'^(.+?)(?:\s+test|\s+scenario|\.)',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, scenario.lower())
            if match:
                title = match.group(1).strip()
                return title.capitalize()
        
        # Default title based on first few words
        words = scenario.split()[:6]
        return ' '.join(words) + ('...' if len(scenario.split()) > 6 else '')

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences for action extraction."""
        # Simple sentence splitting - can be enhanced
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

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
        
        # Add based on context
        if context == "web":
            required.add("SeleniumLibrary")
        elif context == "api":
            required.add("RequestsLibrary")
        elif context == "mobile":
            required.add("AppiumLibrary")
        elif context == "database":
            required.add("DatabaseLibrary")
        
        # Add based on keywords found in scenario
        for library, keywords in self.capability_keywords.items():
            if any(keyword in scenario_lower for keyword in keywords):
                required.add(library)
        
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
        """Detect explicit library preference from scenario text."""
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
        if re.search(r'\b(api|http|rest|request)\b', text_lower):
            return "RequestsLibrary"
        
        return None
    
    def _detect_session_type(self, scenario_text: str, context: str) -> str:
        """Detect session type from scenario text and context."""
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