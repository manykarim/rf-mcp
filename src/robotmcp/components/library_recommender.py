"""Library Recommender component for suggesting Robot Framework libraries based on use cases."""

import re
import logging
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

@dataclass
class LibraryInfo:
    """Information about a Robot Framework library."""
    name: str
    package_name: str
    installation_command: str
    use_cases: List[str]
    categories: List[str]
    description: str
    is_builtin: bool = False
    requires_setup: bool = False
    setup_commands: List[str] = None
    platform_requirements: List[str] = None
    dependencies: List[str] = None

@dataclass
class LibraryRecommendation:
    """A library recommendation with metadata."""
    library: LibraryInfo
    confidence: float
    matching_keywords: List[str]
    rationale: str

class LibraryRecommender:
    """Recommends Robot Framework libraries based on user scenarios and use cases."""
    
    def __init__(self):
        self.libraries_registry: Dict[str, LibraryInfo] = {}
        self.use_case_mapping: Dict[str, Set[str]] = {}
        self.category_mapping: Dict[str, Set[str]] = {}
        self._initialized = False
        
    def _initialize_registry(self) -> None:
        """Initialize the library registry with all available libraries."""
        if self._initialized:
            return
            
        # Core/Built-in Libraries
        builtin_libs = [
            LibraryInfo(
                name="BuiltIn",
                package_name="robotframework",
                installation_command="Built-in with Robot Framework",
                use_cases=["basic operations", "variables", "control flow", "logging", "keywords"],
                categories=["core", "builtin"],
                description="Contains generic often needed keywords. Imported automatically and thus always available.",
                is_builtin=True
            ),
            LibraryInfo(
                name="Collections",
                package_name="robotframework", 
                installation_command="Built-in with Robot Framework",
                use_cases=["list manipulation", "dictionary operations", "data structures"],
                categories=["core", "builtin", "data"],
                description="Contains keywords for handling lists and dictionaries.",
                is_builtin=True
            ),
            LibraryInfo(
                name="DateTime",
                package_name="robotframework",
                installation_command="Built-in with Robot Framework", 
                use_cases=["date operations", "time calculations", "timestamp validation"],
                categories=["core", "builtin", "utilities"],
                description="Supports creating and verifying date and time values as well as calculations between them.",
                is_builtin=True
            ),
            LibraryInfo(
                name="Dialogs",
                package_name="robotframework",
                installation_command="Built-in with Robot Framework",
                use_cases=["user input", "pause execution", "interactive testing", "manual intervention"],
                categories=["core", "builtin", "interaction"],
                description="Supports pausing the test execution and getting input from users.",
                is_builtin=True
            ),
            LibraryInfo(
                name="OperatingSystem",
                package_name="robotframework",
                installation_command="Built-in with Robot Framework",
                use_cases=["file operations", "directory management", "environment variables", "system commands"],
                categories=["core", "builtin", "system"],
                description="Enables performing various operating system related tasks.",
                is_builtin=True
            ),
            LibraryInfo(
                name="Process",
                package_name="robotframework", 
                installation_command="Built-in with Robot Framework",
                use_cases=["execute commands", "process management", "system integration"],
                categories=["core", "builtin", "system"],
                description="Supports executing processes in the system.",
                is_builtin=True
            ),
            LibraryInfo(
                name="Screenshot",
                package_name="robotframework",
                installation_command="Built-in with Robot Framework",
                use_cases=["desktop screenshots", "visual documentation", "debugging"],
                categories=["core", "builtin", "visual"],
                description="Provides keywords to capture and store screenshots of the desktop.",
                is_builtin=True
            ),
            LibraryInfo(
                name="String",
                package_name="robotframework",
                installation_command="Built-in with Robot Framework", 
                use_cases=["text manipulation", "string validation", "pattern matching"],
                categories=["core", "builtin", "utilities"],
                description="Library for manipulating strings and verifying their contents.",
                is_builtin=True
            ),
            LibraryInfo(
                name="Telnet",
                package_name="robotframework",
                installation_command="Built-in with Robot Framework",
                use_cases=["telnet connections", "network protocols", "legacy systems"],
                categories=["core", "builtin", "network"],
                description="Supports connecting to Telnet servers and executing commands on the opened connections.",
                is_builtin=True
            ),
            LibraryInfo(
                name="XML",
                package_name="robotframework",
                installation_command="Built-in with Robot Framework",
                use_cases=["xml parsing", "xml validation", "data verification"],
                categories=["core", "builtin", "data"],
                description="Library for verifying and modifying XML documents.",
                is_builtin=True
            )
        ]
        
        # External Libraries
        external_libs = [
            LibraryInfo(
                name="SeleniumLibrary",
                package_name="robotframework-seleniumlibrary",
                installation_command="pip install robotframework-seleniumlibrary",
                use_cases=["web testing", "browser automation", "web elements", "form filling"],
                categories=["web", "testing", "automation"],
                description="Traditional web browser automation using Selenium WebDriver",
                dependencies=["selenium"]
            ),
            LibraryInfo(
                name="Browser",
                package_name="robotframework-browser",
                installation_command="pip install robotframework-browser",
                use_cases=["modern web testing", "playwright automation", "web performance", "mobile web"],
                categories=["web", "testing", "modern"],
                description="Modern web testing with Playwright backend",
                requires_setup=True,
                setup_commands=["rfbrowser init"],
                dependencies=["playwright", "node.js"]
            ),
            LibraryInfo(
                name="RequestsLibrary",
                package_name="robotframework-requests",
                installation_command="pip install robotframework-requests",
                use_cases=["api testing", "http requests", "rest api", "json validation"],
                categories=["api", "http", "testing"],
                description="HTTP API testing functionalities by wrapping Python Requests Library"
            ),
            LibraryInfo(
                name="AppiumLibrary", 
                package_name="robotframework-appiumlibrary",
                installation_command="pip install robotframework-appiumlibrary",
                use_cases=["mobile testing", "android automation", "ios testing", "app testing"],
                categories=["mobile", "testing", "automation"],
                description="Library for Android and iOS testing using Appium",
                dependencies=["appium"]
            ),
            LibraryInfo(
                name="DatabaseLibrary",
                package_name="robotframework-databaselibrary", 
                installation_command="pip install robotframework-databaselibrary",
                use_cases=["database testing", "sql queries", "data validation", "database connections"],
                categories=["database", "testing", "data"],
                description="Python based library for database testing with multiple DB support"
            ),
            LibraryInfo(
                name="SSHLibrary",
                package_name="robotframework-sshlibrary",
                installation_command="pip install robotframework-sshlibrary", 
                use_cases=["remote connections", "ssh commands", "file transfer", "server management"],
                categories=["network", "system", "remote"],
                description="Library for SSH and SFTP connections"
            ),
            LibraryInfo(
                name="DocTest.VisualTest",
                package_name="robotframework-doctestlibrary",
                installation_command="pip install robotframework-doctestlibrary",
                use_cases=["visual testing", "image comparison", "pdf validation", "document testing"],
                categories=["visual", "testing", "documents"],
                description="Simple Automated Visual Document Testing with OCR support"
            ),
            LibraryInfo(
                name="DocTest.PdfTest", 
                package_name="robotframework-doctestlibrary",
                installation_command="pip install robotframework-doctestlibrary",
                use_cases=["pdf testing", "document validation", "content verification"],
                categories=["documents", "testing", "validation"],
                description="PDF-specific testing and content validation"
            ),
            LibraryInfo(
                name="SikuliLibrary",
                package_name="robotframework-SikuliLibrary",
                installation_command="pip install robotframework-SikuliLibrary",
                use_cases=["image-based automation", "gui testing", "visual recognition"],
                categories=["visual", "gui", "automation"],
                description="GUI automation through image recognition using Sikuli",
                dependencies=["java 11+"]
            ),
            LibraryInfo(
                name="FakerLibrary",
                package_name="robotframework-faker",
                installation_command="pip install robotframework-faker",
                use_cases=["test data generation", "fake data", "random data", "data driven testing"],
                categories=["data", "utilities", "testing"],
                description="Robot Framework wrapper for faker - fake test data generator"
            ),
            LibraryInfo(
                name="RoboSAPiens",
                package_name="robotframework-robosapiens", 
                installation_command="pip install robotframework-robosapiens",
                use_cases=["sap automation", "erp testing", "enterprise applications"],
                categories=["enterprise", "sap", "automation"],
                description="Fully localized Robot Framework library for automating SAP GUI",
                platform_requirements=["windows"],
                dependencies=[".NET SDK 7.0 x64", "PowerShell 7"]
            ),
            LibraryInfo(
                name="REST",
                package_name="robotframework-requests", 
                installation_command="pip install robotframework-requests", 
                use_cases=["rest api testing", "http methods", "json responses", "api validation"],
                categories=["api", "rest", "testing"],
                description="REST API testing using RequestsLibrary"
            ),
            LibraryInfo(
                name="ImageHorizonLibrary",
                package_name="robotframework-imagehorizonlibrary",
                installation_command="pip install robotframework-imagehorizonlibrary",
                use_cases=["cross-platform gui automation", "image recognition", "desktop automation"],
                categories=["gui", "visual", "automation"],
                description="Cross-platform GUI automation based on image recognition"
            ),
            LibraryInfo(
                name="FlaUILibrary",
                package_name="robotframework-flaui",
                installation_command="pip install robotframework-flaui",
                use_cases=["windows automation", "desktop applications", "win32 gui"],
                categories=["windows", "gui", "automation"],
                description="Windows GUI testing library using FlaUI automation",
                platform_requirements=["windows"],
                dependencies=["pythonnet"]
            ),
            LibraryInfo(
                name="DataDriver",
                package_name="robotframework-datadriver",
                installation_command="pip install robotframework-datadriver[XLS]",
                use_cases=["data-driven testing", "excel files", "csv data", "parametrized tests"],
                categories=["data", "testing", "utilities"],
                description="Data-driven testing with CSV/Excel files support"
            )
        ]
        
        # Register all libraries
        all_libraries = builtin_libs + external_libs
        for lib in all_libraries:
            self.libraries_registry[lib.name] = lib
            
            # Build use case mapping
            for use_case in lib.use_cases:
                if use_case not in self.use_case_mapping:
                    self.use_case_mapping[use_case] = set()
                self.use_case_mapping[use_case].add(lib.name)
            
            # Build category mapping  
            for category in lib.categories:
                if category not in self.category_mapping:
                    self.category_mapping[category] = set()
                self.category_mapping[category].add(lib.name)
        
        self._initialized = True
        logger.info(f"Initialized library registry with {len(all_libraries)} libraries")

    def recommend_libraries(
        self,
        scenario: str,
        context: str = "web",
        max_recommendations: int = 5
    ) -> Dict[str, Any]:
        """
        Recommend Robot Framework libraries based on a scenario description.
        
        Args:
            scenario: Natural language description of the test scenario
            context: Testing context (web, mobile, api, desktop, etc.)
            max_recommendations: Maximum number of recommendations to return
            
        Returns:
            Dictionary containing library recommendations
        """
        try:
            self._initialize_registry()
            
            # Normalize scenario text
            normalized_scenario = self._normalize_text(scenario)
            
            # Extract keywords and use cases from scenario
            scenario_keywords = self._extract_keywords(normalized_scenario)
            
            # Find matching libraries
            matches = []
            
            # Strategy 1: Direct use case matching
            use_case_matches = self._match_by_use_cases(scenario_keywords, context)
            matches.extend(use_case_matches)
            
            # Strategy 2: Category-based matching
            category_matches = self._match_by_categories(scenario_keywords, context)
            matches.extend(category_matches)
            
            # Strategy 3: Context-based matching
            context_matches = self._match_by_context(context)
            matches.extend(context_matches)
            
            # Strategy 4: Keyword-based semantic matching
            keyword_matches = self._match_by_keywords(scenario_keywords)
            matches.extend(keyword_matches)
            
            # Remove duplicates and rank
            unique_matches = self._deduplicate_recommendations(matches)
            ranked_matches = self._rank_recommendations(unique_matches, scenario_keywords, context)
            
            # Limit results
            top_recommendations = ranked_matches[:max_recommendations]
            
            return {
                "success": True,
                "scenario": scenario,
                "context": context,
                "recommendations": [
                    {
                        "library_name": rec.library.name,
                        "package_name": rec.library.package_name,
                        "installation_command": rec.library.installation_command,
                        "confidence": rec.confidence,
                        "rationale": rec.rationale,
                        "use_cases": rec.library.use_cases,
                        "categories": rec.library.categories,
                        "description": rec.library.description,
                        "is_builtin": rec.library.is_builtin,
                        "requires_setup": rec.library.requires_setup,
                        "setup_commands": rec.library.setup_commands or [],
                        "platform_requirements": rec.library.platform_requirements or [],
                        "dependencies": rec.library.dependencies or []
                    } for rec in top_recommendations
                ],
                "installation_script": self._generate_installation_script(top_recommendations),
                "total_libraries_considered": len(self.libraries_registry),
                "matching_keywords": list(set(scenario_keywords))
            }
            
        except Exception as e:
            logger.error(f"Error generating library recommendations: {e}")
            return {
                "success": False,
                "error": str(e),
                "recommendations": []
            }

    def _normalize_text(self, text: str) -> str:
        """Normalize text for keyword extraction."""
        # Convert to lowercase and remove extra whitespace
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        
        # Remove punctuation but keep important characters
        normalized = re.sub(r'[^\w\s\-]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from scenario text."""
        # Common testing keywords and patterns
        testing_keywords = {
            'web': ['web', 'browser', 'page', 'click', 'form', 'element', 'html', 'javascript', 
                   'playwright', 'modern', 'fill', 'locator', 'wait', 'viewport', 'headless'],
            'api': ['api', 'rest', 'http', 'json', 'request', 'response', 'endpoint', 'service'],
            'mobile': ['mobile', 'app', 'android', 'ios', 'touch', 'swipe', 'device'],
            'database': ['database', 'sql', 'query', 'table', 'data', 'record'],
            'desktop': ['desktop', 'window', 'gui', 'application', 'dialog'],
            'visual': ['image', 'screenshot', 'visual', 'compare', 'pdf', 'document'],
            'system': ['file', 'directory', 'process', 'command', 'system', 'ssh', 'remote']
        }
        
        extracted_keywords = []
        words = text.split()
        
        # Extract individual words
        extracted_keywords.extend(words)
        
        # Extract domain-specific keywords
        for domain, keywords in testing_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    extracted_keywords.append(keyword)
        
        # Remove duplicates and very short words
        return list(set([kw for kw in extracted_keywords if len(kw) > 2]))

    def _match_by_use_cases(self, keywords: List[str], context: str) -> List[LibraryRecommendation]:
        """Match libraries based on their defined use cases."""
        matches = []
        
        for keyword in keywords:
            for use_case, library_names in self.use_case_mapping.items():
                if keyword in use_case or self._calculate_similarity(keyword, use_case) > 0.7:
                    for lib_name in library_names:
                        library = self.libraries_registry[lib_name]
                        confidence = 0.8 if keyword in use_case else 0.6
                        
                        matches.append(LibraryRecommendation(
                            library=library,
                            confidence=confidence,
                            matching_keywords=[keyword],
                            rationale=f"Matches use case: {use_case}"
                        ))
        
        return matches

    def _match_by_categories(self, keywords: List[str], context: str) -> List[LibraryRecommendation]:
        """Match libraries based on their categories."""
        matches = []
        
        # Context to category mapping
        context_categories = {
            'web': ['web', 'browser', 'http'],
            'mobile': ['mobile', 'app'],
            'api': ['api', 'http', 'rest'], 
            'database': ['database', 'data'],
            'desktop': ['gui', 'windows', 'visual'],
            'system': ['system', 'network']
        }
        
        relevant_categories = context_categories.get(context, [])
        all_relevant_categories = relevant_categories + keywords
        
        for category in all_relevant_categories:
            if category in self.category_mapping:
                for lib_name in self.category_mapping[category]:
                    library = self.libraries_registry[lib_name]
                    confidence = 0.7 if category in relevant_categories else 0.5
                    
                    matches.append(LibraryRecommendation(
                        library=library,
                        confidence=confidence,
                        matching_keywords=[category],
                        rationale=f"Matches category: {category}"
                    ))
        
        return matches

    def _match_by_context(self, context: str) -> List[LibraryRecommendation]:
        """Match libraries based on testing context."""
        context_libraries = {
            'web': ['Browser', 'SeleniumLibrary', 'RequestsLibrary'],  # Browser Library prioritized
            'mobile': ['AppiumLibrary'],
            'api': ['RequestsLibrary', 'REST'],
            'database': ['DatabaseLibrary'],
            'desktop': ['FlaUILibrary', 'ImageHorizonLibrary', 'SikuliLibrary'],
            'system': ['SSHLibrary', 'Process', 'OperatingSystem'],
            'visual': ['DocTest.VisualTest', 'DocTest.PdfTest', 'Screenshot'],
            'enterprise': ['RoboSAPiens'],
            'data': ['DataDriver', 'FakerLibrary', 'XML']
        }
        
        # Priority weighting for web context libraries
        web_priority = {
            'Browser': 0.95,         # Highest priority for modern web testing
            'SeleniumLibrary': 0.85, # Lower priority (legacy)
            'RequestsLibrary': 0.75  # Lowest for pure web UI testing
        }
        
        matches = []
        if context in context_libraries:
            for i, lib_name in enumerate(context_libraries[context]):
                if lib_name in self.libraries_registry:
                    library = self.libraries_registry[lib_name]
                    
                    # Apply priority-based confidence for web context
                    if context == 'web' and lib_name in web_priority:
                        confidence = web_priority[lib_name]
                        if lib_name == 'Browser':
                            rationale = "Modern Playwright-based web testing library (recommended)"
                        elif lib_name == 'SeleniumLibrary':
                            rationale = "Traditional Selenium-based web testing library"
                        else:
                            rationale = f"Primary library for {context} testing"
                    else:
                        confidence = max(0.9 - (i * 0.1), 0.7)  # Decreasing confidence by order
                        rationale = f"Primary library for {context} testing"
                    
                    matches.append(LibraryRecommendation(
                        library=library,
                        confidence=confidence,
                        matching_keywords=[context],
                        rationale=rationale
                    ))
        
        return matches

    def _match_by_keywords(self, keywords: List[str]) -> List[LibraryRecommendation]:
        """Match libraries by analyzing keywords in library descriptions."""
        matches = []
        
        for lib_name, library in self.libraries_registry.items():
            matching_score = 0
            matching_terms = []
            
            # Check description
            lib_text = (library.description + " " + " ".join(library.use_cases)).lower()
            
            for keyword in keywords:
                if keyword in lib_text:
                    matching_score += 1
                    matching_terms.append(keyword)
                else:
                    # Check for partial matches
                    for word in lib_text.split():
                        if self._calculate_similarity(keyword, word) > 0.8:
                            matching_score += 0.5
                            matching_terms.append(keyword)
                            break
            
            if matching_score > 0:
                confidence = min(matching_score / len(keywords), 1.0) * 0.6
                matches.append(LibraryRecommendation(
                    library=library,
                    confidence=confidence,
                    matching_keywords=matching_terms,
                    rationale=f"Keyword match in description ({len(matching_terms)} terms)"
                ))
        
        return matches

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings."""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def _deduplicate_recommendations(self, recommendations: List[LibraryRecommendation]) -> List[LibraryRecommendation]:
        """Remove duplicate recommendations, keeping highest confidence."""
        unique_recs = {}
        
        for rec in recommendations:
            lib_name = rec.library.name
            if lib_name not in unique_recs or rec.confidence > unique_recs[lib_name].confidence:
                # Combine matching keywords if updating
                if lib_name in unique_recs:
                    combined_keywords = list(set(unique_recs[lib_name].matching_keywords + rec.matching_keywords))
                    rec.matching_keywords = combined_keywords
                    rec.rationale = f"{unique_recs[lib_name].rationale}; {rec.rationale}"
                unique_recs[lib_name] = rec
        
        return list(unique_recs.values())

    def _rank_recommendations(
        self,
        recommendations: List[LibraryRecommendation],
        keywords: List[str],
        context: str
    ) -> List[LibraryRecommendation]:
        """Rank recommendations by relevance and confidence."""
        def rank_key(rec: LibraryRecommendation) -> tuple:
            # Primary sort by confidence
            confidence = rec.confidence
            
            # Boost for context-relevant libraries
            context_boost = 0.1 if context in rec.library.categories else 0
            
            # Boost for non-builtin libraries (they're usually more specific)
            specific_boost = 0.05 if not rec.library.is_builtin else 0
            
            # Penalty for platform-specific libraries if not explicitly needed
            platform_penalty = -0.1 if rec.library.platform_requirements else 0
            
            # Strong boost for direct technology matches (e.g., "xml" -> XML Library)
            technology_boost = 0
            lib_name_lower = rec.library.name.lower()
            for keyword in keywords:
                if keyword.lower() == lib_name_lower or keyword.lower() in lib_name_lower:
                    technology_boost = 0.3  # Strong boost for exact tech match
                    break
                # Check if keyword matches main technology (e.g. "xml" matches "XML")
                elif keyword.lower() in [uc.lower() for uc in rec.library.use_cases]:
                    if any(keyword.lower() in uc.lower().split() for uc in rec.library.use_cases):
                        technology_boost = 0.2  # Medium boost for use case match
                        break
            
            final_score = confidence + context_boost + specific_boost + platform_penalty + technology_boost
            
            return (-final_score, rec.library.name)  # Negative for descending order
        
        return sorted(recommendations, key=rank_key)

    def _generate_installation_script(self, recommendations: List[LibraryRecommendation]) -> str:
        """Generate installation script for recommended libraries."""
        script_lines = ["# Robot Framework Library Installation Script", ""]
        
        # Group by installation type
        pip_installs = []
        setup_commands = []
        notes = []
        
        for rec in recommendations:
            lib = rec.library
            if not lib.is_builtin:
                pip_installs.append(lib.installation_command)
                
                if lib.requires_setup and lib.setup_commands:
                    setup_commands.extend(lib.setup_commands)
                
                if lib.platform_requirements:
                    notes.append(f"# {lib.name}: Requires {', '.join(lib.platform_requirements)}")
                
                if lib.dependencies:
                    notes.append(f"# {lib.name}: Dependencies - {', '.join(lib.dependencies)}")
        
        if pip_installs:
            script_lines.append("# Install Robot Framework libraries")
            script_lines.extend(pip_installs)
            script_lines.append("")
        
        if setup_commands:
            script_lines.append("# Additional setup commands")
            script_lines.extend(setup_commands)
            script_lines.append("")
        
        if notes:
            script_lines.append("# Additional Notes")
            script_lines.extend(notes)
        
        return "\n".join(script_lines)