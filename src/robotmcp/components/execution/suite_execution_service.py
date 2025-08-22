"""Suite Execution Service for running Robot Framework test suites in dry run and normal modes."""

import asyncio
import logging
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import json
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

try:
    from robot import run_cli, rebot_cli
    from robot.api import ExecutionResult, TestSuite
    from robot.running import TestSuite as RunningTestSuite
    from robot.parsing import get_model
    from robot.libraries.BuiltIn import BuiltIn
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False
    logger.warning("Robot Framework not available - suite execution will be limited")

from robotmcp.models.config_models import ExecutionConfig

logger = logging.getLogger(__name__)


class SuiteExecutionService:
    """Service for executing Robot Framework test suites in dry run and normal modes."""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.temp_dir = self._setup_temp_directory()
        self.active_executions = {}  # Track active executions for cleanup
        
        # Execution timeouts
        self.dry_run_timeout = getattr(config, 'DRY_RUN_TIMEOUT', 30)
        self.execution_timeout = getattr(config, 'EXECUTION_TIMEOUT', 300)
        
        logger.info(f"SuiteExecutionService initialized with temp dir: {self.temp_dir}")
    
    def _setup_temp_directory(self) -> str:
        """Setup temporary directory for suite execution."""
        base_temp = getattr(self.config, 'TEMP_DIR_BASE', None)
        if base_temp:
            temp_dir = os.path.join(base_temp, f"rf_mcp_execution_{uuid.uuid4().hex[:8]}")
            os.makedirs(temp_dir, exist_ok=True)
            return temp_dir
        else:
            return tempfile.mkdtemp(prefix="rf_mcp_execution_")
    
    async def execute_dry_run(
        self, 
        suite_content: str, 
        session_id: str,
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute suite in dry run mode and parse validation results.
        
        Args:
            suite_content: Robot Framework suite content (.robot format)
            session_id: Session identifier for tracking
            options: Execution options including validation_level, include_warnings
            
        Returns:
            Structured validation results
        """
        if options is None:
            options = {}
        
        execution_id = f"dry_{session_id}_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create temporary suite file
            suite_file = await self._create_temp_suite_file(suite_content, execution_id)
            
            # Execute Robot Framework dry run
            return_code, stdout, stderr = await self._execute_rf_dry_run(suite_file, options)
            
            # Parse results
            validation_results = self._parse_dry_run_output(stdout, stderr, return_code, options)
            
            # Add execution metadata
            validation_results.update({
                "execution_id": execution_id,
                "session_id": session_id,
                "tool": "run_test_suite_dry",
                "execution_time": validation_results.get("execution_time", 0),
                "suite_file_generated": os.path.basename(suite_file)
            })
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in dry run execution for session {session_id}: {e}")
            return {
                "success": False,
                "tool": "run_test_suite_dry",
                "session_id": session_id,
                "error": f"Dry run execution failed: {str(e)}",
                "error_type": "execution_error"
            }
        finally:
            await self._cleanup_execution(execution_id)
    
    async def execute_normal(
        self,
        suite_content: str,
        session_id: str,
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute suite normally and parse execution results.
        
        Args:
            suite_content: Robot Framework suite content (.robot format)
            session_id: Session identifier for tracking
            options: Execution options including variables, tags, output settings
            
        Returns:
            Comprehensive execution results
        """
        if options is None:
            options = {}
        
        execution_id = f"normal_{session_id}_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create temporary suite file
            suite_file = await self._create_temp_suite_file(suite_content, execution_id)
            
            # Execute Robot Framework normally
            return_code, stdout, stderr, output_dir = await self._execute_rf_normal(suite_file, options)
            
            # Parse execution results
            execution_results = await self._parse_execution_results(output_dir, return_code, stdout, stderr, options)
            
            # Add execution metadata
            execution_results.update({
                "execution_id": execution_id,
                "session_id": session_id,
                "tool": "run_test_suite",
                "suite_file_generated": os.path.basename(suite_file)
            })
            
            return execution_results
            
        except Exception as e:
            logger.error(f"Error in normal execution for session {session_id}: {e}")
            return {
                "success": False,
                "tool": "run_test_suite",
                "session_id": session_id,
                "error": f"Normal execution failed: {str(e)}",
                "error_type": "execution_error"
            }
        finally:
            # Cleanup handled by configuration (may preserve output files)
            if not getattr(self.config, 'PRESERVE_OUTPUT_FILES', False):
                await self._cleanup_execution(execution_id)
    
    async def _create_temp_suite_file(self, suite_content: str, execution_id: str) -> str:
        """Create temporary .robot file for execution."""
        suite_file = os.path.join(self.temp_dir, f"suite_{execution_id}.robot")
        
        try:
            with open(suite_file, 'w', encoding='utf-8') as f:
                f.write(suite_content)
            
            logger.debug(f"Created temporary suite file: {suite_file}")
            
            # Track for cleanup
            if execution_id not in self.active_executions:
                self.active_executions[execution_id] = {"files": [], "dirs": []}
            self.active_executions[execution_id]["files"].append(suite_file)
            
            return suite_file
            
        except Exception as e:
            logger.error(f"Failed to create temporary suite file: {e}")
            raise
    
    async def _execute_rf_dry_run(self, suite_file: str, options: Dict) -> Tuple[int, str, str]:
        """Execute Robot Framework dry run using native API."""
        if not ROBOT_AVAILABLE:
            raise Exception("Robot Framework not available for dry run execution")
        
        try:
            start_time = datetime.now()
            
            # Prepare Robot Framework options
            rf_options = ["--dryrun", "--output", "NONE", "--report", "NONE", "--log", "NONE"]
            
            # Add validation level options
            validation_level = options.get("validation_level", "standard")
            if validation_level == "strict":
                rf_options.extend(["--loglevel", "DEBUG"])
            elif validation_level == "minimal":
                rf_options.extend(["--loglevel", "WARN"])
            else:
                rf_options.extend(["--loglevel", "INFO"])
            
            # Add custom options if provided
            if options.get("include_tags"):
                for tag in options["include_tags"]:
                    rf_options.extend(["--include", tag])
            
            if options.get("exclude_tags"):
                for tag in options["exclude_tags"]:
                    rf_options.extend(["--exclude", tag])
            
            # Set output directory
            rf_options.extend(["--outputdir", self.temp_dir])
            
            # Add the suite file
            rf_options.append(suite_file)
            
            logger.debug(f"Executing dry run with options: {rf_options}")
            
            # Capture stdout and stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            # Execute Robot Framework dry run in executor to avoid blocking
            def run_robot_dry():
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    try:
                        # Use run_cli which properly handles command line arguments
                        return run_cli(rf_options, exit=False)
                    except SystemExit as e:
                        return e.code if hasattr(e, 'code') else (e.args[0] if e.args else 1)
                    except Exception as e:
                        logger.error(f"Robot Framework dry run error: {e}")
                        return 252  # Invalid test data
            
            # Run in thread to avoid blocking
            loop = asyncio.get_event_loop()
            return_code = await asyncio.wait_for(
                loop.run_in_executor(None, run_robot_dry),
                timeout=self.dry_run_timeout
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            stdout_content = stdout_capture.getvalue()
            stderr_content = stderr_capture.getvalue()
            
            logger.debug(f"Dry run completed in {execution_time:.2f}s with return code {return_code}")
            
            return return_code, stdout_content, stderr_content
            
        except asyncio.TimeoutError:
            logger.error(f"Dry run execution timed out after {self.dry_run_timeout}s")
            raise Exception(f"Dry run execution timed out after {self.dry_run_timeout}s")
        
        except Exception as e:
            logger.error(f"Error executing dry run: {e}")
            raise
    
    async def _execute_rf_normal(self, suite_file: str, options: Dict) -> Tuple[int, str, str, str]:
        """Execute Robot Framework normal run using native API."""
        if not ROBOT_AVAILABLE:
            raise Exception("Robot Framework not available for normal execution")
        
        execution_id = os.path.basename(suite_file).replace("suite_", "").replace(".robot", "")
        output_dir = os.path.join(self.temp_dir, f"output_{execution_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Track output directory for cleanup
        if execution_id in self.active_executions:
            self.active_executions[execution_id]["dirs"].append(output_dir)
        
        try:
            start_time = datetime.now()
            
            # Prepare Robot Framework options
            rf_options = [
                "--outputdir", output_dir,
                "--output", "output.xml",
                "--report", "report.html",
                "--log", "log.html"
            ]
            
            # Add execution options
            if options.get("variables"):
                for key, value in options["variables"].items():
                    rf_options.extend(["--variable", f"{key}:{value}"])
            
            if options.get("include_tags"):
                for tag in options["include_tags"]:
                    rf_options.extend(["--include", tag])
            
            if options.get("exclude_tags"):
                for tag in options["exclude_tags"]:
                    rf_options.extend(["--exclude", tag])
            
            if options.get("loglevel"):
                rf_options.extend(["--loglevel", options["loglevel"]])
            else:
                rf_options.extend(["--loglevel", "INFO"])
            
            # Add screenshot capture if requested
            if options.get("capture_screenshots", False):
                rf_options.extend(["--variable", "CAPTURE_SCREENSHOTS:True"])
            
            # Add the suite file
            rf_options.append(suite_file)
            
            logger.debug(f"Executing normal run with options: {rf_options}")
            
            # Capture stdout and stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            # Execute Robot Framework in executor to avoid blocking
            def run_robot_normal():
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    try:
                        # Use run_cli which properly handles command line arguments
                        return run_cli(rf_options, exit=False)
                    except SystemExit as e:
                        return e.code if hasattr(e, 'code') else (e.args[0] if e.args else 1)
                    except Exception as e:
                        logger.error(f"Robot Framework execution error: {e}")
                        return 255  # Unexpected internal error
            
            # Run in thread to avoid blocking
            loop = asyncio.get_event_loop()
            return_code = await asyncio.wait_for(
                loop.run_in_executor(None, run_robot_normal),
                timeout=self.execution_timeout
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            stdout_content = stdout_capture.getvalue()
            stderr_content = stderr_capture.getvalue()
            
            logger.debug(f"Normal execution completed in {execution_time:.2f}s with return code {return_code}")
            
            return return_code, stdout_content, stderr_content, output_dir
            
        except asyncio.TimeoutError:
            logger.error(f"Normal execution timed out after {self.execution_timeout}s")
            raise Exception(f"Normal execution timed out after {self.execution_timeout}s")
        
        except Exception as e:
            logger.error(f"Error executing normal run: {e}")
            raise
    
    def _validate_suite_syntax(self, suite_content: str) -> Dict[str, Any]:
        """Validate suite syntax using Robot Framework parsing API."""
        if not ROBOT_AVAILABLE:
            return {"syntax_valid": False, "error": "Robot Framework not available"}
        
        try:
            # Create temporary file for parsing
            temp_file = os.path.join(self.temp_dir, f"syntax_check_{uuid.uuid4().hex[:8]}.robot")
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(suite_content)
            
            # Try to parse the suite
            try:
                model = get_model(temp_file)
                
                # Basic syntax validation passed
                validation_result = {
                    "syntax_valid": True,
                    "suite_name": getattr(model, 'name', 'Unknown'),
                    "sections": []
                }
                
                # Extract information about sections
                if hasattr(model, 'sections'):
                    for section in model.sections:
                        section_info = {
                            "type": section.__class__.__name__,
                            "header": getattr(section, 'header', None)
                        }
                        validation_result["sections"].append(section_info)
                
                return validation_result
                
            except Exception as parse_error:
                return {
                    "syntax_valid": False,
                    "error": f"Syntax error: {str(parse_error)}",
                    "error_type": "parsing_error"
                }
            
            finally:
                # Clean up temp file
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception:
                    pass
                    
        except Exception as e:
            return {
                "syntax_valid": False,
                "error": f"Validation error: {str(e)}",
                "error_type": "validation_error"
            }
    
    def _parse_dry_run_output(self, stdout: str, stderr: str, return_code: int, options: Dict) -> Dict[str, Any]:
        """Parse dry run output for validation issues."""
        issues = []
        warnings = []
        suggestions = []
        
        # Parse stderr for common error patterns
        error_patterns = [
            (r"No keyword with name '(.+)' found", "missing_keyword", "error"),
            (r"Importing library '(.+)' failed", "import_error", "error"),
            (r"Importing resource '(.+)' failed", "resource_error", "error"),
            (r"Invalid number of arguments", "argument_error", "error"),
            (r"Keyword name cannot be empty", "syntax_error", "error"),
            (r"Variable '(.+)' not found", "variable_error", "error"),
            (r"Invalid variable syntax", "syntax_error", "error")
        ]
        
        # Parse warning patterns
        warning_patterns = [
            (r"Variable '(.+)' is not used", "unused_variable", "warning"),
            (r"Keyword '(.+)' is deprecated", "deprecated_keyword", "warning")
        ]
        
        all_output = stdout + "\n" + stderr
        
        for pattern, issue_type, severity in error_patterns:
            matches = re.findall(pattern, all_output, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                issue = {
                    "type": issue_type,
                    "message": self._format_error_message(issue_type, match),
                    "severity": severity,
                    "context": match if isinstance(match, str) else str(match)
                }
                issues.append(issue)
                
                # Add suggestions based on error type
                suggestion = self._generate_suggestion(issue_type, match)
                if suggestion:
                    suggestions.append(suggestion)
        
        # Parse warnings if requested
        if options.get("include_warnings", True):
            for pattern, issue_type, severity in warning_patterns:
                matches = re.findall(pattern, all_output, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    warning = {
                        "type": issue_type,
                        "message": self._format_error_message(issue_type, match),
                        "severity": severity,
                        "context": match if isinstance(match, str) else str(match)
                    }
                    warnings.append(warning)
        
        # Determine validation status
        if return_code == 0:
            validation_status = "passed"
        elif issues:
            validation_status = "failed"
        else:
            validation_status = "warning" if warnings else "passed"
        
        # Parse suite information from output
        suite_info = self._extract_suite_info(stdout, stderr)
        
        result = {
            "success": return_code == 0,
            "validation_status": validation_status,
            "suite_info": suite_info,
            "validation_results": {
                "syntax_valid": return_code == 0 and not any(i["type"] == "syntax_error" for i in issues),
                "imports_valid": not any(i["type"] in ["import_error", "resource_error"] for i in issues),
                "keywords_found": not any(i["type"] == "missing_keyword" for i in issues),
                "issues": issues,
                "warnings": warnings,
                "suggestions": suggestions
            },
            "return_code": return_code
        }
        
        # Include raw output if configured
        if getattr(self.config, 'INCLUDE_RAW_OUTPUT', False):
            result["raw_output"] = {"stdout": stdout, "stderr": stderr}
        
        return result
    
    async def _parse_execution_results(self, output_dir: str, return_code: int, stdout: str, stderr: str, options: Dict) -> Dict[str, Any]:
        """Parse Robot Framework execution results using native Result objects."""
        output_xml = os.path.join(output_dir, "output.xml")
        
        if not os.path.exists(output_xml):
            return {
                "success": False,
                "error": "Output XML not generated",
                "execution_status": "failed",
                "return_code": return_code,
                "raw_output": {"stdout": stdout, "stderr": stderr} if options.get("include_raw_output") else None
            }
        
        try:
            # Use Robot Framework's native result parsing with Result object
            if ROBOT_AVAILABLE:
                try:
                    from robot.result.executionresult import Result
                    from robot.api import ExecutionResult
                    
                    # Load execution result 
                    exec_result = ExecutionResult(output_xml)
                    
                    # Get the Result object which contains detailed statistics and messages
                    result_obj = exec_result.result if hasattr(exec_result, 'result') else exec_result
                    
                    # Extract comprehensive statistics using Result object
                    suite = exec_result.suite
                    stats = suite.statistics
                    
                    statistics = {
                        "total": stats.total,
                        "passed": stats.passed,
                        "failed": stats.failed,
                        "skipped": getattr(stats, 'skipped', 0),
                        "success_rate": stats.passed / max(stats.total, 1) if stats.total > 0 else 0.0
                    }
                    
                    # Extract detailed test information with messages
                    test_details = []
                    failed_tests = []
                    
                    def extract_test_info(suite_obj):
                        """Recursively extract test information from suites."""
                        for test in suite_obj.tests:
                            test_info = {
                                "name": test.name,
                                "status": test.status,
                                "message": test.message,
                                "start_time": test.start_time.isoformat() if test.start_time else None,
                                "end_time": test.end_time.isoformat() if test.end_time else None,
                                "elapsed_time": test.elapsed_time.total_seconds() if test.elapsed_time else 0,
                                "tags": list(test.tags) if hasattr(test, 'tags') else [],
                                "critical": getattr(test, 'critical', True)
                            }
                            test_details.append(test_info)
                            
                            # Add to failed tests if not passed
                            if test.status != "PASS":
                                failed_info = test_info.copy()
                                
                                # Extract failing keyword information
                                failing_keyword = None
                                if hasattr(test, 'body'):
                                    for item in test.body:
                                        if hasattr(item, 'status') and item.status == 'FAIL':
                                            failing_keyword = {
                                                "name": getattr(item, 'name', 'Unknown'),
                                                "message": getattr(item, 'message', ''),
                                                "elapsed_time": item.elapsed_time.total_seconds() if hasattr(item, 'elapsed_time') and item.elapsed_time else 0
                                            }
                                            break
                                
                                if failing_keyword:
                                    failed_info["failing_keyword"] = failing_keyword
                                
                                failed_tests.append(failed_info)
                        
                        # Recursively process nested suites
                        for nested_suite in suite_obj.suites:
                            extract_test_info(nested_suite)
                    
                    extract_test_info(suite)
                    
                    # Build comprehensive execution result
                    execution_result = {
                        "success": return_code == 0,
                        "execution_status": "passed" if return_code == 0 else "failed", 
                        "statistics": statistics,
                        "execution_details": {
                            "start_time": suite.start_time.isoformat() if suite.start_time else None,
                            "end_time": suite.end_time.isoformat() if suite.end_time else None,
                            "elapsed_time": suite.elapsed_time.total_seconds() if suite.elapsed_time else 0,
                            "return_code": return_code,
                            "robot_version": getattr(exec_result, 'generated', 'Unknown')
                        },
                        "suite_info": {
                            "name": suite.name,
                            "full_name": suite.full_name if hasattr(suite, 'full_name') else suite.name,
                            "documentation": suite.doc,
                            "test_count": stats.total,
                            "metadata": dict(suite.metadata) if hasattr(suite, 'metadata') else {},
                            "source": getattr(suite, 'source', 'Unknown')
                        },
                        "output_files": {
                            "output_xml": output_xml,
                            "log_html": os.path.join(output_dir, "log.html"),
                            "report_html": os.path.join(output_dir, "report.html")
                        }
                    }
                    
                    # Add test details based on output level
                    output_level = options.get("output_level", "standard")
                    if output_level == "detailed":
                        execution_result["test_details"] = test_details
                    
                    # Add failed tests if any
                    if failed_tests:
                        execution_result["failed_tests"] = failed_tests[:10]  # Limit to first 10 failures
                        if len(failed_tests) > 10:
                            execution_result["additional_failures"] = len(failed_tests) - 10
                    
                    # Add suite messages and errors
                    if hasattr(suite, 'message') and suite.message:
                        execution_result["suite_message"] = suite.message
                    
                    return execution_result
                    
                except ImportError as e:
                    logger.warning(f"Robot Framework Result parsing not fully available: {e}")
                    return await self._parse_execution_results_fallback(output_xml, return_code, stdout, stderr, output_dir)
            else:
                return await self._parse_execution_results_fallback(output_xml, return_code, stdout, stderr, output_dir)
                
        except Exception as e:
            logger.error(f"Failed to parse execution results: {e}")
            return {
                "success": False,
                "error": f"Failed to parse execution results: {str(e)}",
                "execution_status": "failed",
                "return_code": return_code,
                "error_type": "result_parsing_error"
            }
    
    async def _parse_execution_results_fallback(self, output_xml: str, return_code: int, stdout: str, stderr: str, output_dir: str) -> Dict[str, Any]:
        """Fallback method to parse execution results without Robot Framework API."""
        try:
            # Basic XML parsing to extract statistics
            import xml.etree.ElementTree as ET
            tree = ET.parse(output_xml)
            root = tree.getroot()
            
            # Extract basic statistics from XML
            statistics_elem = root.find(".//total/stat")
            if statistics_elem is not None:
                total = int(statistics_elem.get("pass", 0)) + int(statistics_elem.get("fail", 0))
                passed = int(statistics_elem.get("pass", 0))
                failed = int(statistics_elem.get("fail", 0))
            else:
                total = passed = failed = 0
            
            # Extract suite information
            suite_elem = root.find("suite")
            suite_name = suite_elem.get("name", "Unknown") if suite_elem is not None else "Unknown"
            
            return {
                "success": return_code == 0,
                "execution_status": "passed" if return_code == 0 else "failed",
                "statistics": {
                    "total": total,
                    "passed": passed,
                    "failed": failed,
                    "skipped": 0,
                    "success_rate": passed / max(total, 1)
                },
                "suite_info": {
                    "name": suite_name,
                    "test_count": total
                },
                "output_files": {
                    "output_xml": output_xml,
                    "log_html": os.path.join(output_dir, "log.html"),
                    "report_html": os.path.join(output_dir, "report.html")
                },
                "execution_details": {
                    "return_code": return_code
                },
                "note": "Results parsed using fallback method (limited information available)"
            }
            
        except Exception as e:
            logger.error(f"Fallback parsing also failed: {e}")
            return {
                "success": False,
                "error": f"Both primary and fallback result parsing failed: {str(e)}",
                "execution_status": "failed",
                "return_code": return_code
            }
    
    def _extract_failed_tests(self, suite) -> List[Dict[str, Any]]:
        """Extract information about failed tests from suite results."""
        failed_tests = []
        
        def extract_from_suite(s):
            for test in s.tests:
                if not test.passed:
                    failed_info = {
                        "name": test.name,
                        "status": test.status,
                        "message": test.message,
                        "elapsed_time": test.elapsed_time.total_seconds()
                    }
                    
                    # Try to find the failing keyword
                    for kw in test.body:
                        if hasattr(kw, 'status') and kw.status == 'FAIL':
                            failed_info.update({
                                "failing_keyword": kw.name,
                                "keyword_message": getattr(kw, 'message', '')
                            })
                            break
                    
                    failed_tests.append(failed_info)
            
            # Recursively check nested suites
            for nested_suite in s.suites:
                extract_from_suite(nested_suite)
        
        extract_from_suite(suite)
        return failed_tests
    
    def _extract_suite_info(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Extract suite information from Robot Framework output."""
        suite_info = {
            "name": "Unknown",
            "test_count": 0,
            "keyword_count": 0,
            "libraries_used": []
        }
        
        # Try to extract suite name
        name_match = re.search(r"^(.+?) :: ", stdout, re.MULTILINE)
        if name_match:
            suite_info["name"] = name_match.group(1).strip()
        
        # Try to extract test count from dry run output
        test_match = re.search(r"(\d+) test[s]?, (\d+) passed, (\d+) failed", stdout)
        if test_match:
            suite_info["test_count"] = int(test_match.group(1))
        
        return suite_info
    
    def _format_error_message(self, error_type: str, context: str) -> str:
        """Format error messages based on error type."""
        messages = {
            "missing_keyword": f"Keyword '{context}' not found in available libraries",
            "import_error": f"Failed to import library '{context}'",
            "resource_error": f"Failed to import resource '{context}'",
            "argument_error": f"Invalid arguments provided to keyword",
            "syntax_error": f"Invalid Robot Framework syntax",
            "variable_error": f"Variable '{context}' is not defined",
            "unused_variable": f"Variable '{context}' is defined but not used",
            "deprecated_keyword": f"Keyword '{context}' is deprecated"
        }
        return messages.get(error_type, f"Unknown error type: {error_type}")
    
    def _generate_suggestion(self, error_type: str, context: str) -> Optional[str]:
        """Generate suggestions based on error type."""
        suggestions = {
            "missing_keyword": f"Check if the library containing '{context}' is imported or if keyword name is spelled correctly",
            "import_error": f"Verify that library '{context}' is installed and available",
            "resource_error": f"Check that resource file '{context}' exists and is accessible",
            "argument_error": "Review keyword documentation for correct argument syntax",
            "variable_error": f"Define variable '{context}' in Variables section or pass as parameter"
        }
        return suggestions.get(error_type)
    
    async def _cleanup_execution(self, execution_id: str):
        """Clean up temporary files and directories for an execution."""
        if execution_id not in self.active_executions:
            return
        
        try:
            execution_data = self.active_executions[execution_id]
            
            # Clean up files
            for file_path in execution_data.get("files", []):
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.debug(f"Cleaned up file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up file {file_path}: {e}")
            
            # Clean up directories (only if not preserving output files)
            if not getattr(self.config, 'PRESERVE_OUTPUT_FILES', False):
                for dir_path in execution_data.get("dirs", []):
                    try:
                        if os.path.exists(dir_path):
                            import shutil
                            shutil.rmtree(dir_path)
                            logger.debug(f"Cleaned up directory: {dir_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up directory {dir_path}: {e}")
            
            # Remove from tracking
            del self.active_executions[execution_id]
            
        except Exception as e:
            logger.error(f"Error during cleanup for execution {execution_id}: {e}")
    
    def cleanup_all(self):
        """Clean up all temporary files and directories."""
        logger.info("Cleaning up all suite execution temporary files")
        
        # Clean up tracked executions
        for execution_id in list(self.active_executions.keys()):
            asyncio.create_task(self._cleanup_execution(execution_id))
        
        # Clean up temp directory
        try:
            if os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.debug(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Failed to clean up temp directory {self.temp_dir}: {e}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current status of the suite execution service."""
        return {
            "service": "SuiteExecutionService",
            "temp_directory": self.temp_dir,
            "active_executions": len(self.active_executions),
            "execution_timeouts": {
                "dry_run": self.dry_run_timeout,
                "normal": self.execution_timeout
            },
            "configuration": {
                "preserve_output_files": getattr(self.config, 'PRESERVE_OUTPUT_FILES', False),
                "include_raw_output": getattr(self.config, 'INCLUDE_RAW_OUTPUT', False)
            }
        }