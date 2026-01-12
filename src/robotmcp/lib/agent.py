"""AI Agent setup using pydantic-ai.

Provides the AI agent that translates natural language prompts
into Robot Framework keyword calls.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from robot.api import logger as rf_logger

from robotmcp.lib.providers import ProviderConfig, get_model_instance
from robotmcp.lib.retry import ExecutionResult, RetryContext, build_retry_prompt

logger = logging.getLogger(__name__)

# System prompts for different keyword types
DO_SYSTEM_PROMPT = """You are a Robot Framework automation expert.
Given a natural language prompt describing an ACTION to perform, generate the appropriate
Robot Framework keyword call using the available keywords and current page context.

=== ABSOLUTELY CRITICAL - READ THIS FIRST ===
You MUST use ONLY selectors that appear EXACTLY in the "Page elements:" section below.
NEVER invent, guess, or modify selectors. If the context shows:
  - link: [data-test='shopping-cart-link']
Then you MUST use: [data-test='shopping-cart-link']
NOT: [data-test='shopping-cart'] (WRONG - different value!)
NOT: #shopping-cart (WRONG - different format!)

RULES:
1. For web automation, use Browser Library keywords (Fill Text, Click, Get Text)
2. COPY-PASTE selectors EXACTLY from the Page elements section - character for character
3. Separate keyword name and arguments with 4 spaces (not tabs)
4. Click keyword takes only ONE argument: the selector

SELECTOR LOOKUP PROCESS:
1. Read the "Page elements:" section in the Context
2. Find the element that matches what you need to interact with
3. COPY the exact selector shown (e.g., [data-test='shopping-cart-link'])
4. Use that exact selector in your keyword call

Respond with ONLY the keyword call in the format:
KEYWORD    arg1    arg2    ...

Examples (when context provides these exact selectors):
- Context shows: "link: [data-test='shopping-cart-link']"
  Action: Click shopping cart -> Click    [data-test='shopping-cart-link']

- Context shows: "button 'Checkout': [data-test='checkout']"
  Action: Click checkout -> Click    [data-test='checkout']

- Context shows: "input[type=text]: [data-test='firstName']"
  Action: Fill first name -> Fill Text    [data-test='firstName']    John
"""

CHECK_SYSTEM_PROMPT = """You are a Robot Framework automation expert.
Given a natural language prompt describing a VERIFICATION/ASSERTION, generate the appropriate
Robot Framework keyword call to verify the condition.

=== ABSOLUTELY CRITICAL ===
You MUST use ONLY selectors that appear EXACTLY in the "Page elements:" section.
COPY-PASTE selectors exactly as shown - character for character.

RULES:
1. Use Browser Library assertion keywords
2. For text checks, PREFER contains (*=) over exact match (==) - handles case differences
3. For URL checks, use Get Url with *=
4. For visibility, use Get Element States    selector    contains    visible

TEXT ASSERTION OPERATORS (use *=  by default):
- *=  contains (case-insensitive match) - PREFERRED
- ==  exact match (rarely needed)
- ^=  starts with
- $=  ends with

Respond with ONLY the keyword call in the format:
KEYWORD    arg1    arg2    ...

Examples (when context provides these exact selectors):
- Context shows: "text 'Thank you': [data-test='complete-header']"
  Check: Order confirmed -> Get Text    [data-test='complete-header']    *=    thank you

- Check: Cart shows 3 items -> Get Text    [data-test='cart-badge']    *=    3
- Check: Login was successful -> Get Url    *=    inventory
- Check: Error is visible -> Get Element States    [data-test='error']    contains    visible
"""

ASK_SYSTEM_PROMPT = """You are a Robot Framework automation expert.
Given a natural language prompt asking for INFORMATION, generate the appropriate
Robot Framework keyword call to extract and return the requested data.

=== ABSOLUTELY CRITICAL ===
You MUST use ONLY selectors that appear EXACTLY in the "Page elements:" section.
COPY-PASTE selectors exactly as shown - character for character.

=== IMPORTANT: Get Title vs Get Text ===
- Get Title returns the BROWSER TAB TITLE (e.g., "Swag Labs"), NOT element content
- Get Text returns the TEXT CONTENT of an element on the page
- NEVER use Get Title for questions about products, items, prices, or page content
- ONLY use Get Title when explicitly asked "What is the page/browser title?"

=== SELECTOR PRIORITY ORDER ===
1. FIRST: Look in "Page elements:" for exact matching content or data-test attributes
2. SECOND: Use class-based selectors from context for common patterns:
   - Product names: .inventory_item_name, .product-name, .item-name
   - Product prices: .inventory_item_price, .product-price, .price
   - Product descriptions: .inventory_item_desc, .product-description
3. THIRD: Use nth selector for "first", "second", etc. (0-indexed):
   - "first" -> >> nth=0
   - "second" -> >> nth=1
   - "third" -> >> nth=2
4. LAST RESORT: Get Title (ONLY for actual browser tab title) or Get Url (for current URL)

RULES:
1. Use Get Text to extract text from elements - this is your PRIMARY keyword
2. If multiple elements have same class (like products), append " >> nth=N" to get specific one
   IMPORTANT: Write it as ONE selector string: ".class >> nth=0" (no extra spaces)
3. For ordinal questions (first, second, nth), ALWAYS use the nth selector with Get Text

Respond with ONLY the keyword call in the format:
KEYWORD    arg1

Examples (when context provides these exact selectors):
- Context shows: ".inventory_item_name" (class for product names)
  Question: What is the name of the first product?
  Answer: Get Text    .inventory_item_name >> nth=0

- Context shows: ".inventory_item_price" (class for prices)
  Question: What is the price of the second product?
  Answer: Get Text    .inventory_item_price >> nth=1

- Context shows: "[data-test='inventory-item-name']"
  Question: What product name is shown?
  Answer: Get Text    [data-test='inventory-item-name'] >> nth=0

- Context shows: "text 'Thank you': [data-test='complete-header']"
  Question: What is the confirmation message?
  Answer: Get Text    [data-test='complete-header']

- Question: What is the current URL?
  Answer: Get Url

- Question: What is the browser/page title? (ONLY for this specific question)
  Answer: Get Title
"""


class RFAgent:
    """AI agent for translating natural language to Robot Framework keywords."""

    def __init__(self, config: ProviderConfig):
        """Initialize the RF Agent.

        Args:
            config: ProviderConfig with AI provider settings
        """
        self.config = config
        self._agent = None
        self._model = None

    def _ensure_agent(self, system_prompt: str):
        """Ensure the pydantic-ai agent is initialized.

        Args:
            system_prompt: System prompt for the agent
        """
        try:
            from pydantic_ai import Agent
        except ImportError:
            raise ImportError(
                "pydantic-ai is required for AILibrary. "
                "Install with: pip install rf-mcp[lib]"
            )

        if self._model is None:
            self._model = get_model_instance(self.config)

        self._agent = Agent(
            model=self._model,
            system_prompt=system_prompt,
        )

    async def generate_keyword_for_do(
        self,
        prompt: str,
        context: Dict[str, Any] = None,
        retry_context: RetryContext = None,
    ) -> str:
        """Generate keyword call for a Do action.

        Args:
            prompt: Natural language prompt describing the action
            context: Optional context (page state, variables, etc.)
            retry_context: Optional retry context with error information

        Returns:
            Generated keyword call string
        """
        return await self._generate_keyword(
            prompt=prompt,
            system_prompt=DO_SYSTEM_PROMPT,
            context=context,
            retry_context=retry_context,
        )

    async def generate_keyword_for_check(
        self,
        prompt: str,
        context: Dict[str, Any] = None,
        retry_context: RetryContext = None,
    ) -> str:
        """Generate keyword call for a Check assertion.

        Args:
            prompt: Natural language prompt describing the verification
            context: Optional context (page state, variables, etc.)
            retry_context: Optional retry context with error information

        Returns:
            Generated keyword call string
        """
        return await self._generate_keyword(
            prompt=prompt,
            system_prompt=CHECK_SYSTEM_PROMPT,
            context=context,
            retry_context=retry_context,
        )

    async def generate_keyword_for_ask(
        self,
        prompt: str,
        context: Dict[str, Any] = None,
        retry_context: RetryContext = None,
    ) -> str:
        """Generate keyword call for an Ask query.

        Args:
            prompt: Natural language prompt describing the query
            context: Optional context (page state, variables, etc.)
            retry_context: Optional retry context with error information

        Returns:
            Generated keyword call string
        """
        return await self._generate_keyword(
            prompt=prompt,
            system_prompt=ASK_SYSTEM_PROMPT,
            context=context,
            retry_context=retry_context,
        )

    async def _generate_keyword(
        self,
        prompt: str,
        system_prompt: str,
        context: Dict[str, Any] = None,
        retry_context: RetryContext = None,
    ) -> str:
        """Generate a keyword call using the AI agent.

        Args:
            prompt: Natural language prompt
            system_prompt: System prompt for the agent
            context: Optional context information
            retry_context: Optional retry context with error information

        Returns:
            Generated keyword call string
        """
        self._ensure_agent(system_prompt)

        # Build the user message
        if retry_context and retry_context.attempt_number > 0:
            # This is a retry - include error context
            user_message = build_retry_prompt(retry_context)
        else:
            # Initial request
            user_message = prompt

            # Add context if available
            if context:
                context_parts = []
                if "url" in context:
                    context_parts.append(f"Current URL: {context['url']}")
                if "title" in context:
                    context_parts.append(f"Page title: {context['title']}")
                # Include page elements for selector discovery
                if "elements" in context and context["elements"]:
                    context_parts.append(f"\n{context['elements']}")
                if "available_keywords" in context:
                    kws = context["available_keywords"][:20]
                    context_parts.append(f"Available keywords: {', '.join(kws)}")
                # Include execution history for continuity
                if "execution_history" in context and context["execution_history"]:
                    context_parts.append(f"\n{context['execution_history']}")

                if context_parts:
                    user_message = f"{prompt}\n\nContext:\n" + "\n".join(context_parts)

        # Log what we're sending to the AI
        rf_logger.debug(f"AI prompt: {user_message[:500]}...")
        if context and "elements" in context:
            rf_logger.debug(f"Page elements: {context.get('elements', '')[:300]}...")

        try:
            # Run the agent
            result = await self._agent.run(user_message)

            # Extract the response - pydantic-ai uses .output attribute
            if hasattr(result, "output"):
                response_text = result.output
            elif hasattr(result, "data"):
                response_text = result.data
            else:
                response_text = str(result)

            # Ensure we have a string
            if not isinstance(response_text, str):
                response_text = str(response_text)

            # Clean up the response
            keyword_call = self._clean_response(response_text)

            logger.info(f"Generated keyword: {keyword_call}")
            return keyword_call

        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            raise

    def _clean_response(self, response: str) -> str:
        """Clean up the AI response to extract the keyword call.

        Args:
            response: Raw AI response

        Returns:
            Cleaned keyword call string
        """
        # Remove any markdown code blocks
        if "```" in response:
            # Extract content between code blocks
            parts = response.split("```")
            if len(parts) >= 2:
                # Take the first code block content
                response = parts[1]
                # Remove language identifier if present
                if response.startswith("robotframework"):
                    response = response[14:]
                elif response.startswith("robot"):
                    response = response[5:]

        # Strip whitespace
        response = response.strip()

        # If response has multiple lines, take the first non-empty line
        # (unless it's a multi-keyword action)
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        if lines:
            # Check if this looks like multiple keyword calls
            if all(self._looks_like_keyword(l) for l in lines):
                return "\n".join(lines)
            return lines[0]

        return response

    def _looks_like_keyword(self, line: str) -> bool:
        """Check if a line looks like a keyword call.

        Args:
            line: Line to check

        Returns:
            True if it looks like a keyword
        """
        if not line or line.startswith("#"):
            return False

        # Keywords typically start with a word followed by spaces/arguments
        parts = line.split()
        if not parts:
            return False

        # First word should be a keyword name (starts with letter, can contain spaces)
        first_word = parts[0]
        return first_word[0].isalpha() or first_word.startswith("$")


def parse_keyword_call(keyword_call: str) -> tuple:
    """Parse a keyword call string into keyword name and arguments.

    Args:
        keyword_call: Keyword call string (e.g., "Click    button#submit")

    Returns:
        Tuple of (keyword_name, args_list)
    """
    # Split by multiple spaces (RF separator) or tabs
    parts = []
    current_part = []

    i = 0
    while i < len(keyword_call):
        char = keyword_call[i]

        if char == "\t" or (char == " " and i + 1 < len(keyword_call) and keyword_call[i + 1] == " "):
            # Found separator
            if current_part:
                parts.append("".join(current_part))
                current_part = []
            # Skip all consecutive spaces/tabs
            while i < len(keyword_call) and keyword_call[i] in " \t":
                i += 1
            continue

        current_part.append(char)
        i += 1

    if current_part:
        parts.append("".join(current_part))

    if not parts:
        return ("", [])

    keyword_name = parts[0]
    args = parts[1:] if len(parts) > 1 else []

    return (keyword_name, args)
