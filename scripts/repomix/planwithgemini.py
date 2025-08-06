#!/usr/bin/env python3
"""
planwithgemini.py - Intelligent planning with repomix and Gemini
Combines repomix's file packing capabilities with Gemini's analysis for comprehensive planning.
Supports: gemini, ollama, anthropic CLI, or direct file output for manual analysis.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# Color codes for terminal output
class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    NC = "\033[0m"  # No Color


def print_colored(message: str, color: str = Colors.NC):
    """Print colored message to terminal."""
    print(f"{color}{message}{Colors.NC}")


def detect_available_llm() -> Optional[str]:
    """Detect which LLM CLI is available."""
    llms = {
        "gemini": ["gemini", "--help"],
        "ollama": ["ollama", "list"],
        "anthropic": ["anthropic", "--help"],
        "openai": ["openai", "--help"],
    }

    for llm, cmd in llms.items():
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return llm
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    return None


def check_dependencies() -> Tuple[bool, List[str], Optional[str]]:
    """Check if required tools are installed and detect available LLM."""
    missing = []

    # Check repomix
    try:
        subprocess.run(["repomix", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("repomix (install with: npm install -g repomix)")

    # Detect available LLM
    available_llm = detect_available_llm()

    return len(missing) == 0, missing, available_llm


def extract_keywords(task: str) -> List[str]:
    """Extract relevant keywords from task description for smart file selection."""
    # Common words to exclude
    stop_words = {
        "the",
        "and",
        "or",
        "for",
        "with",
        "from",
        "to",
        "in",
        "on",
        "at",
        "of",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "implement",
        "create",
        "add",
        "update",
        "modify",
        "change",
        "refactor",
    }

    # Extract words
    words = re.findall(r"\b[a-z]+\b", task.lower())

    # Filter out stop words and short words
    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    # Also extract CamelCase and snake_case terms
    special_terms = re.findall(r"[A-Z][a-z]+|[a-z]+_[a-z]+", task)
    keywords.extend([t.lower() for t in special_terms])

    return list(set(keywords))


def build_repomix_command(
    mode: str, task: str, output_file: str, config_file: str, additional_args: Dict
) -> List[str]:
    """Build repomix command based on mode and parameters."""
    base_cmd = [
        "repomix",
        "--style",
        "xml",
        "--output",
        output_file,
    ]

    if config_file and os.path.exists(config_file):
        base_cmd.extend(["--config", config_file])

    if mode == "auto":
        # Smart file selection based on keywords
        keywords = extract_keywords(task)
        print_colored(f"Keywords identified: {', '.join(keywords)}", Colors.BLUE)

        for keyword in keywords:
            base_cmd.extend(["--include", f"**/*{keyword}*,**/{keyword}/**"])
        base_cmd.extend(["--compress"])

    elif mode == "full":
        # Full codebase analysis
        base_cmd.extend(
            [
                "--compress",
                "-i",
                "**/node_modules/**,**/dist/**,**/.git/**,**/coverage/**",
            ]
        )

    elif mode == "quick":
        # Quick compressed analysis
        base_cmd.extend(["--compress", "--remove-comments", "--remove-empty-lines"])

    elif mode == "security":
        # Security-focused analysis
        base_cmd.extend(
            [
                "--include",
                "**/auth/**,**/security/**,**/*config*,**/*.env*",
            ]
        )

    # Add any additional includes/excludes
    if "includes" in additional_args and additional_args["includes"]:
        patterns = ",".join(additional_args["includes"])
        base_cmd.extend(["--include", patterns])

    if "excludes" in additional_args and additional_args["excludes"]:
        patterns = ",".join(additional_args["excludes"])
        base_cmd.extend(["-i", patterns])

    # Add target directory
    base_cmd.append(additional_args.get("target_dir", "."))

    return base_cmd


def create_prompt(mode: str, task: str) -> str:
    """Create appropriate prompt based on mode and task."""
    prompts = {
        "auto": f"""Analyze this codebase context and create a detailed implementation plan for: {task}

Focus on:
1. Understanding current architecture and patterns
2. Identifying integration points for the new feature
3. Potential risks and dependencies
4. Step-by-step implementation approach with specific code changes
5. Testing and validation strategy
6. Performance considerations

Provide a structured plan with:
- Executive summary
- Implementation phases with time estimates
- Specific files to modify with rationale
- Risk assessment and mitigation
- Success criteria
- Code examples where helpful""",
        "full": f"""Perform a comprehensive analysis of this entire codebase for: {task}

Provide:
1. Current Architecture Assessment
   - Design patterns and principles in use
   - Technical debt and improvement areas
   - Component coupling and cohesion analysis

2. Implementation Strategy
   - Detailed refactoring/implementation plan
   - Migration approach if applicable
   - Backward compatibility considerations

3. System-wide Impact Analysis
   - Direct and indirect effects
   - Breaking changes identification
   - Performance implications

4. Risk Management
   - Critical risks and mitigation strategies
   - Rollback procedures
   - Incremental deployment approach

5. Testing and Validation
   - Comprehensive testing strategy
   - Performance benchmarks
   - Monitoring requirements

Structure as a detailed, actionable plan with phases, milestones, and specific tasks.""",
        "quick": f"""Quickly analyze this codebase and provide a concise implementation plan for: {task}

Focus on:
1. Key changes required
2. Main risks and how to address them
3. Implementation steps (high-level)
4. Testing approach
5. Estimated timeline

Keep the plan practical and actionable.""",
        "security": f"""Perform a security-focused analysis for: {task}

Examine:
1. Authentication and Authorization
   - Current implementation review
   - Vulnerabilities and weaknesses
   - Best practice compliance

2. Data Security
   - Sensitive data handling
   - Encryption usage
   - Data flow analysis

3. API Security
   - Endpoint protection
   - Input validation
   - Rate limiting and abuse prevention

4. Security Improvements
   - Prioritized recommendations
   - Implementation approach
   - Testing requirements

Provide actionable security improvements with implementation details.""",
    }

    return prompts.get(mode, f"Analyze this codebase and provide insights for: {task}")


def run_repomix(cmd: List[str], verbose: bool = False) -> Tuple[bool, str]:
    """Run repomix command and return success status and output."""
    if verbose:
        print_colored(f"Running: {' '.join(cmd)}", Colors.YELLOW)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if verbose and result.stdout:
            print(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print_colored(f"Error running repomix: {e.stderr}", Colors.RED)
        return False, e.stderr


def run_llm_analysis(
    context_file: str, prompt: str, output_file: Optional[str], llm_type: Optional[str]
) -> Tuple[bool, str]:
    """Run LLM analysis on the context file using available LLM."""
    if not llm_type:
        print_colored(
            "No LLM CLI detected. Saving context and prompt for manual analysis.",
            Colors.YELLOW,
        )

        # Save prompt to file for manual use
        prompt_file = context_file.replace(".xml", "_prompt.txt")
        with open(prompt_file, "w") as f:
            f.write(prompt)

        instructions = f"""
Context file saved to: {context_file}
Prompt saved to: {prompt_file}

You can analyze this manually by:
1. Using any LLM web interface (Claude, ChatGPT, etc.)
2. Paste the prompt from {prompt_file}
3. Upload or paste the context from {context_file}
"""
        if output_file:
            with open(output_file, "w") as f:
                f.write(instructions)
        print(instructions)
        return True, instructions

    print_colored(f"Running {llm_type} analysis...", Colors.BLUE)

    if llm_type == "gemini":
        cmd = ["gemini", "-p", f"@{context_file}", prompt]
    elif llm_type == "ollama":
        # For ollama, we need to format differently
        with open(context_file, "r") as f:
            context = f.read()
        full_prompt = f"{prompt}\n\nContext:\n{context}"
        cmd = ["ollama", "run", "llama2", full_prompt]
    elif llm_type == "anthropic":
        with open(context_file, "r") as f:
            context = f.read()
        full_prompt = f"{prompt}\n\nContext:\n{context}"
        cmd = [
            "anthropic",
            "messages",
            "create",
            "-m",
            "claude-3-sonnet",
            "-c",
            full_prompt,
        ]
    else:
        return False, f"Unsupported LLM type: {llm_type}"

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        if output_file:
            with open(output_file, "w") as f:
                f.write(result.stdout)
            print_colored(f"Plan saved to: {output_file}", Colors.GREEN)
        else:
            print(result.stdout)

        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print_colored(f"Error running {llm_type}: {e.stderr}", Colors.RED)
        return False, e.stderr


def get_file_stats(file_path: str) -> Dict[str, str]:
    """Get basic statistics about a file."""
    if not os.path.exists(file_path):
        return {}

    stats = os.stat(file_path)
    size = stats.st_size

    # Convert to human-readable size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            size_str = f"{size:.1f} {unit}"
            break
        size /= 1024.0
    else:
        size_str = f"{size:.1f} TB"

    return {
        "size": size_str,
        "modified": datetime.fromtimestamp(stats.st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent planning with repomix and LLM analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s auto "implement caching layer"
  %(prog)s full "refactor authentication system" -o auth-refactor-plan.md
  %(prog)s security "audit API endpoints"
  %(prog)s custom "analyze performance" -- --include "**/perf/**"
        """,
    )

    parser.add_argument(
        "mode",
        choices=["auto", "full", "quick", "security", "custom"],
        help="Analysis mode",
    )
    parser.add_argument("task", help="Task description for planning")
    parser.add_argument("-c", "--config", help="Custom config file")
    parser.add_argument("-o", "--output", help="Save plan to file")
    parser.add_argument(
        "-i",
        "--include",
        action="append",
        default=[],
        help="Additional include patterns",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        action="append",
        default=[],
        help="Additional exclude patterns",
    )
    parser.add_argument("-t", "--target", default=".", help="Target directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--keep-context", action="store_true", help="Keep context file after analysis"
    )
    parser.add_argument(
        "--llm",
        choices=["gemini", "ollama", "anthropic", "auto"],
        default="auto",
        help="LLM to use (default: auto-detect)",
    )

    args, unknown = parser.parse_known_args()

    # Check dependencies
    deps_ok, missing, available_llm = check_dependencies()
    if not deps_ok:
        print_colored("Missing dependencies:", Colors.RED)
        for dep in missing:
            print_colored(f"  - {dep}", Colors.RED)
        print_colored("\nRun: ./scripts/repomix/setup_dependencies.sh", Colors.YELLOW)
        sys.exit(1)

    # Determine which LLM to use
    if args.llm == "auto":
        llm_type = available_llm
    else:
        llm_type = args.llm if args.llm != "auto" else available_llm

    if llm_type:
        print_colored(f"Using {llm_type} for analysis", Colors.GREEN)
    else:
        print_colored(
            "No LLM CLI detected - will save context for manual analysis", Colors.YELLOW
        )

    # Set up paths
    temp_dir = Path(".temp")
    temp_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    context_file = temp_dir / f"context_{args.mode}_{timestamp}.xml"

    config_file = args.config or "repomix-gemini.config.json"

    print_colored(
        f"Planning with {llm_type or 'manual analysis'} - Mode: {args.mode}",
        Colors.GREEN,
    )
    print_colored(f"Task: {args.task}", Colors.GREEN)
    print()

    # Build repomix command
    additional_args = {
        "includes": args.include,
        "excludes": args.exclude,
        "target_dir": args.target,
    }

    if args.mode == "custom" and unknown:
        # For custom mode, use the additional arguments as repomix args
        cmd = ["repomix"] + unknown + ["--output", str(context_file)]
    else:
        cmd = build_repomix_command(
            args.mode, args.task, str(context_file), config_file, additional_args
        )

    # Run repomix
    print_colored(f"Analyzing codebase with repomix ({args.mode} mode)...", Colors.BLUE)
    success, output = run_repomix(cmd, args.verbose)

    if not success:
        sys.exit(1)

    # Check if context file was created
    if not context_file.exists():
        print_colored("Error: Failed to generate context with repomix", Colors.RED)
        sys.exit(1)

    # Show context stats
    stats = get_file_stats(str(context_file))
    if stats:
        print_colored(f"Context file generated: {stats['size']}", Colors.PURPLE)

    # Parse token count from repomix output if available
    token_match = re.search(r"Total tokens:\s*(\d+)", output)
    if token_match:
        tokens = int(token_match.group(1))
        print_colored(f"Total tokens: {tokens:,}", Colors.PURPLE)

    # Create prompt
    prompt = create_prompt(args.mode, args.task)

    # Run LLM analysis
    success, output = run_llm_analysis(str(context_file), prompt, args.output, llm_type)

    if not success:
        sys.exit(1)

    # Cleanup
    if not args.keep_context and context_file.exists():
        context_file.unlink()
    elif args.keep_context:
        print_colored(f"Context file kept at: {context_file}", Colors.YELLOW)

    print_colored("Planning complete!", Colors.GREEN)


if __name__ == "__main__":
    main()
