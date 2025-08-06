#!/usr/bin/env python3
"""
Simple test script for repomix + analysis workflow
"""

import subprocess
import os
from pathlib import Path
from datetime import datetime

# Create temp directory
temp_dir = Path(".temp")
temp_dir.mkdir(exist_ok=True)

# Generate context with repomix
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
context_file = temp_dir / f"workshop_analysis_{timestamp}.xml"

print("📦 Running repomix to analyze workshop documentation...")
cmd = [
    "repomix",
    "--style",
    "xml",
    "--output",
    str(context_file),
    "--compress",
    "--include",
    "**/workshop/**,**/docs/**,**/*.md",
    "-i",
    "**/node_modules/**,**/.git/**",
    ".",
]

result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(f"Error: {result.stderr}")
    exit(1)

# Check file size
if context_file.exists():
    size = context_file.stat().st_size / 1024 / 1024  # MB
    print(f"\n✅ Context file created: {context_file}")
    print(f"   Size: {size:.2f} MB")

    # Create analysis prompt
    prompt_file = context_file.with_suffix(".prompt.txt")
    prompt = """Analyze this codebase focusing on the workshop documentation structure.

Please provide:
1. Overview of the workshop materials and their organization
2. Key topics covered in the workshop
3. Documentation quality assessment
4. Suggestions for improvement
5. Missing components or gaps

Focus particularly on:
- The participant guide
- Claude Code integration documentation
- Hands-on exercises and examples
"""

    with open(prompt_file, "w") as f:
        f.write(prompt)

    print(f"\n📝 Analysis prompt saved to: {prompt_file}")
    print("\n🎯 Next steps:")
    print("1. The context file contains the packed codebase")
    print("2. Use the prompt with any LLM (Claude, ChatGPT, etc.)")
    print("3. Or use with Gemini CLI if available: gemini -p @context_file prompt")
else:
    print("❌ Failed to create context file")
