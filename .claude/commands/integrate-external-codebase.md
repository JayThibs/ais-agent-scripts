---
name: integrate-external-codebase
description: Integrate an external codebase for AI-assisted analysis and development. Example - /integrate-external-codebase https://github.com/user/repo. Creates comprehensive documentation in ai_docs/
---

<role>
You are a Codebase Integration Specialist who excels at:
- Analyzing external repositories and understanding their architecture
- Creating AI-optimized documentation and context
- Setting up efficient development workflows
- Building bridges between external code and your research
- Ensuring reproducible integration processes
</role>

<task_context>
The user wants to integrate an external codebase (GitHub repository or local path) into their AI-assisted workflow. This involves:
1. Fetching/cloning the codebase
2. Analyzing its structure and purpose
3. Creating comprehensive AI-friendly documentation
4. Setting up integration points
5. Generating summaries and context for future AI interactions
</task_context>

## Instructions

<instructions>
1. **Parse and Validate Input**
   <input_validation>
   - Extract URL or path from user input
   - Validate it's a valid GitHub URL or local directory
   - Determine integration approach:
     - GitHub URL → Clone to external_codebases/
     - Local path → Reference in place
   - Extract repository name for organization
   </input_validation>

2. **Fetch/Clone the Codebase**
   <fetch_codebase>
   For GitHub repositories:
   ```bash
   cd external_codebases/
   git clone [repository_url] [repo_name]
   ```
   
   For local paths:
   - Verify the path exists
   - Create symbolic link if appropriate
   - Document the path reference
   </fetch_codebase>

3. **Analyze Codebase Structure**
   <structure_analysis>
   Run comprehensive analysis:
   - Identify main programming languages
   - Detect frameworks and dependencies
   - Map directory structure
   - Find entry points and key files
   - Locate documentation (README, docs/)
   - Identify test suites
   - Check for configuration files
   
   Use the python integration script:
   ```bash
   python scripts/integrate_codebase.py [repository_url_or_path]
   ```
   </structure_analysis>

4. **Create AI-Optimized Documentation**
   <documentation_creation>
   Generate documentation in `ai_docs/codebases/[repo_name]/`:
   
   a) **Overview** (`overview.md`):
      - Repository purpose and goals
      - Key features and capabilities
      - Architecture overview
      - Technology stack
   
   b) **Structure Map** (`structure_map.md`):
      - Directory tree with descriptions
      - Key file locations and purposes
      - Module relationships
      - Data flow diagrams
   
   c) **Key Components** (`components/`):
      - Separate files for major components
      - Function/class documentation
      - API references
      - Important algorithms
   
   d) **Integration Guide** (`integration_guide.md`):
      - How to use this codebase
      - Common workflows
      - Integration points
      - Example usage patterns
   
   e) **Context Summary** (`context_summary.md`):
      - Condensed version for AI context
      - Most important information
      - Quick reference guide
   </documentation_creation>

5. **Extract and Summarize Key Information**
   <information_extraction>
   Use Gemini for large-scale analysis:
   ```bash
   # Analyze entire codebase
   gemini -p "@external_codebases/[repo_name]/**/*.{py,js,ts,go,rs,java} \
   Analyze this codebase and provide:
   1. Main purpose and functionality
   2. Key architectural patterns
   3. Important algorithms or techniques
   4. External dependencies
   5. Potential integration points"
   ```
   
   For specific components:
   ```bash
   # Analyze documentation
   gemini -p "@external_codebases/[repo_name]/README.md @external_codebases/[repo_name]/docs/** \
   Summarize the key concepts and usage patterns"
   ```
   </information_extraction>

6. **Create Integration Configuration**
   <integration_config>
   Create `.claude/integrations/[repo_name]/config.json`:
   ```json
   {
     "repository": "[url_or_path]",
     "integrated_at": "[timestamp]",
     "language": "[primary_language]",
     "frameworks": ["framework1", "framework2"],
     "entry_points": ["main.py", "index.js"],
     "key_files": ["path/to/important/file.py"],
     "ai_docs_path": "ai_docs/codebases/[repo_name]/",
     "update_frequency": "weekly"
   }
   ```
   </integration_config>

7. **Generate Quick Reference**
   <quick_reference>
   Create a cheat sheet for common tasks:
   - How to run the project
   - Key commands and scripts
   - Important functions/classes
   - Common customization points
   - Debugging tips
   </quick_reference>

8. **Set Up Continuous Integration**
   <continuous_integration>
   If requested, set up automated updates:
   - Git hooks for pulling updates
   - Scheduled documentation regeneration
   - Change detection and notification
   </continuous_integration>
</instructions>

## Output Format

<output_format>
```markdown
# Successfully Integrated: [Repository Name]

## 📦 Repository Details
- **Source**: [URL or path]
- **Primary Language**: [Language]
- **Size**: [Size] files, [LOC] lines of code
- **Last Updated**: [Date]

## 📁 Documentation Created
✅ Created comprehensive documentation in `ai_docs/codebases/[repo_name]/`:
- `overview.md` - High-level repository overview
- `structure_map.md` - Complete directory and file mapping  
- `components/` - Detailed component documentation
- `integration_guide.md` - How to work with this codebase
- `context_summary.md` - AI-optimized context summary

## 🔍 Key Findings
### Purpose
[2-3 sentence summary of what this codebase does]

### Architecture
- **Pattern**: [e.g., MVC, microservices, monolithic]
- **Entry Points**: [main files]
- **Key Dependencies**: [major frameworks/libraries]

### Integration Opportunities
1. [How this can be used in your research]
2. [Potential modifications or extensions]
3. [Useful components to leverage]

## 🚀 Quick Start
```bash
# To use this codebase
cd external_codebases/[repo_name]
[setup commands]
```

## 📖 Next Steps
1. Review the generated documentation in `ai_docs/codebases/[repo_name]/`
2. Use `/crud-claude-commands create [repo_name]-helper` to create custom commands
3. Reference the context summary when working with this codebase

## 🤖 AI Usage Tips
When working with this codebase, reference it using:
- `@ai_docs/codebases/[repo_name]/context_summary.md` for quick context
- `@ai_docs/codebases/[repo_name]/components/[component].md` for specific components
- `@external_codebases/[repo_name]/[path]` for actual code files

Integration configuration saved to: `.claude/integrations/[repo_name]/config.json`
```
</output_format>

## Examples

<examples>
### Example 1: GitHub Repository
Input: `/integrate-external-codebase https://github.com/anthropics/anthropic-sdk-python`

Process:
1. Clone to `external_codebases/anthropic-sdk-python/`
2. Analyze Python SDK structure
3. Generate API documentation
4. Create integration examples
5. Set up for AI-assisted SDK usage

### Example 2: Local Research Codebase  
Input: `/integrate-external-codebase ~/research/neural-compression`

Process:
1. Create reference to local directory
2. Analyze research code structure
3. Document experiments and models
4. Create research workflow guides
5. Enable AI assistance for experiments

### Example 3: Complex Framework
Input: `/integrate-external-codebase https://github.com/pytorch/pytorch`

Process:
1. Clone repository (note: large size warning)
2. Focus on most relevant modules
3. Create condensed documentation
4. Build quick reference for common operations
5. Set up for AI-assisted PyTorch development
</examples>

## Error Handling

<error_handling>
Common issues and solutions:

1. **Repository Too Large**
   - Offer shallow clone option
   - Focus on specific subdirectories
   - Use sparse checkout

2. **Private Repository**
   - Request authentication
   - Guide through token setup
   - Offer SSH clone alternative

3. **Complex Dependencies**
   - Document requirements clearly
   - Create setup scripts
   - Note potential issues

4. **Documentation Generation Fails**
   - Fall back to basic structure mapping
   - Use file analysis instead of execution
   - Create placeholder docs for manual filling
</error_handling>

## Best Practices

<best_practices>
1. **Start Small**: For large codebases, begin with core modules
2. **Focus on Your Needs**: Document what's relevant to your research
3. **Keep Updated**: Set reminders to refresh documentation
4. **Create Commands**: Build custom commands for common operations
5. **Use Summaries**: Rely on context summaries to save tokens
</best_practices>

## Integration with Other Commands

<integration>
Works seamlessly with:
- `/setup` - Part of initial research setup
- `/crud-claude-commands` - Create codebase-specific commands
- `/parallel-analysis-example` - Analyze codebase from multiple angles
- `/clean-and-organize` - Keep integration directory clean
</integration>

Remember: Good integration is about making external code feel like a natural extension of your research workflow. Focus on understanding and documenting what matters most for your specific use case.