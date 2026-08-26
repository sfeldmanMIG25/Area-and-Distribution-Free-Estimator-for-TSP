# Area and Distribution Free Estimators for TSP
Code Associated with the paper "	Area and Distribution Free Estimators for TSP"

When unpacking the data use the Chunk_Archiver script and change the mode to unpack after cloning the repository. The data is stored in 50 mb zip bundles handled by LFS.

## Repo Intelligence Toolkit

This repository is integrated with a suite of local-first repository intelligence tools to optimize developer and AI agent workflows.

### Available Tools

1. **GrepAI** (`sgrep` or `grepai`) — *Semantic Search & Call Tracing*
   - **When to use**: When searching by concept/intent (natural language queries) rather than literal matching, or when tracing function calling paths across the codebase.
   - **Example Queries**:
     - `grepai search "where is routing cost calculated"`
     - `grepai search "where are errors logged"`
     - `grepai trace callers "parse_tsp_instance"`

2. **AST-Grep** (`sg` or `ast-grep`) — *AST Structural Search*
   - **When to use**: When matching code syntax patterns (like calling specific functions, repeated variables, or nested conditions) with 100% precision and zero false positives.
   - **Example Queries**:
     - `sg run -p "cdist(\$A, \$A, \$\$\$)" -l python` (find repeated arguments in `cdist` calls)
     - `sg run -p "if \$COND: \$FUNC(\$\$\$)" -l python` (find function calls inside conditionals)

3. **CodeGraph** (`cg` or `codegraph`) — *Code Graph / Dependency & Impact Analysis*
   - **When to use**: When evaluating the structural impact of modifying a symbol, or inspecting caller/callee trees and dependencies in a single step.
   - **Example Queries**:
     - `codegraph callers parse_tsp_instance` (who calls/imports this symbol)
     - `codegraph impact parse_tsp_instance` (recursively show all downstream affected symbols)

### Loading Aliases

To load CLI aliases in your shell session, run:
- **Bash/Git Bash**: `source ./scripts/aliases.sh`
- **PowerShell**: `. .\scripts\aliases.ps1`

### High-Value Integrations

1. **Auto-Scan Rules (AST-Grep)**:
   - Run `sg scan` in the root of the project to check for project-specific rules, such as preferring optimized Delaunay functions from `mst_utils` over raw library calls.
2. **Git Impact Checker (CodeGraph)**:
   - Run `./scripts/check_git_impact.sh` (or `. .\scripts\check_git_impact.ps1` in PowerShell) to scan your current git diff and recursively map the downstream impact of all modified symbols before committing.
3. **MCP Integration for Copilot/Agents**:
   - Run `./scripts/install_mcp.sh` (or `. .\scripts\install_mcp.ps1` in PowerShell) to automatically configure both GrepAI and CodeGraph as Model Context Protocol (MCP) servers in Claude Code and Cursor.
