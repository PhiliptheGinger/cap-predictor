from __future__ import annotations

"""Prompt templates for idea generation and critique."""

IDEA_MINER_PROMPT = """
You are an expert quantitative researcher tasked with proposing a new trading
idea.  The dataset provides the following fields:

{available_fields}

Using only these fields, suggest a single research idea.  Respond with a YAML
object following this template:

```yaml
name: <short descriptive title>
description: <one or two sentence summary of the hypothesis>
params:
  <parameter_name>: <description or default value>
```
"""

CRITIC_PROMPT = """
You are reviewing a proposed trading idea and offering constructive criticism.
The dataset exposes the following fields:

{available_fields}

Provide concise feedback on the idea and suggest improvements.  Your response
must be a YAML object formatted as:

```yaml
verdict: <accept|revise|reject>
critique: <brief analysis of strengths and weaknesses>
suggestions:
  - <first improvement suggestion>
  - <second improvement suggestion>
```
"""

__all__ = ["IDEA_MINER_PROMPT", "CRITIC_PROMPT"]
