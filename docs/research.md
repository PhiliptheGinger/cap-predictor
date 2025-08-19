# Research Hooks

The research package exposes a light‑weight interface for experimenting with new strategies.

```python
from sentimental_cap_predictor.research.idea_schema import Idea
from sentimental_cap_predictor.research.apply_idea import apply_idea

def backtest(idea: Idea):
    # user provided logic
    return idea.params

idea = Idea(name="My idea", params={"window": 5})
result = apply_idea(idea, backtest)
```
