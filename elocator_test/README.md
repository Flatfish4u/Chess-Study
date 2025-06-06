# Elocator Test

Chess position complexity analysis using the Elocator API.

## Installation

```bash
pip install -e .
```

## Usage

```python
from elocator_test.elocator_api import get_position_complexity_with_retry
from elocator_test.model import ChessModel

# Get complexity score for a position
complexity = get_position_complexity_with_retry("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
print(f"Position complexity: {complexity}")
```
