# SChem

https://pypi.org/project/schem/. Install with `pip install schem`

Clean Room implementation of the backend of SpaceChem (https://www.zachtronics.com/spacechem).

## Usage (CLI)

```
python -m schem [-h] [--version] [-l LEVEL_FILE] [--max-cycles MAX_CYCLES]
                [--check-precog] [--max-precog-check-cycles MAX_PRECOG_CHECK_CYCLES]
                [--seed SEED] [--hash-states HASH_STATES]
                [--export] [--no-run] [--strict]
                [--json | --verbose] [--debug [DEBUG]]
                [solution_files ...]
```

E.g. `python -m schem` will validate the cycles-reactors-symbols score of any solution export(s) in the user's clipboard. See `python -m schem --help` for details.

## Usage (python)

Supposing `level_export`, `solution_export` are strings as exported by SpaceChem CE:
```python
from schem import Level, Solution

# Load a solution
solution = Solution(solution_export)  # Auto-use appropriate official level
solution = Solution(solution_export, level=level_export)  # Custom level
solution = Solution(solution_export, level=Level(level_export))  # Alternative

# Run a solution
solution.run()
# => Score(cycles=45, reactors=1, symbols=14)

# Check the expected score that was in the export's metadata
solution.expected_score
# => Score(cycles=44, reactors=1, symbols=14)

# Reset the run state of a solution
solution.reset()

# Validate that a solution matched its expected score
solution.validate()
# =/> ScoreError("[Of Pancakes and Spaceships] 44-1-14 "Cycles" by Zig: Expected 44 cycles but got 45.")

# Check if a solution uses precognition
solution.is_precognitive()  # slow
# => False

# Bundle method for calling validate() if expected score is present, else run(), optionally checking precog,
# and returning a dict of all this info and any error
solution.evaluate()
# => {"level_name": "Tunnels III",
#     "resnet_id": (1, 1, 3),  # Volume, Issue, Puzzle
#     "author": "Zig",
#     "cycles": 244,
#     "reactors": 1,
#     "symbols": 14,
#     "solution_name": "symbols",
#     "error": ScoreError("[Tunnels III] 243-1-14 \"symbols\" by Zig: Expected 243 cycles but got 244.")
#}

solution.evaluate(check_precog=True)
# => {"level_name": "Challenge: Going Green",
#     "author": "Zig",
#     "cycles: 3578,
#     "reactors": 1,
#     "symbols": 103,
#     "solution_name": "assumes 2nd input",
#     "precog": true,
#     "precog_explanation": "Solution is precognitive; failed whenever molecule 2 was Hydrogen Sulfide, for 9 such
#                            appearances (whereas solution success rate was otherwise 100%)."
#}

# Re-export the solution. Sorts export lines to ensure uniqueness
solution.export_str()
# => "SOLUTION:..."
```
