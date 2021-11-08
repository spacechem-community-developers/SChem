# SChem

https://pypi.org/project/schem/

Clean Room implementation of the backend of SpaceChem (https://www.zachtronics.com/spacechem).

## Usage

Command line:
```
python -m schem [-h] [--version] [-l LEVEL_FILE] [--max-cycles MAX_CYCLES]
                [--check-precog] [--max-precog-check-cycles MAX_PRECOG_CHECK_CYCLES]
                [--json | --quiet] [--verbose] [--debug [DEBUG]]
                [solution_file]
```

E.g. `python -m schem` will validate the cycles-reactors-symbols score of any solution export in the user's clipboard.

This is roughly equivalent to:
```
import schem
schem.Solution(soln_str).validate(verbose=True)
```
