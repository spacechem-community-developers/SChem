# SChem

https://pypi.org/project/schem/

Clean Room implementation of the backend of SpaceChem (https://www.zachtronics.com/spacechem).

## Usage

`python -m schem [solution_file]`

Will validate the cycles-reactors-symbols score of the provided solution file, or of any solution in the user's
clipboard if a file is not specified.

This is equivalent to:

```
import schem
schem.validate(soln_str, verbose=True)
```
