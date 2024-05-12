from pathlib import Path
import sys

# Insert the parent directory to sys path so schem is accessible even if it's not available system-wide
sys.path.insert(1, str(Path(__file__).parent.parent))

from schem.components import RandomInput
from schem.schem_random import SChemRandom

# Not all random sequences start from seed 0; use this tool to figure out what seed the random is using.
# In this case, the sequence is: Number of uranium atoms in the random inputs of "No Need for Introductions"
expected_sequence = [
    1, 0, 0, 2, 2,
    1, 1, 2, 0, 0,
    1, 2, 2, 0, 1,
]

candidate_seeds = []
for seed in range(1000):
    random_generator = SChemRandom(seed=seed)
    random_bucket = []

    for i, expected in enumerate(expected_sequence):
        if not random_bucket:
            random_bucket = [0] * 2 + [1] * 2 + [2] * 2

        if random_bucket.pop(random_generator.next(len(random_bucket))) != expected:
            break

        if i == len(expected_sequence) - 1:
            candidate_seeds.append(seed)


print(f'Found {len(candidate_seeds)} candidate seeds:')
for seed in candidate_seeds:
    print(f'Seed {seed} predicts the following outputs after the expected sequence:')

    random_generator = SChemRandom(seed=seed)
    random_bucket = []

    for i in range(len(expected_sequence) + 10):
        if not random_bucket:
            random_bucket = [0] * 2 + [1] * 2 + [2] * 2

        actual = random_bucket.pop(random_generator.next(len(random_bucket)))

        if i > len(expected_sequence) - 1:
            print(actual, end = ', ')
