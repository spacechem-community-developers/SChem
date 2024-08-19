from pathlib import Path
import sys

# Insert the parent directory to sys path so schem is accessible even if it's not available system-wide
sys.path.insert(1, str(Path(__file__).parent.parent))

from schem.level import Level, DefenseLevel
from schem.levels import levels
from pprint import PrettyPrinter
import json

# Dump all existing levels (for reference purposes)
with open('out.txt', 'w', encoding='utf-8') as f:
    pp = PrettyPrinter(indent=2, stream=f)
    for name, level in levels.items():
      pp.pprint(name)
      try:
          l = Level(level)
          pp.pprint(l.dict)
      except:
          pass

# Here are a couple of defence levels to get you started
l = DefenseLevel(None)
l.dict = {
    'name': 'A Most Unfortunate Malfunction',
    'author': 'Zach',
    'type': 'defense',
    'boss': 'Isambard MMD',
    'max-reactors': 3,
    'control-switches-allowed': True,
    'has-starter': True,
    'has-storage': True,
    'fixed-input-zones': {'0': 'Methane;CH~04;01110;10101;21100;12100;11611'},
    'other-components': [{
        'molecules': [{'count': 36, 'molecule': 'Methane;CH~04;01110;10101;21100;12100;11611'}],
        'type': 'drag-weapon-oxygentank',
        'x': 12,
        'y': 12,
    }, {
        'molecules': [{'count': 36, 'molecule': 'Methane;CH~04;01110;10101;21100;12100;11611'}],
        'type': 'drag-weapon-oxygentank',
        'x': 18,
        'y': 12,
    }, {
        'molecules': [{'count': 36, 'molecule': 'Methane;CH~04;01110;10101;21100;12100;11611'}],
        'type': 'drag-weapon-oxygentank',
        'x': 24,
        'y': 12,
    }],
}
print(l.code)


l = DefenseLevel(None)
l.dict = {
    'name': 'No Need for Introductions',
    'author': 'Zach',
    'type': 'defense',
    'boss': 'Quororque',
    'max-reactors': 4,
    'control-switches-allowed': True,
    'has-advanced': True,
    'has-storage': True,
    'other-components': [{
        'molecules': [
            {'count': 36, 'molecule': 'Methane;CH~04;01110;10101;21100;12100;11611'},
            {'count': 36, 'molecule': 'Methane;CH~04;01110;10101;21100;12100;11611'}
        ],
        'type': 'drag-weapon-particleaccelerator',
        'x': 16,
        'y': 11,
    }],
    'random-input-zones': {'0': {
      'inputs': [
          {'count': 2, 'molecule': 'Water;H~02O;11110;22100;21801'},
          {'count': 2, 'molecule': 'Tainted Water;UOH;21801;11110;229200'},
          {'count': 2, 'molecule': 'Tainted Water;U~02O;21801;119210;229200'},
      ],
      'random-seed': 19,
    }},
}
print(l.code)