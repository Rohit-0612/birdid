"""Make the project root importable so `import bird_core` works from tests/."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tests must never write to the real evaluation directory or the real life list.
# Set before bird_core is imported, since it reads these at module level.
os.environ.setdefault("BIRD_SAVE_DIR",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "_artifacts"))
