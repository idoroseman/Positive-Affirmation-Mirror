import sys
from pathlib import Path
import time
import numpy as np
from dataclasses import dataclass, field

# Add src to sys.path
sys.path.append('src')

from mirror.app import recordingSelector
from mirror.tracker import Track

def test():
    recordings_dir = Path("data/recordings")
    selector = recordingSelector(recordings_dir=recordings_dir)
    now = time.time()
    dwell_seconds = 0

    print("--- 1. Known person 'ido' ---")
    t_ido = Track(track_id=1, bbox=(0,0,0,0), centroid=(0,0), encoding=np.array([]), 
                  first_seen_at=now - 10, last_seen_at=now, person_name="ido", 
                  demographics_label=None)
    res_ido = selector.choose([t_ido], dwell_seconds, now)
    print(f"Result for 'ido': {res_ido}")

    print("\n--- 2. Unknown track with gender 'man' ---")
    t_man = Track(track_id=2, bbox=(0,0,0,0), centroid=(0,0), encoding=np.array([]), 
                  first_seen_at=now - 10, last_seen_at=now, person_name=None, 
                  demographics_label=None)
    t_man.metadata["gender"] = "man"
    res_man = selector.choose([t_man], dwell_seconds, now)
    print(f"Result for 'man': {res_man}")

    print("\n--- 3. Direct directory pick for 'couple' ---")
    res_couple = selector._pick_from_directory(recordings_dir / "couple")
    print(f"Result for 'couple' directory: {res_couple}")

    print("\n--- 4. Direct directory pick for 'group' ---")
    res_group = selector._pick_from_directory(recordings_dir / "group")
    print(f"Result for 'group' directory: {res_group}")

if __name__ == "__main__":
    test()
