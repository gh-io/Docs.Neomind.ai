"""
Quick-start for NeomindAI without hardware.
Simulates sensor inputs and allows you to test AI modules.
"""

import sys
import os
import random
import numpy as np

# Add repo folder to path if not installed
repo_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(repo_path)

# Import core AI modules
try:
    from NeomindAI import NeomindAI  # main AI class
    import neomind  # support functions
except ImportError:
    print("Warning: Could not import core AI modules. Check paths.")
    NeomindAI = None

# Simulate sensor inputs (replace hardware calls)
def fake_sensor_input():
    return {
        "temperature": random.uniform(20, 30),
        "pressure": random.uniform(950, 1050),
        "vision_data": np.random.rand(64, 64),  # dummy image
    }

# Initialize AI
if NeomindAI:
    ai = NeomindAI()
    print("NeomindAI initialized successfully!")

    # Run a loop of experiments
    for step in range(5):
        inputs = fake_sensor_input()
        print(f"\n[Step {step+1}] Simulated inputs: {inputs}")
        try:
            # Example: run AI decision method
            if hasattr(ai, "process_inputs"):
                output = ai.process_inputs(inputs)
            elif hasattr(ai, "run"):
                output = ai.run(inputs)
            else:
                output = "No process method found in NeomindAI class"
            print(f"AI output: {output}")
        except Exception as e:
            print(f"Error during AI execution: {e}")
else:
    print("NeomindAI module not available. Exiting.")
