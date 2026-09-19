import numpy as np
import pandas as pd
from src.hyperparameter_optimizer_v3_1 import V31HyperparameterOptimizer

# We'll run the optimizer for DK1 and DK2
for area in ["DK1", "DK2"]:
    opt = V31HyperparameterOptimizer(price_area=area)
    # The optimizer needs X, y. Let's load the V3.1 features
    # Wait, the optimizer already has methods to get data. Let's see if there's a build method.
