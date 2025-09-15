import sys, MetaTrader5 as mt5, pandas as pd, numpy as np, ta
print("Python:", sys.version.split()[0])
print("MT5:", getattr(mt5,"__version__","unknown"))
print("pandas:", pd.__version__, "numpy:", np.__version__, "ta:", getattr(ta,"__version__","unknown"))
