try:
    import pandas_ta as ta
    import pandas as pd
    print("pandas-ta import OK. Version:", getattr(ta, "__version__", "unknown"))
except Exception as e:
    import traceback
    print("Import failed:", e)
    traceback.print_exc()
