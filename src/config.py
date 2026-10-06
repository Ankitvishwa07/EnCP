TARGET = "active_kwh"
TRAIN_END = "2009-12-31 23:00"
VAL_END   = "2010-03-31 23:00"      # train: to end 2009 | val: Jan-Mar 2010 | test: Apr-Nov 2010
HORIZONS  = [1, 24, 168]
OFFPEAK_HOURS = list(range(22, 24)) + list(range(0, 6))
PRICE_OFFPEAK, PRICE_NORMAL = 0.15, 0.25    # EUR/kWh, illustrative