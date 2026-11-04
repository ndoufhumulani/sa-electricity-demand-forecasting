# Data

The raw data is **not included** in this repository, in line with the Eskom Data Portal's terms of use.

## How to get it

1. Open the [Eskom Data Portal data request form](https://www.eskom.co.za/dataportal/data-request-form/).
2. **Demand Side:** choose *All items*.
3. **Supply Side (Conventional):** choose *Selection* and tick **ILS Usage** and **Manual Load_Reduction(MLR)**.
4. **Date range:** 1 April 2022 to the latest available date.
5. Submit. Eskom emails a download link.
6. Save the file in this folder and set `FILE` at the top of `main.py` to its name (e.g. `data/ESK20096.csv`).

Running `main.py` then creates `data/demand_clean.csv`, which `explore.py` and `model.py` use.
