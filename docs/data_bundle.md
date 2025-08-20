# Data Bundle Format

A `DataBundle` groups time-aligned market and alternative data used by the
library.  All contained :class:`pandas.DataFrame` instances must share the same
`DatetimeIndex`.

For **multi‑asset** bundles the columns should use a two‑level
:class:`pandas.MultiIndex` with the first level being the asset identifier and
the second level the field name.  A price DataFrame for two tickers might look
like:

```
index               AAPL                  MSFT
                     open  close         open  close
2024‑01‑01            ...   ...           ...   ...
2024‑01‑02            ...   ...           ...   ...
```

Single‑asset bundles may keep a flat column index with only field names.  Any
optional publication timestamps must mirror the shape of the corresponding data
frames and must not be later than the data's timestamp.
