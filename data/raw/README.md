# Raw data

The original UCI archive is licensed CC BY 4.0. It is intentionally excluded
from Git and must be acquired separately before running the audit, experiment,
or independent verifier. Retrieve the official archive into this directory:

```sh
curl --fail --location \
  --output data/raw/uci_default_credit_card_350.zip \
  'https://archive.ics.uci.edu/static/public/350/default%2Bof%2Bcredit%2Bcard%2Bclients.zip'
shasum -a 256 data/raw/uci_default_credit_card_350.zip
unzip -t data/raw/uci_default_credit_card_350.zip
```

Expected SHA-256:
`56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`.
The archive should be 5,539,494 bytes and contain the original workbook
`default of credit card clients.xls`. Do not extract, rename, or modify the
archive; the project reads the workbook directly from the ZIP. The source,
license, retrieval date, size, and hash are recorded in `docs/dataset_audit.md`
and `results/audit/audit.json`.
