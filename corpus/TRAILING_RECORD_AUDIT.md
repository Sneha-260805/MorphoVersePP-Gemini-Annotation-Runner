# Trailing unlabeled record audit

4 poem-like row(s) were found immediately after the canonical 1,570-record corpus boundary, each missing a required `Language` value. They are preserved here for audit only — never assigned an MV++ ID, never included in the source-only export, and never included in any teammate assignment.

| Excel row | Title | Translated title | Poet | Script observation | Reason excluded |
|---|---|---|---|---|---|
| 1573 | Chand Ka Munh | The Moon's Face | Suryakant Tripathi 'Nirala' | non-Latin script present | Outside canonical 1,570-record boundary and missing required Language field. |
| 1574 | Nadiya | The River | Biharilal Chakbast | non-Latin script present | Outside canonical 1,570-record boundary and missing required Language field. |
| 1575 | Sankalp | Resolution | Kumar Vishwas | non-Latin script present | Outside canonical 1,570-record boundary and missing required Language field. |
| 1576 | Choti Si Baat | A Small Matter | Gulzar | non-Latin script present | Outside canonical 1,570-record boundary and missing required Language field. |

## Embedded/anomalous rows excluded from the canonical count

Rows whose `Language` cell is non-blank but is not one of the 21 supported language names — this is how a mid-sheet duplicate header row is discovered programmatically, without hardcoding a row number.

| Excel row | Observed language-column value | Col B | Col C |
|---|---|---|---|
| 1210 | Original Language | Original Title | English Title |
