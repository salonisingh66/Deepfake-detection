# DeepShield AI: Automated Testing Documentation

## Test Suite Architecture

The DeepShield AI automated test suite is built using `pytest` and structured into 8 modular test suites:

| Test File | Target Subsystem | Tests Executed |
| :--- | :--- | :--- |
| `tests/test_validation.py` | Input validation & file security | 7 |
| `tests/test_hashing.py` | SHA-256 and MD5 cryptographic integrity | 3 |
| `tests/test_database.py` | SQLite analysis audit log CRUD | 4 |
| `tests/test_aggregation.py` | Statistical video score aggregation | 5 |
| `tests/test_image_pipeline.py` | End-to-end image forensic pipeline | 4 |
| `tests/test_video_pipeline.py` | Video frame streaming & inspection | 3 |
| `tests/test_reports.py` | ReportLab forensic PDF generation | 2 |
| `tests/test_api.py` | FastAPI REST endpoints & HTTP codes | 6 |
| **Total** | | **34 Automated Tests** |

## Running Tests

Execute the full test suite with verbose reporting:

```bash
pytest -v
```

Run tests with execution timing summary:

```bash
pytest -v --durations=10
```

Run a specific test module:

```bash
pytest tests/test_validation.py -v
```

## Synthetic Media Policy
In accordance with forensic engineering principles:
- Unit tests use synthetic sample files and public domain NASA images (`astronaut`, `rocket`) to verify pipeline functionality without network downloads.
- Synthetic media is explicitly labeled as testing assets and never presented as genuine deepfake benchmarks.
