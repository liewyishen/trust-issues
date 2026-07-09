"""
FastAPI adapter over src/explain.py.

This package adds an HTTP boundary and nothing else. It does not score, it does
not calibrate, and it does not encode features -- every one of those already
exists in src/ and is reached THROUGH explain_applicants(), not reimplemented
beside it. A second, independently-written encoding path is exactly the
train/serve skew calibrate.py's module docstring warns about; a serving package
is where that skew is most likely to be introduced and least likely to be
noticed.

Four modules:

  config.py     the one number that is not packaged with the model
  schema.py     the request contract (NOT LOAN_SCHEMA) and the response
  artifacts.py  the fail-closed startup load
  app.py        lifespan, GET /healthz, POST /score
"""
