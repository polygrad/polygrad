# Polygrad Native Fuzz Seeds

These are curated seed inputs for native libFuzzer harnesses. Makefile fuzz
targets use these tracked seeds as read-only inputs and write generated corpus
units into the ignored `temp/fuzz-corpus/` work directory.

Keep seeds small, reproducible, and focused on compiler/runtime boundaries. Do
not copy proprietary or third-party corpora here.
