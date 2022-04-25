./standardizers

Contains three separate but related paradigms for how to create and run
standardizers on curves. The earliest, in `_transformer.py`, contains classes
and functions designed for a primarily Transformer oriented configuration
interface, which are directly fitted to Series or Dataframe objects. The
latter, in `_online.py`, contains functions and classes for assembling these
transformers using statistics over the data generated in an online, out-of-core
manner. These two files are retained for backwards compatibility purposes but
are deprecated.

The most recent paradigm, in `_function.py`, streamlines and repackages the
earlier conglomeration of online statistics and class based transformers.
