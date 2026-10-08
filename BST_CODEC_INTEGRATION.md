# Replacing the prototype BST codec later

Step 8.3 deliberately isolates BST-specific code.

## Do not modify

A verified BST-ID implementation should **not** require changes to:

```text
representations/step83/adaptive_core.py
benchmark/step83_benchmark.py
realdata/
```

## Replace / add only

Implement a new codec in `bst_id/codec.py` (or a sibling module) satisfying `SpatialBlockCodec`:

```python
class VerifiedBSTIDCodec:
    name = "verified_bst_id"

    def encode_block(self, block, spec):
        ...

    def spatial_table_bytes(self, blocks, spec):
        ...
```

Then construct:

```python
BSTCompactMap(core, VerifiedBSTIDCodec())
```

All adaptive blocks, temporal signatures, bucket indexing, native queries, decompression, correctness gates, and benchmark workloads remain unchanged.

This separation is important scientifically: it lets the paper distinguish

```text
adaptive mapping algorithm
```

from

```text
BST-ID spatial encoding
```

without rewriting either experiment.
