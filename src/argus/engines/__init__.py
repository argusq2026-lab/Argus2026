"""Inference backends.

Deliberately empty of re-exports. `argus.vision` imports `argus.engines.base`
for the runner protocol while `argus.engines.mock` imports
`argus.vision.blazepose` for the anchor layout, so any eager import in either
package's ``__init__`` would close that loop into a circular import. Import the
submodule you need — `argus.engines.factory` is the entry point.
"""
