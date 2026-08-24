"""PDF decoding, split out of the single 10k-line parser module.

Each module here owns one cohesive concern. ``orbitbrief_pdf`` re-exports
everything, so it remains the single public entry point and no existing
import anywhere in the codebase had to change.
"""
