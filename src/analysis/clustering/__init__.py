"""Descubrimiento de reglas de combinación por clustering de vectores de relación.

Codifica cada tripleta (a, b) -> c como r = emb(c) - (emb(a)+emb(b))/2 y clusteriza
esos vectores (DIFFVEC / SeVeN). Ver FINDINGS.md.
"""
