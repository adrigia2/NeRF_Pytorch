# legacy/

Contiene l'implementazione custom di NeRF pre-refactor (`nerf_module.py`), archiviata per
riferimento storico. Non è importata da nessun file attivo del progetto.

Sostituita dal package `nerf/` che implementa vanilla bmild/nerf in PyTorch con coarse+fine
hierarchical sampling e depth-hints opzionali per l'indirect irradiance pass.
