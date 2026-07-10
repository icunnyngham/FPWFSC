# tokyo_drift model checkpoints

Trained-NN weights for the `model` (Tokyo Drift) predictor live here, one
subdirectory per mode:

```
checkpoints/
├── README.md                                   (committed)
├── vampires_f760_10zern/974j9jqt_torch.pt      (gitignored)
├── vampires_f750_35zern/CHP143_torch.pt        (gitignored)
├── vampires_vvc_f750_35zern/CB9WJU_torch.pt    (gitignored)
└── vampires_vvc_f750_35zern_crop/CKP8EJ_torch.pt (gitignored)
```

Why a central `checkpoints/` tree (not a file beside each mode's config):
the whole directory bundles/ships as one unit, independently of the repo.

## Not in git

The `.pt` files are large (~575 MB each) and **gitignored** (`*.pt`, via
`fpwfsc/tokyo_drift/.gitignore`). Only this README is committed. A mode's
`manifest.yaml` names its checkpoint file; `mode_registry.checkpoint_path`
resolves a bare name to `checkpoints/<mode>/<file>` (an absolute path is
honored as-is), and raises a clear error if the file is absent.

To install locally, either drop the real `.pt` into the mode's subdir, or
symlink it (what the dev setup does, to avoid duplicating the weights):

```bash
cd fpwfsc/tokyo_drift/checkpoints/vampires_f760_10zern
ln -s /path/to/974j9jqt_torch.pt 974j9jqt_torch.pt
```

To bundle for transfer, zip the whole `checkpoints/` tree dereferencing
symlinks (e.g. `zip -r --symlinks` off, or `tar -hczf`), so the archive
carries the real weights.

## Provenance

| mode | file | run id | filter | modes | optics | source | torch↔keras parity |
|---|---|---|---|---|---|---|---|
| `vampires_f760_10zern` | `974j9jqt_torch.pt` | `974j9jqt` (2023-10) | F760 | 10 | no coro, 128px | `.h5`/`.keras`/`.tf` | `final_mag` max-abs 7e-7, 100% sign |
| `vampires_f750_35zern` | `CHP143_torch.pt` | `CHP143~1` (2024-04) | F750 | 35 | no coro, 128px | `.tf` (TF 2.14) | `final_mag` max-abs 7e-7, 100% sign |
| `vampires_vvc_f750_35zern` | `CB9WJU_torch.pt` | `CB9WJU~4` (2024-05) | F750 | 35 | VVC charge-4, 128px | `.tf` (TF 2.14) | `final_mag` max-abs 8e-7, 100% sign |
| `vampires_vvc_f750_35zern_crop` | `CKP8EJ_torch.pt` | `CKP8EJ~6` (2024-06) | F750 | 35 | VVC charge-4, 120px | `.tf` (TF 2.14) | `final_mag` max-abs 8e-7, 100% sign |

All four are the same `FFModel` family (see `../model_torch.py`); the `.pt`
is self-describing (`meta` carries the full architecture — including
`input_hw`, 120 for the crop model — plus this provenance). The VVC models
put the coronagraph in the optical sim, not the network.
The full conversion + closed-loop validation lives in the model-side
project (`external/tokyo_drift_training/`); the integration into this
pipeline is documented in `../MODEL_INTEGRATION_NOTES.md`.
