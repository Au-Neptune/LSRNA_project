## LSR training internals

Use `scripts/train_lsr.sh` from the repository root. The handoff dataset already
contains paired SDXL latents, so the downloader and preprocessing scripts under
`datasets/scripts/` are retained only for provenance and are not part of the
normal reproduction path.

See the root `README.md` for installation, dataset extraction, validation,
single/multi-GPU training, resume, and final pipeline commands.
