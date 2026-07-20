# Artifacts

This folder can contain small local artifacts needed for the submission
(e.g. a small fitted parameter file, a tiny lookup table).

Do not commit:

- large checkpoints
- private keys
- API tokens
- raw private data

Large model artifacts (e.g. HSSM-fit traces, HGF posterior checkpoints) should
be referenced through Hugging Face checkpoint paths recorded in
`submission.yaml` (`artifacts.hf_checkpoint`) or `config.yaml`, and kept out of
git so this repository stays small.