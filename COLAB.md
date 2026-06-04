# Running Tambour on Google Colab

```python
# 1. clone + install (registers the `tambour` command and pulls missing deps)
!git clone https://github.com/Lourdhu02/tambour.git
%cd tambour
!pip install -q -e .

# 2. verify all 4 variants build / train / export
!tambour sweep

# 3. train  (use %cd above, NOT !cd — see notes)
!tambour train --model tambour-s --data /content/analog_data --device cuda --batch 32
```

## Gotchas

- **Use `%cd tambour`, not `!cd tambour`.** `!cd` runs in a throwaway subshell and does
  not change the notebook's working directory, so the next cell still runs from
  `/content`, where `python -m tambour` resolves the *repo folder* (no `__main__`)
  instead of the package. `%cd` (a magic) persists. `pip install -e .` sidesteps it
  entirely by giving you the `tambour` command on PATH.

- **Small dataset → smaller `--batch`.** With only a few hundred images, the default
  `--batch 128` is ~5 optimizer steps per epoch and the model barely moves. Use
  `--batch 32` (or less) for quick experiments. For the full 2-7M-image set, keep
  128+ and raise `workers` in the config.

- **fp16 GPUs (T4/V100) are fine.** They have no bf16, so AMP runs in fp16; the
  recognizer forces the CTC log-softmax to fp32 internally, so training stays stable.

- **Multi-GPU:**
  ```python
  !torchrun --nproc_per_node=2 -m tambour train --model tambour-b --data DATA --device cuda
  ```

- **Calibrate + abstain after training** (for a trustworthy production read):
  ```python
  !tambour calibrate --ckpt runs/exp/best.pth --data /content/analog_data --precision 0.995
  !tambour predict   --ckpt runs/exp/best.pth --source /content/test_imgs --tau <tau-from-calibrate>
  ```
