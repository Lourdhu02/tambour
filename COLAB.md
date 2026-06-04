# Running Tambour on Google Colab

```python
# 1. clone + install (registers the `tambour` command and pulls missing deps)
!git clone https://github.com/Lourdhu02/tambour.git
%cd tambour
!pip install -q -e .

# 2. verify all 4 variants build / train / export
!tambour sweep

# 3a. train on the FULL 2-7M-image set — just use the defaults
!tambour train --model tambour-b --data /content/analog_data --device cuda

# 3b. ...or on the tiny ~800-image sample, train long with a small batch
!tambour train --model tambour-b --data /content/analog_data --device cuda \
    --batch 16 --epochs 300 --img-w 192
```

## Gotchas

- **Use `%cd tambour`, not `!cd tambour`.** `!cd` runs in a throwaway subshell and does
  not change the notebook's working directory, so the next cell still runs from
  `/content`, where `python -m tambour` resolves the *repo folder* (no `__main__`)
  instead of the package. `%cd` (a magic) persists. `pip install -e .` sidesteps it
  entirely by giving you the `tambour` command on PATH.

- **From-scratch CTC is data-hungry — expect slow early progress.** `exact` legitimately
  sits at 0 for many epochs while the model first learns to emit the right *number* of
  digits; watch `char` climb first, then `exact` follows. On the **full 2-7M set** one
  epoch is tens of thousands of steps, so it converges fast on defaults. On a **few-hundred-
  image sample** it is heavily undertrained: use `--batch 16 --epochs 300 --img-w 192` and
  give it time. Reported `val`/`test` numbers are the **EMA** model (what you deploy).

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
