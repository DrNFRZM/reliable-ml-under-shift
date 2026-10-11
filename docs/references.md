# References and data license

These describe existing data/methods. Their published results are not this
repository's results.

- Malinin et al. (2021), *Shifts: A Dataset of Real Distributional Shift Across
  Multiple Large-Scale Tasks*. [Paper](https://arxiv.org/abs/2107.07455),
  [official repository](https://github.com/Shifts-Project/shifts),
  [dataset description](https://shifts.ai/dataset).
- Guo, Pleiss, Sun, and Weinberger (2017), *On Calibration of Modern Neural
  Networks*. [Paper](https://arxiv.org/abs/1706.04599).
- Lakshminarayanan, Pritzel, and Blundell (2017), *Simple and Scalable Predictive
  Uncertainty Estimation using Deep Ensembles*.
  [Paper](https://arxiv.org/abs/1612.01474). This project uses independently
  initialized MLP probability averaging; it does not reproduce every detail of
  that paper's training procedure.
- Angelopoulos and Bates (2021; revised 2022), *A Gentle Introduction to Conformal
  Prediction and Distribution-Free Uncertainty Quantification*.
  [Paper](https://arxiv.org/abs/2107.07511).

The canonical Weather archive is
[canonical-partitioned-dataset.tar](https://storage.yandexcloud.net/yandex-research/shifts/weather/canonical-partitioned-dataset.tar).
Shifts Weather data is separately licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Downloading it accepts those terms according to the official repository. Data
and prepared arrays are excluded from Git. The MIT `LICENSE` applies to this
repository's code, not to the upstream data.
