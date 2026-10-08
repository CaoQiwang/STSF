# VisiFold on PEMS04, PEMS08, and PEMS-BAY

This directory adapts the authors' [VisiFold implementation](https://github.com/PlanckChang/VisiFold) while retaining its original PEMS04/PEMS08 experiment settings and adding PEMS-BAY. The model follows the paper's temporal folding graph, embedding fusion, node-level masking, random subgraph sampling, Transformer encoder, MLP prediction head, and Huber objective.

## Data protocol

- PEMS04 and PEMS08: the authors' released `data.npz` files, containing flow, normalized time-of-day, and day-of-week. They are stored alongside, rather than replacing, the repository's original three-signal NPZ files.
- PEMS-BAY: traffic speed from all 325 sensors; time features are generated from its timestamps.
- Frequency: 5 minutes, or 288 observations per day.
- Split: chronological 60%/20%/20%, matching the VisiFold paper.
- Normalization: global z-score using only the training split.
- Inputs: standardized flow/speed plus time-of-day and day-of-week features.
- Topology: `adj_mx_bay.pkl` is intentionally unused. VisiFold learns node embeddings and does not use graph topology as a hard prior.
- Forecast tasks: 24, 36, and 48 history/forecast steps (2, 3, and 4 hours).

The original PEMS04/PEMS08 presets are recorded in [`model/original_config.yaml`](model/original_config.yaml) and selected automatically from the dataset and input horizon. These preserve the effective settings in official commit `cc035f7`, including its PEMS08-24 mask ratio of 0.3 and PEMS08-48 embedding dimension of 64. The paper's summary table reports 0.2 and 32 respectively; the released-code values are used for code-level reproducibility.

The paper does not report PEMS-BAY hyperparameters. Its defaults transfer the PEMS04 settings because PEMS-BAY (325 nodes) is closest in node count to PEMS04 (307 nodes).

## Run

From `base/Visifold`, reproduce one original PEMS04 experiment with:

```bash
python -m pip install -r requirements.txt
python script/train.py --dataset PEMS04 --in-steps 24 --out-steps 24
```

PEMS08 and PEMS-BAY examples:

```bash
python script/train.py --dataset PEMS08 --in-steps 24 --out-steps 24
python script/train.py --dataset PEMS-BAY --in-steps 24 --out-steps 24
```

Run all six original PEMS04/PEMS08 experiments with:

```bash
bash script/train.sh
```

Useful overrides include `--device cuda`, `--batch-size`, `--embedding-dim`, `--mask-ratio`, and `--subgraph-size`. On Windows, `--num-workers 0` is the safe default. Checkpoints are written to `base/Visifold/output/`.

Official file checksums:

| File | SHA-256 |
| --- | --- |
| `data/PEMS04/data.npz` | `6E46317AA8C87AC307FC5DF9EB2A01E6D87497FBC7E881C96B21BABBF171647E` |
| `data/PEMS08/data.npz` | `709383C68DD591F52A84117BB7C54D5C7FB55ACD05155C7CCAEF83A913A77CD0` |

## Implementation notes

During training, one random node subset is retained for the whole mini-batch, as in the released implementation. Each sample then receives its own random node permutation before tokens are split into fixed-size subgraphs. The same retained-node indices select the training targets. During validation and testing, node visibility is disabled and the full graph is passed to self-attention.

The encoder retains the post-normalization order of the authors' released code. This differs from the pre-normalization notation in Equations 6-7 of the paper and is intentional for code-level reproducibility.

## Sources

- Z. Zhang et al., [VisiFold: Long-Term Traffic Forecasting via Temporal Folding Graph and Node Visibility](https://arxiv.org/abs/2603.11816), 2026.
- [Official code](https://github.com/PlanckChang/VisiFold), commit `cc035f7` used as the adaptation reference.
