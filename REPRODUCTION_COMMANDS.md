# Formal experiment command matrix

Use the Python executable from the `grpo_py311_v2` environment. Every training
entry resumes from `latest.pt` by default.

## Table 1: all-zone experiments

```powershell
python -m repro_scripts.classical --config repro_configs/paper.json --model rand
python -m repro_scripts.classical --config repro_configs/paper.json --model rand_train
python -m repro_scripts.classical --config repro_configs/paper.json --model qrgbm
python -m repro_scripts.train --config repro_configs/paper.json --model wgan
python -m repro_scripts.train --config repro_configs/paper.json --model vae
python -m repro_scripts.train --config repro_configs/paper.json --model nf
python -m repro_scripts.train --config repro_configs/paper.json --model ddpm
python -m repro_scripts.train --config repro_configs/paper.json --model mscadm
```

Run `repro_scripts.generate` for every trained model, then
`repro_scripts.evaluate` and `repro_scripts.tables`.

## Table 2 and Figure 6: independently trained single-zone experiments

Run `repro_scripts.single_zone` for each model. Zone 1 is the declared default
because the article does not identify the selected zone. Aggregate with
`repro_scripts.single_zone_table`.

## Table 3

```powershell
python -m repro_scripts.sampling_steps --config repro_configs/paper.json
```

## Table 4

Run `repro_scripts.ablate` once for each of `full`, `no_ce`, `no_adaln`,
`no_lv`, and `no_rcm`. These are independently trained on the selected zone.

## Figures and Table 5

Run `repro_scripts.figures`, then `repro_scripts.suc`. The SUC command defaults
to seven days because that is what the methods paragraph says. Use `--days 100`
to reproduce the contradictory Table-5 caption claim. The number of K-Means
clusters is undisclosed; the declared default is 10 and can be changed with
`--clusters`.
