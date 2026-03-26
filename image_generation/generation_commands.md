# Image Generation Commands

Each setting uses 6 workers. Running 10 datasets concurrently uses 60 workers, which fits within the 64 CPUs on deeplearning-01.stat.lsa.umich.edu.

## Single dataset

```bash
python generate_images.py --config setting1.yaml --dataset-id 1
```

## All 10 datasets for one setting

```bash
for d in $(seq 1 10); do
  nohup python -u generate_images.py \
    --config setting1.yaml --dataset-id $d \
    &> generate_setting1_d${d}.out &
done
```

## All 5 settings (one setting at a time, 10 datasets in parallel)

```bash
for s in $(seq 1 5); do
  for d in $(seq 1 10); do
    nohup python generate_images.py \
      --config setting${s}.yaml --dataset-id $d \
      > generate_s${s}_d${d}.out 2>&1 &
  done
  wait  # wait for all 10 datasets to finish before starting next setting
done
```

## Output structure

Each run produces 5,000 `.pt` files in its own folder:

```
/nfs/turbo/lsa-regier/scratch/descwl/
├── setting1_1/    (5000 images)
├── setting1_2/
├── ...
├── setting1_10/
├── setting2_1/
├── ...
└── setting5_10/
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--config` | `sim_config.yaml` | Path to setting YAML file |
| `--dataset-id` | None | Dataset ID (1 to num-datasets). Each gets a unique seed and output folder |
| `--num-datasets` | 10 | Total number of datasets per setting |
