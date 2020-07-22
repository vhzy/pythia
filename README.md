

Pythia is a modular framework for vision and language multimodal research. Built on top
of PyTorch, it features:


# run cig model
1) to train the M4C model on the TextVQA training set:
```
# Distributed Data Parallel (on a 4-GPU machine)
# (change `--nproc_per_node 4` to the actual GPU number on your machine)
python -m torch.distributed.launch --nproc_per_node 4 tools/run.py --tasks vqa --datasets m4c_textvqa --model cig \
--config configs/vqa/cig_textvqa/cig.yml \
--save_dir save/cig \
training_parameters.distributed True

# alternative: Data Parallel (slower, but results should be the same)
python tools/run.py --tasks vqa --datasets m4c_textvqa --model cig \
--config configs/vqa/cig_textvqa/cig.yml \
--save_dir save/cig \
training_parameters.data_parallel True
```
(Replace `cig_textvqa` with other datasets and `configs/vqa/cig_textvqa/cig.yml` with other config files to train with other datasets and configurations. See the table above. You can also specify a different path to `--save_dir` to save to a location you prefer.)

2) to evaluate the pretrained CIG model locally on the TextVQA validation set (assuming the pretrained model is downloaded to `data/models/cig_textvqa_m4c.ckpt`):
```
python tools/run.py --tasks vqa --datasets m4c_textvqa --model m4c \
--config configs/vqa/m4c_textvqa/m4c.yml \
--save_dir save/m4c \
--run_type val \
--resume_file data/models/m4c_textvqa_m4c.ckpt
```
(Note: use `--resume 1` instead of `--resume_file data/models/m4c_textvqa_m4c.ckpt` to evaluated your trained snapshots.)

3) to generate the EvalAI prediction files for the TextVQA test set (assuming the pretrained model is downloaded to `data/models/m4c_textvqa_m4c.ckpt`):
```
python tools/run.py --tasks vqa --datasets m4c_textvqa --model cig \
--config configs/vqa/cig_textvqa/cig.yml \
--save_dir save/cig \
--run_type inference --evalai_inference 1 \
--resume_file data/models/m4c_textvqa_cig.ckpt
```
(Note: use `--resume 1` instead of `--resume_file data/models/m4c_textvqa_m4c.ckpt` to evaluated your trained snapshots. For running inference on val set, use `--run_type val` and rest of the arguments remain same.)

1) to train the CIG model on the TextVQA training set:
```
# Distributed Data Parallel (on a 4-GPU machine)
# (change `--nproc_per_node 4` to the actual GPU number on your machine)
python -m torch.distributed.launch --nproc_per_node 4 tools/run.py --tasks vqa --datasets m4c_textvqa --model m4c \
--config configs/vqa/m4c_textvqa/m4c.yml \
--save_dir save/m4c \
training_parameters.distributed True

# alternative: Data Parallel (slower, but results should be the same)
python tools/run.py --tasks vqa --datasets m4c_textvqa --model m4c \
--config configs/vqa/m4c_textvqa/m4c.yml \
--save_dir save/m4c \
training_parameters.data_parallel True
```
(Replace `m4c_textvqa` with other datasets and `configs/vqa/m4c_textvqa/m4c.yml` with other config files to train with other datasets and configurations. See the table above. You can also specify a different path to `--save_dir` to save to a location you prefer.)

2) to evaluate the pretrained M4C model locally on the TextVQA validation set (assuming the pretrained model is downloaded to `data/models/m4c_textvqa_m4c.ckpt`):
```
python tools/run.py --tasks vqa --datasets m4c_textvqa --model m4c \
--config configs/vqa/m4c_textvqa/m4c.yml \
--save_dir save/m4c \
--run_type val \
--resume_file data/models/m4c_textvqa_m4c.ckpt
```
(Note: use `--resume 1` instead of `--resume_file data/models/m4c_textvqa_m4c.ckpt` to evaluated your trained snapshots.)

3) to generate the EvalAI prediction files for the TextVQA test set (assuming the pretrained model is downloaded to `data/models/m4c_textvqa_m4c.ckpt`):
```
python tools/run.py --tasks vqa --datasets m4c_textvqa --model m4c \
--config configs/vqa/m4c_textvqa/m4c.yml \
--save_dir save/m4c \
--run_type inference --evalai_inference 1 \
--resume_file data/models/m4c_textvqa_m4c.ckpt
```
(Note: use `--resume 1` instead of `--resume_file data/models/m4c_textvqa_m4c.ckpt` to evaluated your trained snapshots. For running inference on val set, use `--run_type val` and rest of the arguments remain same.)


## License

Pythia is licensed under BSD license available in [LICENSE](LICENSE) file
