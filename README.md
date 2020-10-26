

Pythia is a modular framework for vision and language multimodal research. We fork this framework: https://github.com/ronghanghu/pythia


# run TIG model
1) to train the M4C model on the TextVQA training set:
```
# Distributed Data Parallel (on a 4-GPU machine)
# (change `--nproc_per_node 4` to the actual GPU number on your machine)
python -m torch.distributed.launch --nproc_per_node 4 tools/run.py --tasks vqa --datasets m4c_textvqa --model tig \
--config configs/vqa/tig_textvqa/tig.yml \
--save_dir save/tig \
training_parameters.distributed True

# alternative: Data Parallel (slower, but results should be the same)
python tools/run.py --tasks vqa --datasets m4c_textvqa --model tig \
--config configs/vqa/tig_textvqa/tig.yml \
--save_dir save/tig \
training_parameters.data_parallel True
```
(Replace `tig_textvqa` with other datasets and `configs/vqa/tig_textvqa/tig.yml` with other config files to train with other datasets and configurations. See the table above. You can also specify a different path to `--save_dir` to save to a location you prefer.)

2) to evaluate the pretrained TIG model locally on the TextVQA validation set (assuming the pretrained model is downloaded to `data/models/tig_textvqa_m4c.ckpt`):
```
python tools/run.py --tasks vqa --datasets m4c_textvqa --model tig \
--config configs/vqa/tig_textvqa/tig.yml \
--save_dir save/tig \
--run_type val \
--resume_file data/models/m4c_textvqa_tig.ckpt
```
(Note: use `--resume 1` instead of `--resume_file data/models/m4c_textvqa_m4c.ckpt` to evaluated your trained snapshots.)

3) to generate the EvalAI prediction files for the TextVQA test set (assuming the pretrained model is downloaded to `data/models/m4c_textvqa_tig.ckpt`):
```
python tools/run.py --tasks vqa --datasets m4c_textvqa --model tig \
--config configs/vqa/tig_textvqa/tig.yml \
--save_dir save/tig \
--run_type inference --evalai_inference 1 \
--resume_file data/models/m4c_textvqa_tig.ckpt
```
(Note: use `--resume 1` instead of `--resume_file data/models/m4c_textvqa_tig.ckpt` to evaluated your trained snapshots. For running inference on val set, use `--run_type val` and rest of the arguments remain same.)


## Batch Clean up GPU：
sudo fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh
## Single Clean up GPU:
sudo fuser -v /dev/nvidia4 |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh

## License

Pythia is licensed under BSD license available in [LICENSE](LICENSE) file
