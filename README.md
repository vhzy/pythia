

Pythia is a modular framework for vision and language multimodal research. Built on top
of PyTorch, it features:


# run cig model
python -m torch.distributed.launch --nproc_per_node 4 tools/run.py --tasks vqa --datasets m4c_textvqa --model cig --config configs/vqa/cig_textvqa/cig.yml --save_dir save/cig training_parameters.distributed True


## License

Pythia is licensed under BSD license available in [LICENSE](LICENSE) file
