# Train on DMControl dataset (1M frames)
# PYTHONPATH=. python main_moco.py environment.slurm=True logging.name=dmcontrol-5tasks environment.ngpu=8 environment.world_size=2 data.train_filelist=/checkpoint/nihansen/data/tdmpc2/dmcontrol_5tasks.txt

# Train on Meta-World dataset (1M frames)
# PYTHONPATH=. python main_moco.py environment.slurm=True logging.name=metaworld-5tasks environment.ngpu=8 environment.world_size=2 data.train_filelist=/checkpoint/nihansen/data/tdmpc2/metaworld_5tasks.txt

# Finetune on DMControl dataset (1M frames)
# PYTHONPATH=. python main_moco.py environment.slurm=True logging.name=ego4d_walker_finetune environment.ngpu=8 optim.lr=0.001 environment.world_size=2 environment.load_path=/checkpoint/nihansen/data/tdmpc2/encoders/moco_v2_15ep_pretrain_ego4d.pth.tar
