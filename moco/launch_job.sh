# Test (unit test)
# PYTHONPATH=. python main_moco.py environment.slurm=True logging.name=moco_ego4d_unit_test data.train_filelist=datasets/ego4d_tiny.txt environment.ngpu=8 environment.world_size=2 optim.epochs=2

# Train on 100K frames
# PYTHONPATH=. python main_moco.py environment.slurm=True logging.name=ego4d_100k data.train_filelist=datasets/ego4d_100k.txt environment.ngpu=8 environment.world_size=2

# Train on approx 5 million frames (r3m dataset)
# PYTHONPATH=. python main_moco.py environment.slurm=True logging.name=ego4d_5m data.train_filelist=datasets/ego4d_5m.txt environment.ngpu=8 environment.world_size=2 data.type=standard model.input_sec=1
# PYTHONPATH=. python main_moco.py environment.slurm=True logging.name=ego4d_5m_crop data.train_filelist=datasets/ego4d_5m.txt environment.ngpu=8 environment.world_size=2 data.type=standard data.augmentations=crop model.input_sec=1 

# Train on DMControl dataset (1M frames)
# PYTHONPATH=. python main_moco.py environment.slurm=True logging.name=dmcontrol-walker environment.ngpu=8 environment.world_size=2
# PYTHONPATH=. python main_moco.py environment.slurm=True logging.name=ego4d_walker_finetune environment.ngpu=8 optim.lr=0.001 environment.world_size=2 environment.load_path=/checkpoint/nihansen/data/tdmpc2/encoders/moco_v2_15ep_pretrain_ego4d.pth.tar

# Train on DMControl dataset (1M frames; june 22 experiment)
# PYTHONPATH=. python main_moco.py environment.slurm=True logging.name=dmcontrol-walker-5m data.train_filelist="/checkpoint/nihansen/data/tdmpc2/dmcontrol_5m.txt" environment.ngpu=8 environment.world_size=2 data.type=dmcontrol model.input_sec=3 data.imsize=84

# Train on DMControl dataset (1M frames; input 224x224; seq3)
PYTHONPATH=. python main_moco.py environment.slurm=True logging.name=dmcontrol-walker-seq3-fr environment.ngpu=8 environment.world_size=2 data.type=dmcontrol model.input_sec=3
