wandb login --host=https://fairwandb.org/ --relogin

# Test (unit test)
PYTHONPATH=. python main_moco.py environment.slurm=False
        environment.world_size=1 environment.ngpu=2 \
        model.arch=vit_small optim.batch_size=128 optim.epochs=5 \
	logging.name=moco_v3_test_local logging.wandb_project=moco_v3_test \
        data.train_filelist=datasets/ego4d_tiny.txt

PYTHONPATH=. python main_moco.py environment.slurm=True 
        environment.world_size=1 environment.ngpu=2 \
        model.arch=vit_small optim.batch_size=128 optim.epochs=5 \
        logging.name=moco_v3_test_slurm logging.wandb_project=moco_v3_test \
        data.train_filelist=datasets/ego4d_tiny.txt

