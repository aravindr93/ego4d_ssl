# wandb login --host=https://fairwandb.org/ --relogin

# Start with MoCo checkpoint trained on Ego4D (released model)
PYTHONPATH=. python main_launcher.py environment.slurm=False \
    dynamics='inverse' \
    logging.wandb_project="inverse_dynamics_adapt_rep" logging.name="inverse_dynamics_dmc" \
    environment.ngpu=8 environment.world_size=1 \
    model.embedding='moco' \
    data.suite='DMC' data.envs='walker_walk' data.data_dir='/checkpoint/aravraj/pvr_project_data/datasets/'
