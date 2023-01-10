# wandb login --host=https://fairwandb.org/ --relogin

# Start with MoCo checkpoint trained on Ego4D (released model)
PYTHONPATH=. python main_launcher.py environment.slurm=False \
    dynamics=inverse \
    logging.wandb_project="inverse_dynamics_adapt_rep" logging.name="inverse_dynamics_dmc" \
    environment.ngpu=1 environment.world_size=1 \
    model.embedding=moco_vit \
    data.pickle_dir="/home/aryanjain/data/expert_data/" \
    data.frames_dir="/shared/aryanjain/data/expert_data/" \
    data/suite=DMC 'data.envs=["walker_walk", "walker_stand", "cheetah_run", "finger_spin", "reacher_easy"]' 