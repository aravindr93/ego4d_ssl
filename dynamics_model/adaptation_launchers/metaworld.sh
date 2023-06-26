# wandb login --host=https://fairwandb.org/ --relogin

# Start with MoCo checkpoint trained on Ego4D (released model)
PYTHONPATH=. python main_launcher.py environment.slurm=False \
    dynamics=inverse \
    logging.wandb_project="inverse_dynamics_adapt_rep" logging.name="inverse_dynamics_metaworld-mvp" \
    environment.ngpu=1 environment.world_size=1 \
    model.embedding=mvp \
    data.pickle_dir="/home/aryanjain/data/expert_data/" \
    data.frames_dir="/shared/aryanjain/expert_data/" \
    data/suite=Metaworld 'data.envs=["assembly", "bin-picking", "button-press-topdown", "drawer-open", "hammer"]' \
    optim.batch_size=32