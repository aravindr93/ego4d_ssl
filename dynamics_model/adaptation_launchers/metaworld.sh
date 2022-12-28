# wandb login --host=https://fairwandb.org/ --relogin

# Start with MoCo checkpoint trained on Ego4D (released model)
PYTHONPATH=. python main_launcher.py environment.slurm=True \
    logging.wandb_project="inverse_dynamics_adapt_rep" logging.name="inverse_dynamics_metaworld" \
    environment.ngpu=8 environment.world_size=1 \
    model.embedding='moco' \
    data.suite='Metaworld' 'data.envs=["assembly", "bin-picking", "button-press-topdown", "drawer-open", "hammer"]' \
    data.data_dir='/checkpoint/aravraj/pvr_project_data/datasets/' data.suite.prop_key='gripper_proprio'