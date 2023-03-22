# wandb login --host=https://fairwandb.org/ --relogin

# Start with MoCo checkpoint trained on Ego4D (released model)
PYTHONPATH=. python main_launcher.py environment.slurm=False \
    dynamics=inverse \
    logging.wandb_project="inverse_dynamics_adapt_rep" logging.name="inverse_dynamics_rlbench_moco" \
    environment.ngpu=1 environment.world_size=1 \
    model.embedding=moco \
    data.pickle_dir="/home/aryanjain/data/expert_data/" \
    data.frames_dir="/shared/aryanjain/data/expert_data/" \
    data/suite=RLBench 'data.envs=["reach_target", "pick_up_cup", "take_lid_off_saucepan", "take_umbrella_out_of_umbrella_stand"]' \
    optim.batch_size=32