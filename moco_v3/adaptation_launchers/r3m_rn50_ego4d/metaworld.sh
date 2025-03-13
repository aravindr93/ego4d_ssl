# wandb login --host=https://fairwandb.org/ --relogin

# Start with R3M checkpoint trained on Ego4D (released model)
PYTHONPATH=. python main_moco.py environment.slurm=True \
    logging.wandb_project="mujoco_adapt_rep" logging.name="moco-v3_rn50_metaworld_adapt_try3" \
    model.arch=resnet50 environment.ngpu=8 environment.world_size=1  \
    data.type="picklepaths" data.frameskip=1 \
    data.train_filelist="/checkpoint/yixinlin/eaif/datasets/metaworld-expert-v0.1/" \
    model.load_path="/checkpoint/yixinlin/eaif/models/r3m/r3m_50/model.pt"