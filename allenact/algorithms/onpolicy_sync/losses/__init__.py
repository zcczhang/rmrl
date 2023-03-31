from .a2cacktr import A2C, ACKTR, A2CACKTR
from .ppo import PPO, RNDPPO

losses_map = {"ppo_loss": PPO, "rnd_ppo_loss": RNDPPO}
