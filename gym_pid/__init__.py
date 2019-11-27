from gym.envs.registration import register

register(
    id='pid-v0',
    entry_point='gym_pid.envs:pidEnv',
)
register(
    id='pid-extrahard-v0',
    entry_point='gym_pid.envs:pidExtraHardEnv',
)
