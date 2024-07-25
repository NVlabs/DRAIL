from gym.envs.registration import register
register(
    id='VizFetchPickAndPlaceCustom-v0',
    entry_point='goal_prox.envs.viz.pick:VizFetchPickAndPlaceEnv',
    max_episode_steps=100,
    )

register(
    id='VizFetchPushEnv-v0',
    entry_point='goal_prox.envs.viz.push:VizFetchPushEnv',
    max_episode_steps=60,
    )


