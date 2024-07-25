from gym.envs.registration import register

register(
    id='FetchPickAndPlaceCustom-v0',
    entry_point='goal_prox.envs.fetch.custom_fetch:FetchPickAndPlaceCustom',
    max_episode_steps=50,
    )

register(
    id='FetchPickAndPlaceHarder-v0',
    entry_point='goal_prox.envs.fetch.custom_fetch:FetchPickAndPlaceHarder',
    max_episode_steps=50,
    )

register(
    id='FetchPickAndPlaceDiff-v0',
    entry_point='goal_prox.envs.fetch.custom_fetch:FetchPickAndPlaceDiff',
    max_episode_steps=50,
    )

#50
register(
    id='FetchPickAndPlaceDiffHoldout-v0',
    entry_point='goal_prox.envs.fetch.custom_fetch:FetchPickAndPlaceDiffHoldout',
    max_episode_steps=50,
    )


register(
    id='FetchViz-v0',
    entry_point='goal_prox.envs.fetch.custom_fetch:FetchViz',
    max_episode_steps=50,
    )

# TODO: 50
register(
    id='FetchPushEnvCustom-v0',
    entry_point='goal_prox.envs.fetch.custom_push:FetchPushEnvCustom',
    max_episode_steps=60,
    )

register(
    id='FetchDebugPushEnv-v0',
    entry_point='goal_prox.envs.fetch.custom_push:FetchDebugPushEnv',
    max_episode_steps=60,
    )


