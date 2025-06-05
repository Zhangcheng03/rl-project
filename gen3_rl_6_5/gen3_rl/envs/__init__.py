from gymnasium.envs.registration import register

register(
    id="KinovaTracking-v0",
    entry_point="envs.gen3_tracking_env:KinovaTrackingEnv",
    max_episode_steps=2000
)

register(
    id="KinovaReach-v0",
    entry_point="envs.gen3_reach_env:KinovaReachEnv",
    max_episode_steps=2000
)