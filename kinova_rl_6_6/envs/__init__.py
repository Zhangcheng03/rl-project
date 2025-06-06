from gymnasium.envs.registration import register


register(
    id="KinovaReach-v0",
    entry_point="envs.kinova_reach_env:KinovaReachEnv"
)