from gymnasium.envs.registration import register

register(
    id="airhockeyenv/MoveToTarget-v0",
    entry_point="airhockeyenv.airhockeyenv:AirHockey2D",
    max_episode_steps=1000
)
