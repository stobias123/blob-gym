from gym.envs.registration import register

register(
    id='blob2d-v1',
    entry_point='blob_env.envs.blob_env:BlobEnv')

register(
    id='blob2d-safe-v1',
    entry_point='blob_env.envs.blob_env:BlobEnvNoEnemy')