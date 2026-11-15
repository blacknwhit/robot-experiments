"""Disjoint episode ownership across independent torchrun jobs."""


def assigned_episodes(episodes, rank, world, shard_index=0, num_shards=1):
    if world < 1 or num_shards < 1 or not 0 <= rank < world or not 0 <= shard_index < num_shards:
        raise ValueError('Invalid rank/world or shard-index/num-shards')
    global_rank = shard_index * world + rank
    return episodes[global_rank::num_shards * world]
