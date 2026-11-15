"""Coverage, exclusivity, resume ownership and compatibility of job partitioning."""
import unittest
from cache_partition import assigned_episodes


class PartitionTest(unittest.TestCase):
    def test_five_two_gpu_jobs_cover_every_episode_once(self):
        for length in (3,30000,30007):
            episodes=list(range(length))
            owners=[assigned_episodes(episodes,rank,2,shard,5) for shard in range(5) for rank in range(2)]
            self.assertEqual(sorted(e for group in owners for e in group),episodes)
            self.assertLessEqual(max(map(len,owners))-min(map(len,owners)),1)
            if length==30000:
                self.assertEqual([len(group) for group in owners],[3000]*10)
                for group in owners:
                    self.assertEqual({task:sum(e//100==task for e in group) for task in range(300)},dict.fromkeys(range(300),10))

    def test_resume_reuses_completed_work_without_changing_ownership(self):
        episodes=list(range(301));done={1,5,12,20,87,299}
        pending=[]
        for shard in range(5):
            for rank in range(2):
                assigned=assigned_episodes(episodes,rank,2,shard,5)
                self.assertEqual(assigned,assigned_episodes(episodes,rank,2,shard,5))
                pending.extend(e for e in assigned if e not in done)
        self.assertEqual(sorted(pending),[e for e in episodes if e not in done])

    def test_default_preserves_original_single_job_partition(self):
        episodes=list(range(103))
        for rank in range(4):self.assertEqual(assigned_episodes(episodes,rank,4),episodes[rank::4])

    def test_invalid_topology_is_rejected(self):
        for args in [(0,0,0,5),(2,2,0,5),(0,2,5,5),(0,2,-1,5),(0,2,0,0)]:
            with self.assertRaises(ValueError):assigned_episodes([], *args)


if __name__=='__main__':unittest.main()
