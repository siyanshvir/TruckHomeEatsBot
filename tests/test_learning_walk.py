import copy
import unittest

from game import SPOTS
from learning import Learner, PARTICLES

WALK = {'walk_over': True}


def day(sold, customers=80, walked_in=0, walked_out=0, price=11., meals=60, walk_fields=True):
    info = {'customers': customers,
            'trucks': [{'id': 'r', 'quality': 'standard', 'price': price, 'meals': meals, 'sold': sold}]}
    if walk_fields:
        info.update(walked_in=walked_in, walked_out=walked_out)
    return {'day': 1, 'weather': 'clear', 'spots': {'northres': info}}


def mean_budget(learner):
    post = learner.posterior('northres')
    return sum(p * particle[0] for p, particle in zip(post, PARTICLES))


class WalkLearningTests(unittest.TestCase):
    def test_fewer_local_buyers_at_high_price_lowers_budget_mean(self):
        learner = Learner()
        learner.activate(WALK)
        before = mean_budget(learner)
        for _ in range(4):
            learner.learn(day(sold=10, walked_in=6, walked_out=3, price=11.), features=WALK)
        self.assertEqual(learner.counts['northres'], 4)
        self.assertLess(mean_budget(learner), before - .1)

    def test_walk_ins_are_not_local_evidence(self):
        no_crowd = Learner()
        no_crowd.learn(day(sold=25, customers=0, walked_in=25), features=WALK)
        self.assertEqual(no_crowd.counts['northres'], 0)
        self.assertEqual(no_crowd.weights['northres'], [0.] * len(PARTICLES))
        # Sales that were all walk-ins equal zero local sales.
        a, b = Learner(), Learner()
        a.learn(day(sold=25, walked_in=25), features=WALK)
        b.learn(day(sold=0), features=WALK)
        self.assertEqual(a.weights['northres'], b.weights['northres'])

    def test_missing_walk_fields_withhold_fitting(self):
        learner = Learner()
        learner.learn(day(sold=10, walk_fields=False), features=WALK)
        self.assertEqual(learner.counts['northres'], 0)
        self.assertEqual(learner.weights['northres'], [0.] * len(PARTICLES))
        self.assertEqual(learner.rows, 1)

    def test_legacy_ignores_walk_fields(self):
        a, b = Learner(), Learner()
        a.learn(day(sold=10, walked_in=6, walked_out=2))
        b.learn(day(sold=10, walk_fields=False))
        self.assertEqual(a.weights, b.weights)
        self.assertEqual(a.counts['northres'], 1)

    def test_old_checkpoint_loads_and_walk_bank_is_separate(self):
        legacy = Learner().to_dict()
        for key in ('regimes', 'active_features', 'crowd_variance'):
            legacy.pop(key)
        learner = Learner.from_dict(copy.deepcopy(legacy))
        learner.learn(day(sold=10, walked_in=4), features=WALK)
        self.assertEqual(learner.counts['northres'], 1)
        learner.activate({})
        self.assertEqual(learner.counts['northres'], 0)
        self.assertEqual(set(learner.weights), set(SPOTS))


if __name__ == '__main__':
    unittest.main()
