"""Small-scale checks; these do not establish paper-scale stationarity."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from model import advance, penalty, seed_events, state_index
from run import simulate


class ModelTests(unittest.TestCase):
    def test_pressure_rules(self):
        self.assertAlmostEqual(penalty(0, 3, 3, .1, False), .3)
        self.assertEqual(penalty(1, 3, 3, .1, False), 0)
        self.assertAlmostEqual(penalty(1, 2, 3, .1, False), .3)
        self.assertEqual(penalty(0, 2, 3, .1, False), 0)
        self.assertAlmostEqual(penalty(1, 4, 3, .1, True), .1)
        self.assertAlmostEqual(penalty(0, 1, 3, .1, True), .1)

    def test_states(self):
        self.assertEqual([state_index(a, g, 0) for a, g in [(1, True), (1, False), (0, True), (0, False)]], [0, 1, 2, 3])

    def test_chunks_and_event_accounting(self):
        initial = np.random.default_rng(7).integers(0, 2, (5, 5), dtype=np.int8)
        outputs = []
        for chunks in ([12], [4, 8]):
            actions = initial.copy()
            q = np.zeros((5, 5, 4, 2))
            seed_events(91)
            rows = [advance(actions, q, 3.5, 3, .1, .1, .9, .01, False, 0, n) for n in chunks]
            rho = np.concatenate([row[0] for row in rows])
            counts = np.concatenate([row[1] for row in rows])
            advantages = np.concatenate([row[2] for row in rows])
            np.testing.assert_array_equal(counts.sum(axis=1), 5 * 25)
            self.assertTrue(np.all((rho >= 0) & (rho <= 1)))
            outputs.append((actions, q, rho, counts, advantages))
        for a, b in zip(*outputs):
            np.testing.assert_array_equal(a, b)

    def test_state_variants_and_zero_pressure(self):
        initial = np.random.default_rng(7).integers(0, 2, (5, 5), dtype=np.int8)
        for representation in range(3):
            outcomes = []
            for independent in (False, True):
                actions = initial.copy()
                q = np.zeros((5, 5, 4 if representation == 0 else 2, 2))
                seed_events(19)
                advance(actions, q, 3.5, 3, 0., .1, .9, .01, independent, representation, 8)
                outcomes.append((actions, q))
            for a, b in zip(*outcomes):
                np.testing.assert_array_equal(a, b)

    def test_output_and_exact_tail_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'run'
            summary = simulate(dict(length=5, mcs=20, tail=7, progress_every=6, snapshots='0,4,20'), folder)
            data = np.genfromtxt(folder / 'trajectory.csv', delimiter=',', names=True)
            self.assertAlmostEqual(summary['mean_cooperation'], data['rho_C'][-7:].mean())
            self.assertTrue((folder / 'actions_mcs_4.npy').exists())
            with self.assertRaises(FileExistsError):
                simulate(dict(length=5, mcs=20, tail=7), folder)


if __name__ == '__main__':
    unittest.main()
