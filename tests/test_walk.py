import json,math,tempfile,unittest
from pathlib import Path
import numpy as np
from walk_lab.sim import Simulation,INITIAL,LOW,HIGH
from walk_lab.evaluate import walking_success,CONDITIONS
from walk_lab.train import train

class PhysicsTests(unittest.TestCase):
    def setUp(self):self.s=Simulation()
    def test_free_base_is_not_actuated(self):
        self.assertEqual(self.s.model.nu,6)
        self.assertEqual(self.s.model.nq,9)
        self.assertEqual(self.s.model.actuator_trnid[:,0].tolist(),list(range(3,9)))
    def test_real_gravity_and_no_lateral_dof(self):
        self.assertAlmostEqual(self.s.model.opt.gravity[2],-9.81)
        for axis in self.s.model.jnt_axis:
            self.assertEqual(sum(abs(axis)),1)
        self.assertTrue(np.array_equal(self.s.model.jnt_axis[:3],[[1,0,0],[0,0,1],[0,1,0]]))
    def test_initial_parameters_within_search_bounds(self):
        self.assertTrue(np.all(INITIAL>=LOW));self.assertTrue(np.all(INITIAL<=HIGH))
    def test_torques_are_limited(self):
        self.s.reset();self.s.data.qpos[3:]=10;self.s.data.qvel[3:]=-100
        self.assertTrue(np.all(np.abs(self.s.action(INITIAL))<=np.array([80,80,35]*2)))
    def test_action_does_not_script_root_motion(self):
        self.s.reset();before=self.s.data.qpos.copy();self.s.action(INITIAL)
        np.testing.assert_array_equal(before,self.s.data.qpos)
    def test_reset_seed_is_repeatable(self):
        self.s.reset(100);q=self.s.data.qpos.copy();v=self.s.data.qvel.copy();self.s.reset(100)
        np.testing.assert_array_equal(q,self.s.data.qpos);np.testing.assert_array_equal(v,self.s.data.qvel)
    def test_different_seeds_change_initial_state(self):
        self.s.reset(100);q=self.s.data.qvel.copy();self.s.reset(101)
        self.assertFalse(np.array_equal(q,self.s.data.qvel))
    def test_rollout_is_deterministic(self):
        a=self.s.rollout(INITIAL,seed=17,seconds=.3);b=self.s.rollout(INITIAL,seed=17,seconds=.3)
        self.assertEqual(a,b)
    def test_initial_policy_falls_and_terminates(self):
        r=self.s.rollout(INITIAL,seed=100,seconds=4)
        self.assertTrue(r['fallen']);self.assertLess(r['duration'],4)
    def test_replay_contains_physics_states(self):
        r=self.s.rollout(INITIAL,seconds=.12,record=True)
        self.assertEqual(len(r['frames']),4);self.assertEqual(len(r['frames'][0]['qpos']),9)
        self.assertGreater(r['frames'][-1]['t'],r['frames'][0]['t'])
    def test_invalid_policy_rejected(self):
        for p in ([0]*4,[math.nan]*16):
            with self.assertRaises(ValueError):self.s.rollout(p)
    def test_invalid_experiment_rejected(self):
        for args in ({'seconds':0},{'friction':-1},{'push':math.nan},{'push':300}):
            with self.assertRaises(ValueError):self.s.rollout(INITIAL,**args)
    def test_condition_reset_does_not_leak_push_or_friction(self):
        self.s.reset(1,.35,80);self.s.reset(1)
        self.assertEqual(self.s.push,0);self.assertTrue(np.all(self.s.model.geom_friction[:,0]==.9))
        self.assertTrue(np.all(self.s.data.xfrc_applied==0))
    def test_scheduled_push_changes_motion_only_after_two_seconds(self):
        policy=json.loads((Path(__file__).resolve().parents[1]/'artifacts/policy.json').read_text())['parameters']
        a=self.s.rollout(policy,seed=100,seconds=2.5,record=True)
        b=self.s.rollout(policy,seed=100,seconds=2.5,push=-80,record=True)
        self.assertEqual(a['frames'][:60],b['frames'][:60])
        self.assertGreater(abs(a['distance']-b['distance']),.05)

class EvaluationTests(unittest.TestCase):
    def good(self):return dict(fallen=False,duration=12,speed=.6,qualified_alternations=12,qualified_touchdowns=[6,6])
    def test_stationary_survival_is_not_walking(self):
        r=self.good();r['speed']=0;self.assertFalse(walking_success(r,12))
    def test_sliding_without_steps_is_not_walking(self):
        r=self.good();r['qualified_alternations']=0;self.assertFalse(walking_success(r,12))
    def test_one_foot_hopping_is_not_walking(self):
        r=self.good();r['qualified_touchdowns']=[0,10];self.assertFalse(walking_success(r,12))
    def test_fall_cannot_count_as_success(self):
        r=self.good();r['fallen']=True;self.assertFalse(walking_success(r,12))
    def test_short_trial_cannot_count_as_success(self):
        r=self.good();r['duration']=5;self.assertFalse(walking_success(r,12))
    def test_criteria_accept_complete_alternating_walk(self):self.assertTrue(walking_success(self.good(),12))
    def test_seed_groups_are_disjoint(self):
        seen={11,22}
        for _,_,_,base in CONDITIONS:
            group=set(range(base,base+20));self.assertFalse(seen&group);seen|=group
    def test_invalid_training_size_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):train(tmp,population=2)

if __name__=='__main__':unittest.main()
