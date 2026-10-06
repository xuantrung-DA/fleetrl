"""Focused conversion tests; no simulator behavior implemented here."""
import unittest
from export_fleetrl_replay import snapshot_to_frame


class AdapterTest(unittest.TestCase):
    def source(self):
        return {"sim_time_s": 2, "observed_at_s": 2, "robots": [
            {"id": 0, "kind": "L", "position": [2, 2], "capacity_wh": 180,
             "battery_wh": 90, "speed_mps": 2, "status": "docking", "load_kg": 0}
        ], "tasks": [], "ports": [{"id": 0, "position": [2, 2], "queue_cells": [[2, 3]]}],
                "bookings": [{"robot_id": 0, "port_id": 0, "start_s": 10,
                              "end_s": 20, "requested_at_s": 1, "target_soc": .8}]}

    def test_occupancy_not_charging(self):
        source = self.source()
        frame = snapshot_to_frame(source)
        self.assertEqual(frame["robots"][0]["soc"], .5)
        self.assertEqual(frame["ports"][0]["occupied_by"], 0)
        self.assertIsNone(frame["ports"][0]["charging_robot"])
        self.assertEqual(frame["kpi"], {})
        source["robots"][0]["status"] = "charging"
        self.assertEqual(snapshot_to_frame(source)["ports"][0]["charging_robot"], 0)

    def test_motion_and_metrics(self):
        source = self.source()
        source["robots"][0].update(move_to=[3, 2], move_remaining_s=.25)
        frame = snapshot_to_frame(source, kpi={"pending": 8})
        self.assertEqual(frame["robots"][0]["edge_duration_s"], .5)
        self.assertEqual(frame["kpi"], {"pending": 8})

    def test_future_booking(self):
        source = self.source()
        source["bookings"][0]["requested_at_s"] = 3
        with self.assertRaises(ValueError):
            snapshot_to_frame(source)

    def test_future_task(self):
        source = self.source()
        source["tasks"] = [{"created_s": 3}]
        with self.assertRaises(ValueError):
            snapshot_to_frame(source)


if __name__ == "__main__":
    unittest.main()
