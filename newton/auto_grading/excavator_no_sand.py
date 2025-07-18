# Author: Lance Pharand
# Based on ex_anymal_c_walk

import argparse
import numpy as np
import warp as wp
import os
import sys

import newton
from newton.solvers.featherstone import FeatherstoneSolver
from newton.solvers.implicit_mpm import ImplicitMPMSolver
from newton.solvers.mujoco import MuJoCoSolver
import newton.utils
import newton.sim


class Excavator:
    def __init__(self, options: argparse.Namespace):
        articulation_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        self.device = wp.get_device()

        # NOTE: may need to update these
        articulation_builder.default_body_armature = 0.01
        articulation_builder.default_joint_cfg.armature = 0.01
        articulation_builder.default_joint_cfg.mode = newton.JOINT_MODE_TARGET_POSITION
        articulation_builder.default_joint_cfg.target_ke = 10.0
        articulation_builder.default_joint_cfg.target_kd = 3.0

        articulation_builder.default_shape_cfg.mu = 1.0

        newton.utils.parse_mjcf(
            Excavator.get_asset("excavator.xml"),
            articulation_builder,
            up_axis=newton.Axis.Z,
            xform=wp.transform([0.0, 0.0, 2.0], wp.quat_identity()),
            floating=None,  
            # enable_self_collisions=True,
            ignore_inertial_definitions=False,
            # force_show_colliders=True,
            ignore_names=["floor", "ground"],
        )
        # newton.utils.parse_urdf(
        #     Excavator.get_asset("excavator.urdf"),
        #     articulation_builder,
        #     up_axis=newton.Axis.Z,
        #     xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
        #     floating=False,  
        #     enable_self_collisions=True,
        #     ignore_inertial_definitions=False,
        #     force_show_colliders=True,
        # )

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_builder(builder=articulation_builder, 
                            xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()))
        builder.add_ground_plane()

        # self.actuated_joints = 4
        # # target_vals = [0.0, 0.646, 2.47, -1.92]  # rads
        # init_vals = [0.0, 0.4, 0.0, 0.0]  # rads
        # target_vals = [0.0, 0.8, 0.0, 0.0]  # rads
        # # init_vals = [0.0, 0.8, 0.0, 0.0]  # rads
        # builder.joint_q[-self.actuated_joints:] = init_vals
        # builder.joint_target[-self.actuated_joints:] = target_vals
        # builder.joint_target[:-self.actuated_joints] = builder.joint_q[:-self.actuated_joints]

        model: newton.Model = builder.finalize()
        self.model = model

        self.sim_time = 0.0
        self.frame_dt = 1.0 / options.fps
        self.sim_substeps = options.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.solver = newton.solvers.MuJoCoSolver(self.model, 
                                                  disable_contacts=False,
                                                  use_mujoco=False,  # needs to be false since parallelizing the MPM solver
                                                  solver="newton",
                                                  integrator="euler",
                                                  iterations=10,
                                                  ncon_per_env=150,
                                                  ls_iterations=5,
                                                  )

        if options.headless:
            self.renderer = None
        else:
            self.renderer = newton.utils.SimRendererOpenGL(self.model, "Excavator + Dunes", scaling=1.0, show_joints=True)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state=self.state_0)
        
        self.use_cuda_graph = self.device.is_cuda and wp.is_mempool_enabled(wp.get_device()) \
                                and not self.solver.use_mujoco
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate_robot()
            self.robot_graph = capture.graph
        else:
            self.robot_graph = None

    def simulate_robot(self):
        self.contacts = None

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # self.contacts not needed for mujoco solver
            if not isinstance(self.solver, MuJoCoSolver):
                self.contacts = self.model.collide(self.state_0, rigid_contact_margin=0.01)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
    
    def step(self):
        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.robot_graph)
            else:
                self.simulate_robot()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return
        
        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            # self.renderer.render_contacts(self.state_0, self.contacts, contact_point_radius=1e-2)
            self.renderer.end_frame()

    #
    # Static methods
    #
    @staticmethod
    def get_source_directory() -> str:
        return os.path.realpath(os.path.dirname(__file__))

    @staticmethod
    def get_asset_directory() -> str:
        return os.path.join(Excavator.get_source_directory(), "assets")

    @staticmethod
    def get_asset(filename: str) -> str:
        return os.path.join(Excavator.get_asset_directory(), filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    
    # parser.add_argument("--collider", type=str)

    # parser.add_argument("--urdf", type=str, default="./assets/excavator.urdf")
    # parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -10])
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--max_fraction", type=float, default=1.0)
    parser.add_argument("--voxel_size", "-dx", type=float, default=0.1)
    parser.add_argument("--num_frames", type=int, default=500, help="Total number of frames.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show_mujoco_viewer", action=argparse.BooleanOptionalAction, default=True)
 
    args = parser.parse_known_args()[0]

    if wp.get_device(args.device).is_cpu:
        print("Error: This example requires a GPU device.")
        sys.exit(1)

    with wp.ScopedDevice(args.device):
        sim = Excavator(args)

        if args.show_mujoco_viewer:
            import mujoco
            import mujoco.viewer
            import mujoco_warp

            mjm, mjd = sim.solver.mj_model, sim.solver.mj_data
            m, d = sim.solver.mjw_model, sim.solver.mjw_data
            viewer = mujoco.viewer.launch_passive(mjm, mjd)

        for _ in range(args.num_frames):
            sim.step()
            sim.render()

            if args.show_mujoco_viewer:
                # !!! ISSUE: .get_data_into method not working? This worked for all the other examples 
                if not sim.solver.use_mujoco:
                    mujoco_warp.get_data_into(mjd, mjm, d)
                viewer.sync()

        if sim.renderer:
            sim.renderer.save()
