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

import newton.solvers.euler.kernels  # For graph capture on CUDA <12.3
import newton.solvers.euler.particles  # For graph capture on CUDA <12.3
import newton.solvers.solver  # For graph capture on CUDA <12.3


@wp.kernel
def update_collider_mesh(
    src_points: wp.array(dtype=wp.vec3),
    src_shape: wp.array(dtype=int),
    res_mesh: wp.uint64,
    shape_transforms: wp.array(dtype=wp.transform),
    shape_body_id: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    dt: float,
):
    v = wp.tid()
    res = wp.mesh_get(res_mesh)

    shape_id = src_shape[v]
    p = wp.transform_point(shape_transforms[shape_id], src_points[v])

    X_wb = body_q[shape_body_id[shape_id]]

    cur_p = res.points[v] + dt * res.velocities[v]
    next_p = wp.transform_point(X_wb, p)
    res.velocities[v] = (next_p - cur_p) / dt
    res.points[v] = cur_p


class Excavator:
    def __init__(self, dune_params: list, options: argparse.Namespace):
        articulation_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        self.device = wp.get_device()

        # NOTE: may need to update these
        articulation_builder.default_body_armature = 0.01
        articulation_builder.default_joint_cfg.armature = 0.01
        articulation_builder.default_joint_cfg.mode = newton.JOINT_MODE_TARGET_POSITION
        articulation_builder.default_joint_cfg.target_ke = 10.0
        articulation_builder.default_joint_cfg.target_kd = 3.0

        # Excavator always spawns inside the floor regardless of xform used in parse_urdf (bug)
        # !!! Temporary solution until issue #259 on github is solved
        # Issue #259: When the asset is moved upward out of floor, it's only visual and the collision isn't cleared. 
        # Make sense since 1e6 contact damping is needed to prevent the excavator from flying away 
        articulation_builder.default_shape_cfg.ke = 1.0e4
        articulation_builder.default_shape_cfg.kd = 1.0e2
        articulation_builder.default_shape_cfg.kf = 1.0e2
        # builder.default_shape_cfg.ke = 1.0e2
        # builder.default_shape_cfg.kd = 1.0e2
        # builder.default_shape_cfg.kf = 1.0e2
        articulation_builder.default_shape_cfg.mu = 1.0

        # newton.utils.parse_urdf(
        #     Excavator.get_asset("excavator.urdf"),
        #     articulation_builder,
        #     up_axis=newton.Axis.Z,
        #     xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
        #     floating=True,  
        #     enable_self_collisions=True,
        #     ignore_inertial_definitions=False,
        #     force_show_colliders=True,
        # )
        newton.utils.parse_mjcf(
            Excavator.get_asset("excavator.xml"),
            articulation_builder,
            up_axis=newton.Axis.Z,
            xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
            floating=None,  
            # enable_self_collisions=True,
            ignore_inertial_definitions=False,
            # force_show_colliders=True,
            ignore_names=["floor", "ground"],
        )
        # [0.0, -5.0, 1.31]

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_builder(builder=articulation_builder, 
                            xform=wp.transform([0.0, -5.0, 0.0], wp.quat_identity()))
        builder.add_ground_plane()

        # Needed so the excavator doesn't fall through the ground (it's too heavy and can't counter this weight with contact stiffness params)
        # -1 is the world frame
        # builder.add_joint(joint_type=newton.JOINT_PRISMATIC,
        #                   parent=-1,  
        #                   child=builder.body_key.index("base_link"),
        #                   axis=[0, 0, 1],  # lock height
        #                   limits=[0.0, 0.0],
        #                   )

        # lock_roll = newton.ModelBuilder.JointDofConfig(axis=Axis.X, limit_lower=-0.01, limit_upper=0.01)
        # lock_pitch = newton.ModelBuilder.JointDofConfig(axis=Axis.Y, limit_lower=-0.01, limit_upper=0.01)

        # builder.add_joint(joint_type=newton.JOINT_REVOLUTE,
        #                   parent=-1,
        #                   child=builder.body_key.index("base_link"),
        #                   linear_axes=[lock_roll, lock_pitch], 
        #                   )

        self.actuated_joints = 4
        # target_vals = [0.0, 0.646, 2.47, -1.92]  # rads
        init_vals = [0.0, 0.4, 0.0, 0.0]  # rads
        target_vals = [0.0, 0.8, 0.0, 0.0]  # rads
        # init_vals = [0.0, 0.8, 0.0, 0.0]  # rads
        builder.joint_q[-self.actuated_joints:] = init_vals
        builder.joint_target[-self.actuated_joints:] = target_vals
        builder.joint_target[:-self.actuated_joints] = builder.joint_q[:-self.actuated_joints]

        # test_name = Excavator.get_asset("excavator.urdf")
        # if test_name:
        #     print("Got the filename: " + test_name)

        # options.grid_padding = 0 if options.dynamic_grid else 5
        # builder.gravity = wp.vec3(options.gravity)

        # options.yield_stresses = wp.vec3(
        #     options.yield_stress,
        #     -options.stretching_yield_stress,
        #     options.compression_yield_stress,
        # )

        # TODO: Uncomment
        # # add sand particles
        # Excavator._emit_gaussian_dunes(builder, dune_params, options)

        model: newton.Model = builder.finalize()
        self.model = model
        model.particle_mu = options.friction_coeff  # sand friction

        # From MathScavator9000.csv file 
        self.model.body_inertia = wp.array(
            [[[390840.535733491, 1.50663950062099e-10, 1.45611998268111e-11],
              [1.50663950062099e-10, 391481.853388216, -3915.94409741702],
              [1.45611998268111e-11, -3915.94409741702, 759007.34593037]],
              
              [[52076.3140370513, 2716.06120599586, 816.956173700325],
               [2716.06120599586, 26944.2657197114, 606.221908596681],
               [816.956173700325, 606.221908596681, 68968.3416230044]],
               
              [[22079.1743026141, 1.98204188980526e-10, 1.08721982722814e-08],
               [1.98204188980526e-10, 3017.47837524302, 1247.0083877576],
               [1.08721982722814e-08, 1247.0083877576, 19647.7828147159]],
               
              [[3745.72992467607, 1.27627827542214e-08, 4.66017295519215e-08],
               [1.27627827542214e-08, 3546.50673779034, -450.045626010779],
               [4.66017295519215e-08, -450.045626010779, 257.342729612262]],
               
              [[3257.00344965316, -1.05357918855887e-08, -1.63126056129540e-08],
               [-1.05357918855887e-08, 2016.29928536516, 638.207540993486],
               [-1.63126056129540e-08, 638.207540993486, 2725.30205196773]],
               
               ],
                    dtype=wp.mat33f,
                )

        self.model.body_mass = wp.array(
            [139411.733176679, 17434.1065917836, 7226.51425767388,
            3327.88972318624, 5119.51057768466],
            dtype=wp.float32,
        )

        # TODO: Uncomment
        # # From anymal sand walk ex. Grab meshes for collisions w/ sand
        # # collider_body_idx = [idx for idx, key in enumerate(builder.body_key) 
        # #                      if "link" in key]
        # # collider_body_idx = [idx for idx, key in enumerate(builder.body_key) 
        # #                      if key == "base_link" or key == "base_chassis_link" or 
        # #                      key == "stick_bucket_link"]
        # collider_body_idx = [idx for idx, key in enumerate(builder.body_key) 
        #                      if key == "base_link" or key == "stick_bucket_link"]
        # collider_shape_ids = np.concatenate(
        #     [[m for m in self.model.body_shapes[b] if self.model.shape_geo_src[m]] for b in collider_body_idx]
        # )

        # collider_points, collider_indices, collider_v_shape_ids = _merge_meshes(
        #     [self.model.shape_geo_src[m].vertices for m in collider_shape_ids],
        #     [self.model.shape_geo_src[m].indices for m in collider_shape_ids],
        #     [self.model.shape_geo.scale.numpy()[m] for m in collider_shape_ids],
        #     collider_shape_ids,
        # )

        # self.collider_mesh = wp.Mesh(wp.clone(collider_points), collider_indices, wp.zeros_like(collider_points))
        # self.collider_rest_points = collider_points
        # self.collider_shape_ids = wp.array(collider_v_shape_ids, dtype=int)

        self.sim_time = 0.0
        self.frame_dt = 1.0 / options.fps
        self.sim_substeps = options.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        # self.solver = newton.solvers.FeatherstoneSolver(self.model)
        self.solver = newton.solvers.MuJoCoSolver(self.model, 
                                                  disable_contacts=False,
                                                  use_mujoco=False,  # needs to be false since parallelizing the MPM solver
                                                  solver="newton",
                                                  integrator="euler",
                                                  iterations=10,
                                                  ncon_per_env=150,
                                                  ls_iterations=5,
                                                  )
        # TODO: Uncomment
        # options_mpm = ImplicitMPMSolver.Options()
        # options_mpm.voxel_size = options.voxel_size
        # options_mpm.max_fraction = options.max_fraction
        # options_mpm.tolerance = options.tolerance
        # options_mpm.unilateral = options.unilateral
        # options_mpm.max_iterations = options.max_iterations
        # # options_mpm.gauss_seidel = False
        # options_mpm.dynamic_grid = options.dynamic_grid
        # if not options.dynamic_grid:
        #     options_mpm.grid_padding = 5

        # self.mpm_solver = ImplicitMPMSolver(self.model, options_mpm)
        # self.mpm_solver.setup_collider(self.model, [self.collider_mesh])

        if options.headless:
            self.renderer = None
        else:
            self.renderer = newton.utils.SimRendererOpenGL(self.model, "Excavator + Dunes", scaling=1.0, show_joints=True)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        # TODO: add control OR pretrained policy

        # self.state_0.clear_forces()
        # self.state_1.clear_forces()

        # TODO: Uncomment
        # self.mpm_solver.enrich_state(self.state_0)
        # self.mpm_solver.enrich_state(self.state_1)
        # self._update_collider_mesh(self.state_0)

        newton.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        
        self.use_cuda_graph = self.device.is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            # # Initial graph launch, load modules (necessary for drivers prior to CUDA 12.3)
            # wp.load_module(newton.solvers.euler.kernels, device=wp.get_device())
            # wp.load_module(newton.solvers.euler.particles, device=wp.get_device())
            # wp.load_module(newton.solvers.solver, device=wp.get_device())

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

            # self.controller.assign_control(self.control, self.state_0)
            # TODO: add control
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
    
    def simulate_sand(self):
        self._update_collider_mesh(self.state_0)
        # solve in-place, avoids having to resync robot sim state
        self.mpm_solver.step(self.state_0, self.state_0, contacts=None, control=None, dt=self.frame_dt)

    def step(self):
        # TODO: change bac to True
        with wp.ScopedTimer("step", synchronize=False):
            if self.use_cuda_graph:
                wp.capture_launch(self.robot_graph)
            else:
                self.simulate_robot()
            
            # TODO: Uncomment
            # self.simulate_sand()

            # self.controller.get_control(self.state_0)

        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return
        
        # TODO: change bac to True
        with wp.ScopedTimer("render", synchronize=False):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            # self.renderer.render_contacts(self.state_0, self.contacts, contact_point_radius=1e-2)
            self.renderer.end_frame()

    def _update_collider_mesh(self, state):
        wp.launch(
            update_collider_mesh,
            dim=self.collider_rest_points.shape[0],
            inputs=[
                self.collider_rest_points,
                self.collider_shape_ids,
                self.collider_mesh.id,
                self.model.shape_transform,
                self.model.shape_body,
                state.body_q,
                self.frame_dt,
            ],
        )
        self.collider_mesh.refit()

    #
    # Static methods
    #
    @staticmethod
    def _emit_gaussian_dunes(builder: newton.ModelBuilder, dunes, args):
        """
        Args: list of dicts with keys:
          - center: (x, z)
          - amplitude: peak height
          - sigma: gaussian std-dev
        """
        voxel_size = args.voxel_size
        packing_fraction = args.max_fraction
        all_points = []

        for dune in dunes:
            x0, y0 = dune['center']
            amplitude = dune['amplitude']
            sigma = dune['sigma']

            # truncate Gaussian at 3*sigma
            trunc = 3.0 * sigma

            # sample resolution roughly one third of voxel
            samples_per_axis = int(np.ceil((2 * trunc) / (voxel_size / 3)))
            xs = np.linspace(x0 - trunc, x0 + trunc, samples_per_axis)
            ys = np.linspace(y0 - trunc, y0 + trunc, samples_per_axis)

            for x in xs:
                for y in ys:
                    height = amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
                    layers = max(1, int(np.ceil(height / (voxel_size / args.particles_per_height))))  # want about 3 particles per voxel height
                    # ys = np.linspace(voxel_size / 2, height, layers)  # NOTE: y is set as the up direction 
                    zs = np.linspace(0, height, layers)  # NOTE: z is set as the up direction 
                    for z in zs:
                        all_points.append([x, y, z])

        pts = np.array(all_points)
        vel = np.zeros_like(pts)

        cell_volume = voxel_size ** 3
        masses = np.full(len(pts), cell_volume * packing_fraction)
        radii = np.full(len(pts), voxel_size * 0.5)
        flags = np.zeros(len(pts), dtype=int)

        builder.particle_q = pts
        builder.particle_qd = vel
        builder.particle_mass = masses
        builder.particle_radius = radii
        builder.particle_flags = flags

        print("Particle count: ", pts.shape[0])

    @staticmethod
    def get_source_directory() -> str:
        return os.path.realpath(os.path.dirname(__file__))

    @staticmethod
    def get_asset_directory() -> str:
        return os.path.join(Excavator.get_source_directory(), "assets")

    @staticmethod
    def get_asset(filename: str) -> str:
        return os.path.join(Excavator.get_asset_directory(), filename)


def _merge_meshes(
    points: list[np.array],
    indices: list[np.array],
    scales: list[np.array],
    shape_ids: list[int],
):
    pt_count = np.array([len(pts) for pts in points])
    offsets = np.cumsum(pt_count) - pt_count

    mesh_id = np.repeat(np.arange(len(points), dtype=int), repeats=pt_count)

    merged_points = np.vstack([pts * scale for pts, scale in zip(points, scales)])

    merged_indices = np.concatenate([idx + offsets[k] for k, idx in enumerate(indices)])

    return (
        wp.array(merged_points, dtype=wp.vec3),
        wp.array(merged_indices, dtype=int),
        wp.array(np.array(shape_ids)[mesh_id], dtype=int),
    )


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

    # NOTE: Update these for sand
    parser.add_argument("--friction_coeff", "-mu", type=float, default=0.5)  # for sand
    parser.add_argument("--max_iterations", "-it", type=int, default=250)
    parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-5)
    parser.add_argument("--particles_per_height", "-pph", type=float, default=3.0)  # want about 3 particles per voxel height
    parser.add_argument("--dynamic_grid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unilateral", action=argparse.BooleanOptionalAction, default=False)

    ## Extra for sand
    # parser.add_argument("--compliance", type=float, default=0.0)
    # parser.add_argument("--poisson_ratio", "-nu", type=float, default=0.3)
    # parser.add_argument("--yield_stress", "-ys", type=float, default=0.0)
    # parser.add_argument("--compression_yield_stress", "-cys", type=float, default=1.0e8)
    # parser.add_argument("--stretching_yield_stress", "-sys", type=float, default=1.0e8)
    # parser.add_argument("--gauss_seidel", "-gs", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_known_args()[0]

    if wp.get_device(args.device).is_cpu:
        print("Error: This example requires a GPU device.")
        sys.exit(1)

    # define two dunes
    # dunes = [
    #     {"center": (2.5,  1.0), "amplitude": 0.5, "sigma": 0.3},
    #     {"center": (3.5, 2.5), "amplitude": 0.7, "sigma": 0.4},
    # ]
    dunes = [
        {"center": (0.0,  1.0), "amplitude": 0.5, "sigma": 0.3},
        {"center": (1.5, -0.5), "amplitude": 0.7, "sigma": 0.4},
    ]

    with wp.ScopedDevice(args.device):
        sim = Excavator(dunes, args)

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
                # !!! ISSUE: .get_data_into method not working. This worked for all the other examples 
                if not sim.solver.use_mujoco:
                    mujoco_warp.get_data_into(mjd, mjm, d)
                viewer.sync()

        if sim.renderer:
            sim.renderer.save()
