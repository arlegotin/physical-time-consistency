"""Fully observed smooth MuJoCo plant, with authoritative integer time."""
from __future__ import annotations

from pathlib import Path
import mujoco
import numpy as np

from .clocks import to_ns
from .io import sha256_file


class Plant:
    def __init__(self, actuator: str, physics_step_ns: int = 500_000, asset_root: Path = Path('assets')):
        if actuator not in {'direct', 'lagged'}:
            raise ValueError('Unknown actuator chain')
        if isinstance(physics_step_ns, bool) or not isinstance(physics_step_ns, (int,np.integer)) or physics_step_ns <= 0:
            raise ValueError('Positive integer numerical tick required')
        self.actuator, self.physics_step_ns = actuator, int(physics_step_ns)
        self.xml_path = Path(asset_root) / f'two_link_{actuator}.xml'
        self.xml_sha256 = sha256_file(self.xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.model.opt.timestep = self.physics_step_ns/1e9
        self.data = mujoco.MjData(self.model)
        self.time_ns = 0
        self._assert_compiled()
        self.reset(np.zeros(4 if actuator == 'direct' else 6))

    def _assert_compiled(self):
        model = self.model
        if (model.nq,model.nv,model.nu,model.na) != (2,2,2,0 if self.actuator == 'direct' else 2):
            raise ValueError('Plant state dimensions do not match protocol')
        expected = mujoco.mjtDyn.mjDYN_NONE if self.actuator == 'direct' else mujoco.mjtDyn.mjDYN_FILTER
        tests = [np.all(model.opt.gravity == 0), not np.any(model.jnt_limited),
                 np.all(model.dof_armature == .02), np.all(model.dof_damping == .05),
                 np.all(model.actuator_dyntype == expected), not np.any(model.actuator_actearly),
                 np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED),
                 np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE),
                 np.all(model.actuator_gainprm[:,0] == 1), np.all(model.actuator_gear[:,0] == 1),
                 np.all(model.actuator_gear[:,1:] == 0),
                 np.array_equal(model.actuator_trnid[:,0], [0,1]),
                 np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT),
                 bool(model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)),
                 bool(model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)),
                 model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4,
                 np.all(model.actuator_ctrllimited), np.array_equal(model.actuator_ctrlrange, [[-2,2]]*2)]
        if self.actuator == 'lagged':
            tests += [np.all(model.actuator_dynprm[:,0] == .08), not np.any(model.actuator_actlimited)]
        if not all(tests):
            raise ValueError('Compiled plant parameters differ from the exact protocol')

    def compiled_manifest(self) -> dict:
        model = self.model
        arrays = ('body_mass','body_ipos','body_inertia','body_iquat','body_pos','body_quat','jnt_axis',
                  'dof_armature','dof_damping','actuator_gear','actuator_trnid','actuator_gainprm','actuator_biasprm',
                  'actuator_dynprm','actuator_ctrlrange','actuator_actadr','actuator_trntype','actuator_gaintype','actuator_biastype')
        result = {name: getattr(model,name).tolist() for name in arrays}
        result.update({name: int(getattr(model,name)) for name in ('nq','nv','nu','na')})
        result.update({'actuator': self.actuator, 'xml_sha256': self.xml_sha256, 'gravity': model.opt.gravity.tolist(),
                       'physics_step_ns': self.physics_step_ns, 'timestep_s': float(model.opt.timestep),
                       'integrator': 'RK4', 'joint_limited': model.jnt_limited.astype(bool).tolist(),
                       'actuator_actearly': model.actuator_actearly.astype(bool).tolist(),
                       'actuator_actlimited': model.actuator_actlimited.astype(bool).tolist(),
                       'actuator_dyntype': ['none' if value == mujoco.mjtDyn.mjDYN_NONE else 'filter' for value in model.actuator_dyntype],
                       'disableflags': int(model.opt.disableflags),
                       'contact_disabled': bool(model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)),
                       'autoreset_disabled': bool(model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)),
                       'units': {'q':'rad','v':'rad/s','u':'N*m','w':'N*m','time':'ns'}})
        return result

    def state(self) -> np.ndarray:
        return np.concatenate((self.data.qpos.copy(), self.data.qvel.copy(), self.data.act.copy())).astype(np.float64)

    def reset(self, state: np.ndarray, time_s: float = 0.) -> None:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (4 if self.actuator == 'direct' else 6,) or not np.isfinite(state).all():
            raise ValueError('Finite fully observed q/v[/w] state required')
        time_ns = to_ns(time_s)
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:], self.data.qvel[:] = state[:2], state[2:4]
        if self.actuator == 'lagged':
            self.data.act[:] = state[4:]
        self.data.time = time_ns/1e9
        self.time_ns = time_ns
        self.data.ctrl[:], self.data.qfrc_applied[:], self.data.xfrc_applied[:], self.data.qacc_warmstart[:] = 0,0,0,0
        mujoco.mj_forward(self.model,self.data)
        self._validate()

    def _validate(self):
        if any(warning.number for warning in self.data.warning):
            raise RuntimeError('MuJoCo warning invalidates the plant reference run')
        if self.data.ncon:
            raise RuntimeError('Unexpected contact in smooth plant')
        if not np.isfinite(self.state()).all():
            raise RuntimeError('Nonfinite true plant state')
        if abs(self.data.time-self.time_ns/1e9) > 1e-10:
            raise RuntimeError('Floating simulator time differs from integer physical clock')

    def advance(self, command: np.ndarray, duration_ns: int, external_torque: np.ndarray | None = None,
                log_every_step: bool = False) -> dict:
        command = np.asarray(command, dtype=np.float64)
        if command.shape != (2,) or not np.isfinite(command).all() or np.any(np.abs(command)>2):
            raise ValueError('Two finite physical commands in [-2,2] N*m required')
        if isinstance(duration_ns,bool) or not isinstance(duration_ns,(int,np.integer)) or duration_ns<=0 or duration_ns%self.physics_step_ns:
            raise ValueError('Positive integer hold duration divisible by numerical tick required')
        torque = np.zeros(2) if external_torque is None else np.asarray(external_torque,dtype=np.float64)
        if torque.shape != (2,) or not np.isfinite(torque).all():
            raise ValueError('External torque must have two finite components')
        self._validate()
        self.data.ctrl[:], self.data.qfrc_applied[:] = command,torque
        count = int(duration_ns)//self.physics_step_ns
        times, states = [], []
        if log_every_step:
            for _ in range(count):
                mujoco.mj_step(self.model,self.data)
                self.time_ns += self.physics_step_ns
                self._validate()
                times.append(self.time_ns)
                states.append(self.state())
        else:
            mujoco.mj_step(self.model,self.data,nstep=count)
            self.time_ns += int(duration_ns)
            self._validate()
        return {'state': self.state(), 'time_ns': self.time_ns, 'status': 'ok', 'physics_steps': count,
                'trajectory': {'time_ns': np.asarray(times,dtype=np.int64),
                               'state': np.asarray(states,dtype=np.float64).reshape(-1,4 if self.actuator == 'direct' else 6)}}

    def snapshot(self) -> dict:
        specification = mujoco.mjtState.mjSTATE_INTEGRATION
        integration = np.empty(mujoco.mj_stateSize(self.model,specification),dtype=np.float64)
        mujoco.mj_getState(self.model,self.data,integration,specification)
        return {'state': self.state().tolist(), 'time_ns': self.time_ns, 'actuator': self.actuator,
                'physics_step_ns': self.physics_step_ns, 'xml_sha256': self.xml_sha256,
                'integration_state': integration.tolist(), 'state_specification': int(specification),
                'controls': self.data.ctrl.tolist(), 'external_torque': self.data.qfrc_applied.tolist(),
                'warnings': [{'number': int(warning.number),'lastinfo': int(warning.lastinfo)} for warning in self.data.warning]}

    def restore(self, snapshot: dict) -> None:
        if snapshot['actuator'] != self.actuator or snapshot['physics_step_ns'] != self.physics_step_ns or snapshot['xml_sha256'] != self.xml_sha256:
            raise ValueError('Snapshot belongs to a different plant or numerical tick')
        specification = mujoco.mjtState.mjSTATE_INTEGRATION
        integration = np.asarray(snapshot['integration_state'],dtype=np.float64)
        if integration.shape != (mujoco.mj_stateSize(self.model,specification),) or not np.isfinite(integration).all():
            raise ValueError('Incomplete/nonfinite integration snapshot')
        self.data = mujoco.MjData(self.model)
        mujoco.mj_setState(self.model,self.data,integration,specification)
        self.time_ns = int(snapshot['time_ns'])
        mujoco.mj_forward(self.model,self.data)
        for warning, record in zip(self.data.warning,snapshot['warnings']):
            warning.number, warning.lastinfo = record['number'],record['lastinfo']
        if not np.array_equal(self.state(),snapshot['state']):
            raise ValueError('Restored physical state differs from serialized snapshot')
        self._validate()
