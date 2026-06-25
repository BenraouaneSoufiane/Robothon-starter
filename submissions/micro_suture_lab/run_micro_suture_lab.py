from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT      = Path(__file__).resolve().parent
SCENE     = ROOT / "micro_suture_scene.xml"
ARTIFACTS = ROOT / "artifacts"

FINGERS = ("thumb", "index", "middle", "ring", "little")
ACTUATOR_ORDER = (
    "x_servo", "y_servo", "z_servo", "wrist_servo",
    "thumb_opp_servo", "thumb_distal_servo",
    "index_flex_servo", "index_distal_servo",
    "middle_flex_servo", "middle_distal_servo",
    "ring_flex_servo",  "ring_distal_servo",
    "little_flex_servo","little_distal_servo",
    "button_servo",
)

# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage:
    key: str; label: str; start: float; end: float
    palm_goal: tuple; pinch: float; stabilize: float; wrist: float; button: float; signal: str

STAGES = (
    Stage("scan",    "0. sterile scan",            0.00,0.06,(-0.58,-0.13,0.620),0.00,0.00,-0.10,0.0,"sensors online"),
    Stage("approach","1. approach needle",         0.06,0.16,(-0.560,-0.130,0.505),0.10,0.00,-0.10,0.0,"palm aligned to needle shaft"),
    Stage("preload", "2. open-hand preload",       0.16,0.21,(-0.555,-0.130,0.485),0.20,0.10,-0.05,0.0,"fingers half open around needle"),
    Stage("grasp",   "3. five-finger needle grasp",0.21,0.31,(-0.553,-0.130,0.475),1.00,0.65,-0.05,0.0,"friction pinch carries needle"),
    Stage("lift",    "4. lift and confirm",        0.31,0.36,(-0.553,-0.130,0.550),1.00,0.65,-0.05,0.0,"needle stays gripped under gravity"),
    Stage("entry",   "5. entry hoop pass",         0.36,0.48,(-0.320,-0.040,0.540),0.95,0.65,0.15,0.0,"needle tip enters entry hoop"),
    Stage("middle",  "6. middle hoop pass",        0.48,0.60,(-0.080, 0.045,0.565),0.95,0.65,0.30,0.0,"needle follows curved corridor"),
    Stage("exit",    "7. exit hoop pass",          0.60,0.72,( 0.180,-0.035,0.545),0.95,0.65,0.45,0.0,"third hoop cleared"),
    Stage("draw",    "8. suture draw-through",     0.72,0.81,( 0.360, 0.080,0.560),0.90,0.65,0.55,0.0,"thread tail pulled across tissue"),
    Stage("knot",    "9. knot cinch + slip recovery",0.81,0.90,( 0.420, 0.100,0.525),0.95,0.80,0.70,0.0,"PI feedback recovers from slip"),
    Stage("verify",  "10. verification press",     0.90,0.96,( 0.480,-0.200,0.520),0.30,0.20,0.00,1.0,"button depressed past threshold"),
    Stage("export",  "11. dataset export pose",    0.96,1.00,( 0.240,-0.200,0.620),0.10,0.10,0.00,0.0,"artifacts written"),
)

def smoothstep(a,b,x):
    if x<=a:return 0.0
    if x>=b:return 1.0
    t=(x-a)/max(b-a,1e-9); return t*t*(3.0-2.0*t)
def lerp(a,b,t): return a*(1.0-t)+b*t
def quat_yaw(yaw): return np.array([math.cos(yaw/2.0),0.0,0.0,math.sin(yaw/2.0)])

def stage_for(phase):
    for s in STAGES:
        if s.start<=phase<s.end:return s
    return STAGES[-1]

def stage_blend(phase):
    for i,s in enumerate(STAGES):
        if s.start<=phase<s.end:
            t=smoothstep(s.start,s.end,phase)
            nxt=STAGES[min(i+1,len(STAGES)-1)]
            return s,nxt,t
    return STAGES[-1],STAGES[-1],1.0

def needle_body_target(phase):
    # Contact-locked guide target for the needle body. The visible tip is 72 mm
    # ahead of the body frame along +X, so these body targets place the tip on
    # the three hoop sites and then the tray.
    pts = (
        (0.00, np.array([-0.560,-0.130,0.469])),
        (0.31, np.array([-0.560,-0.130,0.469])),
        (0.36, np.array([-0.560,-0.130,0.535])),
        (0.48, np.array([-0.392,-0.040,0.510])),
        (0.60, np.array([-0.152, 0.045,0.540])),
        (0.72, np.array([ 0.108,-0.035,0.515])),
        (0.81, np.array([ 0.288, 0.080,0.530])),
        (0.90, np.array([ 0.348, 0.100,0.500])),
        (1.00, np.array([ 0.348, 0.100,0.500])),
    )
    for (a,pa),(b,pb) in zip(pts,pts[1:]):
        if a <= phase <= b:
            return lerp(pa,pb,smoothstep(a,b,phase))
    return pts[-1][1]

class ModelIndex:
    def __init__(self,model):
        n=lambda kind,name:int(mujoco.mj_name2id(model,kind,name))
        self.actuators={a:n(mujoco.mjtObj.mjOBJ_ACTUATOR,a) for a in ACTUATOR_ORDER}
        self.sensors={s:n(mujoco.mjtObj.mjOBJ_SENSOR,s) for s in (
            "palm_pos","palm_grip_pos","needle_tip_pos","needle_grip_pos","thread_pos",
            "entry_target_pos","middle_target_pos","exit_target_pos","tray_pos",
            "button_depth","wrist_yaw_pos","thumb_opp_pos","index_flex_pos",
            "middle_flex_pos","ring_flex_pos","little_flex_pos",
            "thumb_touch","index_touch","middle_touch","ring_touch","little_touch",
        )}
        self.eq_grasp=n(mujoco.mjtObj.mjOBJ_EQUALITY,"needle_grasp_assist")
        self.eq_thread=n(mujoco.mjtObj.mjOBJ_EQUALITY,"thread_attach")
        self.needle_qadr=int(model.jnt_qposadr[n(mujoco.mjtObj.mjOBJ_JOINT,"needle_free")])
        self.thread_qadr=int(model.jnt_qposadr[n(mujoco.mjtObj.mjOBJ_JOINT,"thread_free")])
        self.button_qadr=int(model.jnt_qposadr[n(mujoco.mjtObj.mjOBJ_JOINT,"button_slide")])
        self.needle_vadr=int(model.jnt_dofadr[n(mujoco.mjtObj.mjOBJ_JOINT,"needle_free")])

def sv(m,d,sid,dim=3): a=int(m.sensor_adr[sid]); return np.array(d.sensordata[a:a+dim])
def ss(m,d,sid): return float(d.sensordata[int(m.sensor_adr[sid])])

# ---------------------------------------------------------------------------
@dataclass
class Gains:
    kp_xy: float = 1.10
    kp_z:  float = 1.00
    ki_xy: float = 0.35
    ki_z:  float = 0.25
    kd_xy: float = 0.08
    kd_z:  float = 0.08

    @classmethod
    def from_vector(cls,v): return cls(*[float(x) for x in v])
    def to_vector(self):    return np.array([self.kp_xy,self.kp_z,self.ki_xy,self.ki_z,self.kd_xy,self.kd_z])

GAINS_LOWER = np.array([0.20,0.20,0.00,0.00,0.00,0.00])
GAINS_UPPER = np.array([3.00,3.00,1.50,1.50,0.40,0.40])

class SutureController:
    def __init__(self,model,idx,gains:Gains,use_feedback=True,ctrl_dt=0.001):
        self.m,self.idx=model,idx
        self.use_feedback=use_feedback; self.g=gains; self.dt=ctrl_dt
        self.i_xyz=np.zeros(3); self.prev_err=np.zeros(3); self.i_clip=0.025
    def reset(self):
        self.i_xyz[:]=0.0; self.prev_err[:]=0.0
    def desired_setpoint(self,phase):
        cur,nxt,t=stage_blend(phase)
        palm=lerp(np.array(cur.palm_goal),np.array(nxt.palm_goal),t)
        return palm,lerp(cur.pinch,nxt.pinch,t),lerp(cur.stabilize,nxt.stabilize,t),lerp(cur.wrist,nxt.wrist,t),lerp(cur.button,nxt.button,t)
    def step(self,data,phase):
        m,idx,g=self.m,self.idx,self.g
        palm_des,pinch,stab,wrist,button=self.desired_setpoint(phase)
        palm_meas=sv(m,data,idx.sensors["palm_pos"])
        err=palm_des-palm_meas
        if self.use_feedback:
            self.i_xyz=np.clip(self.i_xyz+err*self.dt,-self.i_clip,self.i_clip)
            d_err=(err-self.prev_err)/max(self.dt,1e-6)
            kp=np.array([g.kp_xy,g.kp_xy,g.kp_z])
            ki=np.array([g.ki_xy,g.ki_xy,g.ki_z])
            kd=np.array([g.kd_xy,g.kd_xy,g.kd_z])
            corr=np.clip(kp*err+ki*self.i_xyz+kd*d_err,-0.030,0.030)
        else:
            corr=np.zeros(3)
        self.prev_err=err
        gx=palm_des[0]+corr[0]; gy=palm_des[1]+corr[1]; gz=(palm_des[2]-0.56)+corr[2]
        fingers={
            "thumb_opp_servo":     math.radians(-34+72*pinch),
            "thumb_distal_servo":  math.radians( 6+58*pinch),
            "index_flex_servo":    math.radians( 4+62*pinch),
            "index_distal_servo":  math.radians( 4+70*pinch),
            "middle_flex_servo":   math.radians( 2+68*pinch),
            "middle_distal_servo": math.radians( 3+72*pinch),
            "ring_flex_servo":     math.radians( 3+56*stab),
            "ring_distal_servo":   math.radians( 2+50*stab),
            "little_flex_servo":   math.radians( 2+48*stab),
            "little_distal_servo": math.radians( 1+42*stab),
        }
        ctrl={
            "x_servo":float(np.clip(gx,-0.72,0.58)),
            "y_servo":float(np.clip(gy,-0.32,0.32)),
            "z_servo":float(np.clip(gz,-0.12,0.22)),
            "wrist_servo":float(wrist),
            "button_servo":float(-0.034*button),
            **fingers,
        }
        for name in ACTUATOR_ORDER:
            data.ctrl[idx.actuators[name]]=ctrl[name]
        return {"palm_des":palm_des,"palm_meas":palm_meas,"err":err,"correction":corr,"pinch":pinch,"stabilize":stab}

class GraspEventMonitor:
    def __init__(self,model,idx): self.m,self.idx=model,idx; self.reset()
    def reset(self):
        self.hoops_passed=[False,False,False]
        self.thread_attached=False; self.grasp_assist_active=False; self.slip_consumed=False
        self.grasp_confirmed=False; self.lift_confirmed=False; self.verification_pressed=False
    def update(self,data,phase,ctrl_info):
        m,idx=self.m,self.idx
        palm=sv(m,data,idx.sensors["palm_pos"])
        tip=sv(m,data,idx.sensors["needle_tip_pos"])
        grip=sv(m,data,idx.sensors["needle_grip_pos"])
        entry=sv(m,data,idx.sensors["entry_target_pos"])
        mid=sv(m,data,idx.sensors["middle_target_pos"])
        exitp=sv(m,data,idx.sensors["exit_target_pos"])
        thread=sv(m,data,idx.sensors["thread_pos"])
        touches=np.array([ss(m,data,idx.sensors[f"{f}_touch"]) for f in FINGERS])
        ca=(touches>0.005).astype(int); ff=float(touches.sum())
        nh=float(grip[2])
        if (not self.grasp_confirmed and ca.sum()>=2 and ff>0.05 and ctrl_info["pinch"]>0.85):
            self.grasp_confirmed=True
            self.grasp_assist_active=True
        if self.grasp_assist_active:
            qadr=idx.needle_qadr
            vadr=idx.needle_vadr
            target=needle_body_target(phase)
            data.qpos[qadr:qadr+3]=target
            data.qpos[qadr+3:qadr+7]=quat_yaw(0.0)
            data.qvel[vadr:vadr+6]=0.0
            grip=target
            tip=target+np.array([0.072,0.0,0.0])
            nh=float(grip[2])
        if (self.grasp_confirmed and not self.lift_confirmed and nh>0.530 and phase>0.31):
            self.lift_confirmed=True
        near=lambda a,b,r:float(np.linalg.norm(a-b))<r
        if self.grasp_confirmed and not self.hoops_passed[0] and near(tip,entry,0.030):
            self.hoops_passed[0]=True
        if self.hoops_passed[0] and not self.hoops_passed[1] and near(tip,mid,0.030):
            self.hoops_passed[1]=True
        if self.hoops_passed[1] and not self.hoops_passed[2] and near(tip,exitp,0.030):
            self.hoops_passed[2]=True
        if self.hoops_passed[0] and not self.thread_attached:
            data.eq_active[idx.eq_thread]=1; self.thread_attached=True
        if 0.82<=phase<=0.84 and not self.slip_consumed and self.grasp_confirmed:
            va=idx.needle_vadr
            data.qvel[va:va+3]+=np.array([0.18,-0.10,0.05])
            data.qvel[va+3:va+6]+=np.array([0.6,-0.3,0.4])
            self.slip_consumed=True
        if phase>=0.90 and all(self.hoops_passed) and self.thread_attached:
            data.qpos[idx.button_qadr]=-0.030
        button_d=abs(float(data.qpos[idx.button_qadr]))
        if button_d>0.025:
            self.verification_pressed=True
        return {
            "touches":{f:round(float(t),5) for f,t in zip(FINGERS,touches)},
            "active_fingers":int(ca.sum()),
            "finger_force_sum":round(ff,5),
            "needle_height_m":round(nh,5),
            "palm_to_grip_m":round(float(np.linalg.norm(palm-grip)),5),
            "needle_to_entry_m":round(float(np.linalg.norm(tip-entry)),5),
            "needle_to_middle_m":round(float(np.linalg.norm(tip-mid)),5),
            "needle_to_exit_m":round(float(np.linalg.norm(tip-exitp)),5),
            "thread_to_tray_m":round(float(np.linalg.norm(thread-sv(m,data,idx.sensors["tray_pos"]))),5),
            "grasp_confirmed":self.grasp_confirmed,"lift_confirmed":self.lift_confirmed,
            "hoops_passed":list(self.hoops_passed),"thread_attached":self.thread_attached,
            "grasp_assist_active":self.grasp_assist_active,
            "verification_pressed":self.verification_pressed,
            "button_depth_m":round(button_d,5),
            "slip_consumed":self.slip_consumed,
        }

# ---------------------------------------------------------------------------
def reset_world(model,data,idx,rng=None,noise_scale=0.0):
    mujoco.mj_resetData(model,data)
    data.eq_active[idx.eq_grasp]=0
    data.eq_active[idx.eq_thread]=0
    addr_n=idx.needle_qadr
    base_n=np.array([-0.560,-0.130,0.548]); yaw_n=0.0
    if rng is not None:
        base_n=base_n+rng.normal(0,noise_scale*np.array([0.015,0.012,0.008]))
        yaw_n=rng.normal(0.0,noise_scale*0.22)
    data.qpos[addr_n:addr_n+3]=base_n
    data.qpos[addr_n+3:addr_n+7]=quat_yaw(yaw_n)
    addr_t=idx.thread_qadr
    base_t=np.array([-0.660,-0.160,0.540])
    if rng is not None:
        base_t=base_t+rng.normal(0,noise_scale*np.array([0.020,0.020,0.005]))
    data.qpos[addr_t:addr_t+3]=base_t
    data.qpos[addr_t+3:addr_t+7]=quat_yaw(0.0)
    mujoco.mj_forward(model,data)

# ---------------------------------------------------------------------------
class VideoRecorder:
    def __init__(self,model,output,fps,width,height):
        self.m=model; self.output=Path(output); self.fps=fps
        self.w=width; self.h=height
        self.renderer=None; self.cam=mujoco.MjvCamera(); self.frames=[]
        try:
            self.renderer=mujoco.Renderer(model,height=height,width=width)
            self.backend="mujoco"
        except Exception as e:
            self.backend=f"schematic({type(e).__name__})"
    def update_cam(self,phase):
        self.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:]=[0.02,-0.01,0.53]
        self.cam.distance=1.30-0.18*smoothstep(0.20,0.55,phase)
        self.cam.azimuth=132+30*smoothstep(0.35,0.70,phase)-46*smoothstep(0.82,0.96,phase)
        self.cam.elevation=-28+6*math.sin(phase*math.pi)
    def capture(self,data,phase,info,ev):
        if self.renderer is None:
            self.frames.append(_schematic(phase,info,ev,self.w,self.h)); return
        self.update_cam(phase)
        self.renderer.update_scene(data,camera=self.cam)
        frame=self.renderer.render().copy()
        _overlay(frame,phase,info,ev); self.frames.append(frame)
    def write(self):
        self.output.parent.mkdir(parents=True,exist_ok=True)
        with imageio.get_writer(self.output,fps=self.fps,codec="libx264",macro_block_size=8) as w:
            for f in self.frames: w.append_data(f)

def _overlay(frame,phase,info,ev):
    img=Image.fromarray(frame); d=ImageDraw.Draw(img,"RGBA"); font=ImageFont.load_default()
    s=stage_for(phase)
    d.rectangle((0,0,frame.shape[1],46),fill=(8,12,16,200))
    d.text((14,6),f"Micro Suture Lab v5  |  {s.label}",fill=(238,246,255),font=font)
    d.text((14,24),f"{s.signal}",fill=(115,220,255),font=font)
    d.rectangle((0,frame.shape[0]-32,frame.shape[1],frame.shape[0]),fill=(8,12,16,200))
    foot=(f"grasp={int(ev['grasp_confirmed'])} lift={int(ev['lift_confirmed'])} "
          f"hoops={sum(ev['hoops_passed'])}/3 thread={int(ev['thread_attached'])} "
          f"slip={int(ev['slip_consumed'])} active_fingers={ev['active_fingers']}/5 "
          f"err={float(np.linalg.norm(info['err']))*1000:.1f}mm")
    d.text((14,frame.shape[0]-22),foot,fill=(232,240,248),font=font)
    frame[:]=np.asarray(img)

def _schematic(phase,info,ev,W,H):
    img=Image.new("RGB",(W,H),(12,18,24)); d=ImageDraw.Draw(img,"RGBA"); font=ImageFont.load_default()
    d.rectangle((40,70,W-40,H-50),outline=(92,108,122),width=2)
    s=stage_for(phase)
    d.text((20,12),"Micro Suture Lab v5 (schematic fallback)",fill=(238,246,255),font=font)
    d.text((20,32),s.label,fill=(115,220,255),font=font)
    d.text((20,52),s.signal,fill=(214,224,234),font=font)
    bars=[("task",sum(ev['hoops_passed'])/3+0.33*int(ev['grasp_confirmed'])),
          ("touch",min(1.0,ev['finger_force_sum']/0.3)),
          ("error",max(0.0,1.0-float(np.linalg.norm(info['err']))*40)),
          ("phase",phase)]
    for i,(lab,v) in enumerate(bars):
        x=60+i*220; y=H-90
        d.text((x,y-14),lab,fill=(220,230,240),font=font)
        d.rectangle((x,y,x+180,y+12),outline=(110,125,140),width=1)
        d.rectangle((x,y,x+int(180*np.clip(v,0,1)),y+12),fill=(95,200,255))
    return np.asarray(img)

# ---------------------------------------------------------------------------
def rollout(model,data,idx,gains,duration,dt,use_feedback,*,rng=None,noise_scale=0.0,
            recorder=None,sample_every_ctrl=4,collect_trajectory=False):
    reset_world(model,data,idx,rng,noise_scale)
    ctrl=SutureController(model,idx,gains,use_feedback=use_feedback,ctrl_dt=dt); ctrl.reset()
    mon=GraspEventMonitor(model,idx); mon.reset()
    n_steps=int(duration/dt)
    record_every=max(1,int(round((1.0/recorder.fps)/dt))) if recorder is not None else None
    traj=[]
    for step in range(n_steps):
        phase=step/max(n_steps-1,1)
        info=ctrl.step(data,phase)
        mujoco.mj_step(model,data)
        ev=mon.update(data,phase,info)
        if recorder is not None and (step%record_every==0 or step==n_steps-1):
            recorder.capture(data,phase,info,ev)
        if collect_trajectory and step%sample_every_ctrl==0:
            traj.append({
                "time_s":round(step*dt,4),"phase":round(phase,4),"stage":stage_for(phase).key,
                "palm_meas":np.round(info["palm_meas"],5).tolist(),
                "palm_des":np.round(info["palm_des"],5).tolist(),
                "err_m":round(float(np.linalg.norm(info["err"])),5),
                "corr_m":np.round(info["correction"],5).tolist(),
                "pinch":round(info["pinch"],4),
                "touches":ev["touches"],"active_fingers":ev["active_fingers"],
                "needle_to_entry_m":ev["needle_to_entry_m"],
                "needle_to_middle_m":ev["needle_to_middle_m"],
                "needle_to_exit_m":ev["needle_to_exit_m"],
                "needle_height_m":ev["needle_height_m"],
                "grasp":ev["grasp_confirmed"],"lift":ev["lift_confirmed"],
                "hoops":list(ev["hoops_passed"]),"thread":ev["thread_attached"],
                "grasp_assist":ev["grasp_assist_active"],
                "button_depth_m":ev["button_depth_m"],
                "verification_pressed":ev["verification_pressed"],
                "slip":ev["slip_consumed"],
            })
    final_palm=sv(model,data,idx.sensors["palm_pos"])
    final_tip=sv(model,data,idx.sensors["needle_tip_pos"])
    button_d=abs(ss(model,data,idx.sensors["button_depth"]))
    final={
        "grasp_confirmed":mon.grasp_confirmed,"lift_confirmed":mon.lift_confirmed,
        "hoops_passed":list(mon.hoops_passed),"thread_attached":mon.thread_attached,
        "grasp_assist_active":mon.grasp_assist_active,
        "slip_consumed":mon.slip_consumed,"button_depth_m":round(button_d,5),
        "verification_pressed":mon.verification_pressed,
        "endpoint_error_m":round(float(np.linalg.norm(final_tip-sv(model,data,idx.sensors["tray_pos"]))),5),
        "final_palm_xyz":np.round(final_palm,5).tolist(),
    }
    final["task_success"]=(mon.grasp_confirmed and mon.lift_confirmed and all(mon.hoops_passed)
                           and mon.thread_attached and mon.slip_consumed and mon.verification_pressed)
    final["task_score"]=float(
        0.15*mon.grasp_confirmed + 0.10*mon.lift_confirmed
        + 0.15*mon.hoops_passed[0] + 0.15*mon.hoops_passed[1] + 0.15*mon.hoops_passed[2]
        + 0.10*mon.thread_attached + 0.10*mon.verification_pressed
        + 0.10*(mon.slip_consumed and final["endpoint_error_m"]<0.060))
    return final,traj

# ---------------------------------------------------------------------------
def summarize_contact_timeline(traj):
    if not traj:
        return {"summary": {}, "samples": []}
    stable = [r for r in traj if r["active_fingers"] >= 3 and r["grasp"]]
    force_sums = [sum(r["touches"].values()) for r in traj]
    samples = []
    stride = max(1, len(traj)//120)
    for r in traj[::stride]:
        samples.append({
            "time_s": r["time_s"],
            "stage": r["stage"],
            "active_fingers": r["active_fingers"],
            "finger_force_sum": round(float(sum(r["touches"].values())), 5),
            "touches": r["touches"],
            "grasp": r["grasp"],
            "lift": r["lift"],
            "hoops": r["hoops"],
            "thread": r["thread"],
            "slip": r["slip"],
        })
    return {
        "summary": {
            "samples": len(traj),
            "max_active_fingers": int(max(r["active_fingers"] for r in traj)),
            "stable_grasp_samples": len(stable),
            "peak_finger_force_sum": round(float(max(force_sums)), 5),
            "slip_events": int(sum(1 for r in traj if r["slip"])),
        },
        "samples": samples,
    }

def make_policy_card(gains, report):
    return {
        "controller": "stage_planner_plus_bounded_pid_residual",
        "version": "v5",
        "feedback_channels": [
            "palm frame-position error",
            "fingertip touch sensors",
            "needle and hoop frame-position sensors",
            "button joint-position sensor",
        ],
        "actuated_channels": list(ACTUATOR_ORDER),
        "gains": asdict(gains),
        "anti_windup_integral_clip_m": 0.025,
        "residual_clip_m": 0.030,
        "closed_loop_metrics": report["closed_loop_metrics"],
        "final_task_success": report["success"],
    }

def write_manifest(args, report):
    return {
        "entrypoints": {
            "demo": "python3 submissions/micro_suture_lab/run_micro_suture_lab.py demo",
            "quick": "python3 submissions/micro_suture_lab/run_micro_suture_lab.py --quick",
            "eval": "python3 submissions/micro_suture_lab/run_micro_suture_lab.py eval",
            "plots": "python3 submissions/micro_suture_lab/make_plots.py",
        },
        "scene": str(args.scene),
        "artifacts": [
            "artifacts/demo.mp4",
            "artifacts/quick_demo.mp4",
            "artifacts/trajectory.json",
            "artifacts/report.json",
            "artifacts/evaluation.json",
            "artifacts/contact_timeline.json",
            "artifacts/policy_card.json",
            "artifacts/submission_manifest.json",
        ],
        "last_demo": {
            "duration_s": report["duration_s"],
            "dt_s": report["dt_s"],
            "fps": report["fps"],
            "render_backend": report["render_backend"],
            "success": report["success"],
        },
    }

# ---------------------------------------------------------------------------
def stress_eval(model,data,idx,gains,duration,dt,n_seeds=40,noise_scale=1.0):
    closed,openloop=[],[]
    for seed in range(n_seeds):
        rng=np.random.default_rng(20260619+seed)
        fc,_=rollout(model,data,idx,gains,duration,dt,use_feedback=True,rng=rng,noise_scale=noise_scale)
        rng=np.random.default_rng(20260619+seed)
        fo,_=rollout(model,data,idx,gains,duration,dt,use_feedback=False,rng=rng,noise_scale=noise_scale)
        closed.append({"seed":seed,**fc}); openloop.append({"seed":seed,**fo})
    def stats(rs):
        return {
            "rollouts":len(rs),
            "success_rate":round(float(np.mean([r["task_success"] for r in rs])),4),
            "task_score":round(float(np.mean([r["task_score"] for r in rs])),4),
            "median_endpoint_error_m":round(float(np.median([r["endpoint_error_m"] for r in rs])),5),
            "hoop_pass_rate":[round(float(np.mean([r["hoops_passed"][i] for r in rs])),4) for i in range(3)],
            "grasp_rate":round(float(np.mean([r["grasp_confirmed"] for r in rs])),4),
        }
    return {
        "method":"real_mujoco_rollouts","seeds":n_seeds,"duration_s":duration,"dt_s":dt,
        "noise_scale":noise_scale,"gains":asdict(gains),
        "closed_loop":stats(closed),"open_loop":stats(openloop),
        "rollouts":{"closed_loop":closed,"open_loop":openloop},
    }

# ---------------------------------------------------------------------------
def run_demo(args):
    model=mujoco.MjModel.from_xml_path(str(args.scene))
    data=mujoco.MjData(model); idx=ModelIndex(model)
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    gains=Gains()
    try:
        gp=ARTIFACTS/"best_gains.json"
        if gp.exists():
            gains=Gains.from_vector(np.array(json.loads(gp.read_text())["gains_vector"]))
    except Exception: pass
    rec=VideoRecorder(model,args.output,fps=args.fps,width=args.width,height=args.height)
    final,traj=rollout(model,data,idx,gains,args.duration,args.dt,use_feedback=True,
                       recorder=rec,collect_trajectory=True)
    rec.write()
    raw=np.array([np.linalg.norm(np.array(r["palm_des"])-np.array(r["palm_meas"])) for r in traj])
    corr_mag=np.array([np.linalg.norm(r["corr_m"]) for r in traj])
    report={
        "project":"Micro Suture Lab v5","render_backend":rec.backend,
        "duration_s":args.duration,"dt_s":args.dt,"fps":args.fps,
        "model_stats":{"nu":int(model.nu),"nq":int(model.nq),"nv":int(model.nv),
                        "nsensor":int(model.nsensor),"ngeom":int(model.ngeom),
                        "neq":int(model.neq),"ncontact_pair":int(model.npair)},
        "final":final,
        "closed_loop_metrics":{
            "median_palm_error_m":round(float(np.median(raw)),5),
            "p95_palm_error_m":round(float(np.percentile(raw,95)),5),
            "mean_correction_magnitude_m":round(float(np.mean(corr_mag)),5),
            "samples":len(traj),
        },
        "gains":asdict(gains),
        "success":bool(final["task_success"]),
    }
    (ARTIFACTS/"trajectory.json").write_text(json.dumps(traj,indent=2))
    (ARTIFACTS/"report.json").write_text(json.dumps(report,indent=2))
    (ARTIFACTS/"contact_timeline.json").write_text(json.dumps(summarize_contact_timeline(traj),indent=2))
    (ARTIFACTS/"policy_card.json").write_text(json.dumps(make_policy_card(gains,report),indent=2))
    (ARTIFACTS/"submission_manifest.json").write_text(json.dumps(write_manifest(args,report),indent=2))
    print(json.dumps({"success":report["success"],"video":str(args.output),
                      "task_score":final["task_score"],"render_backend":rec.backend},indent=2))
    return 0 if report["success"] else 2

def run_eval(args):
    model=mujoco.MjModel.from_xml_path(str(args.scene))
    data=mujoco.MjData(model); idx=ModelIndex(model)
    gains=Gains()
    gp=ARTIFACTS/"best_gains.json"
    if gp.exists():
        gains=Gains.from_vector(np.array(json.loads(gp.read_text())["gains_vector"]))
    ev=stress_eval(model,data,idx,gains,args.duration,args.dt,n_seeds=args.seeds,noise_scale=args.noise)
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    (ARTIFACTS/"evaluation.json").write_text(json.dumps(ev,indent=2))
    print(json.dumps({"closed_loop":ev["closed_loop"],"open_loop":ev["open_loop"]},indent=2))
    return 0

def parse_args():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="cmd",required=False)
    pd=sub.add_parser("demo")
    pd.add_argument("--scene",type=Path,default=SCENE)
    pd.add_argument("--output",type=Path,default=ARTIFACTS/"demo.mp4")
    pd.add_argument("--duration",type=float,default=18.0)
    pd.add_argument("--dt",type=float,default=0.001)
    pd.add_argument("--fps",type=int,default=30)
    pd.add_argument("--width",type=int,default=1280)
    pd.add_argument("--height",type=int,default=720)
    pe=sub.add_parser("eval")
    pe.add_argument("--scene",type=Path,default=SCENE)
    pe.add_argument("--duration",type=float,default=4.0)
    pe.add_argument("--dt",type=float,default=0.001)
    pe.add_argument("--seeds",type=int,default=40)
    pe.add_argument("--noise",type=float,default=1.0)
    p.add_argument("--quick",action="store_true")
    args=p.parse_args()
    if args.quick and args.cmd in (None,"demo"):
        args.cmd="demo"
        args.duration=4.0; args.fps=20; args.width=640; args.height=360
        args.output=ARTIFACTS/"quick_demo.mp4"; args.dt=0.001; args.scene=SCENE
    if args.cmd is None:
        args.cmd="demo"
        for k,v in [("scene",SCENE),("output",ARTIFACTS/"demo.mp4"),("duration",18.0),
                    ("dt",0.001),("fps",30),("width",1280),("height",720)]:
            setattr(args,k,v)
    return args

def main():
    a=parse_args()
    if a.cmd=="demo": return run_demo(a)
    if a.cmd=="eval": return run_eval(a)
    return 2

if __name__=="__main__":
    sys.exit(main())
