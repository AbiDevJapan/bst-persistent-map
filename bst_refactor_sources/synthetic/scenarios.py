from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random

from .generator import GeneratedDataset
from .objects import CuboidObject
from .world import Change, Session, WorldSpec


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    target_static_fraction: float = 0.80
    change_probability: float = 0.15
    coherence: str = "mixed"       # coherent | mixed | fragmented
    world_xy_m: float = 100.0
    world_z_m: float = 5.0
    voxel_size_m: float = 0.5


@dataclass(frozen=True)
class ScenarioMetrics:
    target_static_fraction: float
    initial_static_fraction: float
    ever_changed_voxel_fraction: float
    mean_changed_voxels_per_session: float
    static_voxels_initial: int
    dynamic_voxels_initial: int
    initial_occupied_voxels: int
    ever_occupied_voxels: int
    dynamic_object_count: int


def default_scenarios() -> list[ScenarioConfig]:
    return [
        ScenarioConfig("static95", target_static_fraction=0.95),
        ScenarioConfig("static80", target_static_fraction=0.80),
        ScenarioConfig("static50", target_static_fraction=0.50),
        ScenarioConfig("change_low", change_probability=0.03),
        ScenarioConfig("change_base", change_probability=0.15),
        ScenarioConfig("change_high", change_probability=0.60),
        ScenarioConfig("coherent", coherence="coherent"),
        ScenarioConfig("mixed", coherence="mixed"),
        ScenarioConfig("fragmented", coherence="fragmented"),
        ScenarioConfig("voxel_1m", voxel_size_m=1.0),
        ScenarioConfig("voxel_05m", voxel_size_m=0.5),
        ScenarioConfig("voxel_025m", voxel_size_m=0.25),
        ScenarioConfig("world_50m", world_xy_m=50.0),
        ScenarioConfig("world_100m", world_xy_m=100.0),
        ScenarioConfig("world_200m", world_xy_m=200.0),
    ]


def quick_scenarios() -> list[ScenarioConfig]:
    # Reviewer-style smoke set spanning the difficult dimensions without the
    # expensive 0.25 m / 200 m cases.
    return [
        ScenarioConfig("static95", target_static_fraction=0.95),
        ScenarioConfig("static50", target_static_fraction=0.50),
        ScenarioConfig("change_low", change_probability=0.03),
        ScenarioConfig("change_high", change_probability=0.60),
        ScenarioConfig("coherent", coherence="coherent"),
        ScenarioConfig("fragmented", coherence="fragmented"),
    ]


class AdversarialScenarioGenerator:
    """
    Procedural object-level generator for Step 6.

    It deliberately separates the four variables that can favor or hurt the
    adaptive representation: static fraction, temporal change rate, spatial
    coherence, map resolution, and world scale.
    """

    def __init__(self, config: ScenarioConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)
        self.spec = WorldSpec(
            config.world_xy_m,
            config.world_xy_m,
            config.world_z_m,
            config.voxel_size_m,
        )

    def _scale(self) -> float:
        return self.config.world_xy_m / 100.0

    def _static_objects(self) -> list[CuboidObject]:
        s = self._scale()
        W = self.config.world_xy_m
        H = min(3.0, self.config.world_z_m)
        t = max(self.config.voxel_size_m, 0.5 * s)
        objs: list[CuboidObject] = []

        # Two wall-rich structures, scaled with the world. Thin surfaces are
        # intentionally included because they are an adversarial case for
        # isotropic octree merging and a natural case for anisotropic prefixes.
        rects = [
            (0.05*W, 0.40*W, 0.08*W, 0.38*W, "A"),
            (0.08*W, 0.38*W, 0.60*W, 0.88*W, "W"),
        ]
        for x0,x1,y0,y1,prefix in rects:
            area=(x0,x1,y0,y1)
            objs.extend([
                CuboidObject(f"{prefix}_n","wall","static",((x0+x1)/2,y1,H/2),(x1-x0,t,H),area),
                CuboidObject(f"{prefix}_s","wall","static",((x0+x1)/2,y0,H/2),(x1-x0,t,H),area),
                CuboidObject(f"{prefix}_w","wall","static",(x0,(y0+y1)/2,H/2),(t,y1-y0,H),area),
                CuboidObject(f"{prefix}_e","wall","static",(x1,(y0+y1)/2,H/2),(t,y1-y0,H),area),
            ])

        # Internal long walls to create planar/anisotropic structure.
        for j, y in enumerate((0.18*W, 0.28*W, 0.70*W, 0.80*W)):
            objs.append(CuboidObject(
                f"inner_{j}", "wall", "static",
                (0.23*W, y, H/2), (0.24*W, t, H),
                (0.08*W,0.38*W,max(0,y-0.02*W),min(W,y+0.02*W)),
            ))

        # Poles / compact structures.
        for i, y in enumerate((0.10,0.25,0.40,0.55,0.70,0.85)):
            x=0.55*W; yy=y*W
            objs.append(CuboidObject(
                f"pole_{i}","pole","static",(x,yy,H/2),
                (max(t,0.5*s),max(t,0.5*s),H),
                (x-W*0.01,x+W*0.01,yy-W*0.01,yy+W*0.01),
            ))
        return objs

    def _dynamic_size(self, i: int) -> tuple[float,float,float]:
        s = self._scale()
        r = self.config.voxel_size_m
        if self.config.coherence == "coherent":
            return (max(4.0*s,2*r), max(2.0*s,2*r), max(1.5,2*r))
        if self.config.coherence == "fragmented":
            return (max(0.6*s,r), max(0.6*s,r), max(0.6,r))
        # mixed
        if i % 3 == 0:
            return (max(4.0*s,2*r), max(2.0*s,2*r), max(1.5,2*r))
        return (max(1.2*s,r), max(1.0*s,r), max(1.0,r))

    def _make_dynamic_object(self, i: int) -> CuboidObject:
        W=self.config.world_xy_m
        size=self._dynamic_size(i)
        # Open right half of the map. Keeps overlap with static buildings low.
        bounds=(0.60*W,0.96*W,0.05*W,0.95*W)
        sx,sy,sz=size
        margin=max(self.config.voxel_size_m,0.5)
        lo_x,hi_x=bounds[0]+sx/2+margin,bounds[1]-sx/2-margin
        lo_y,hi_y=bounds[2]+sy/2+margin,bounds[3]-sy/2-margin
        if lo_x >= hi_x or lo_y >= hi_y:
            raise ValueError(f"world too small for dynamic object {size}")
        return CuboidObject(
            f"dyn_{i}","dynamic","dynamic",
            (self.rng.uniform(lo_x,hi_x),self.rng.uniform(lo_y,hi_y),max(sz/2,self.config.voxel_size_m/2)),
            size,bounds,True,
        )

    def _objects(self):
        static=self._static_objects()
        static_vox=set()
        for o in static: static_vox |= o.voxels(self.spec)

        f=min(max(self.config.target_static_fraction,0.05),0.995)
        target_dynamic=int(round(len(static_vox)*(1.0-f)/f))
        dynamic=[]; dyn_vox=set(); i=0
        # Add coherent objects until the requested initial occupancy ratio is
        # approximately reached. Overlap is allowed but measured afterwards.
        max_objects=8000
        while len(dyn_vox) < target_dynamic and i < max_objects:
            o=self._make_dynamic_object(i)
            dynamic.append(o); dyn_vox |= o.voxels(self.spec); i += 1
        return static,dynamic,static_vox,dyn_vox

    def generate(self, num_sessions: int = 250) -> tuple[GeneratedDataset, ScenarioMetrics]:
        if num_sessions < 2: raise ValueError("num_sessions must be >= 2")
        self.rng=random.Random(self.seed)
        static,dynamic,static_vox,dyn_vox=self._objects()
        state={o.object_id:o for o in static+dynamic}

        def occupied(st):
            out=set()
            for obj in st.values(): out |= obj.voxels(self.spec)
            return frozenset(out)

        sessions=[];changes=[];history=[]
        occ=occupied(state);sessions.append(Session(0,occ));history.append(dict(state))
        for t in range(1,num_sessions):
            nxt={}
            for oid,obj in state.items():
                if obj.zone != "dynamic" or self.rng.random() >= self.config.change_probability:
                    nxt[oid]=obj; continue
                # Toggling preserves object-level spatial coherence while making
                # the temporal process independently tunable.
                nxt[oid]=obj.toggled()
            state=nxt;new_occ=occupied(state)
            changes.append(Change(t,frozenset(new_occ-occ),frozenset(occ-new_occ)))
            sessions.append(Session(t,new_occ));history.append(dict(state));occ=new_occ

        ds=GeneratedDataset(self.spec,sessions,changes,history);ds.validate()
        ever=set().union(*(s.occupied for s in sessions))
        changed=set()
        for c in changes: changed |= set(c.added);changed |= set(c.removed)
        initial=set(sessions[0].occupied)
        static_initial=set(static_vox) & initial
        dynamic_initial=set(dyn_vox) - set(static_vox)
        metrics=ScenarioMetrics(
            target_static_fraction=self.config.target_static_fraction,
            initial_static_fraction=(len(static_initial)/len(initial) if initial else 0.0),
            ever_changed_voxel_fraction=(len(changed)/len(ever) if ever else 0.0),
            mean_changed_voxels_per_session=(sum(c.changed_count for c in changes)/len(changes) if changes else 0.0),
            static_voxels_initial=len(static_initial),
            dynamic_voxels_initial=len(dynamic_initial),
            initial_occupied_voxels=len(initial),
            ever_occupied_voxels=len(ever),
            dynamic_object_count=len(dynamic),
        )
        return ds,metrics
