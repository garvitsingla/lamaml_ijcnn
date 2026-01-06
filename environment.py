from minigrid.envs.babyai.goto import GoToLocal, GoTo, GoToObjDoor, GoToObj
from minigrid.envs.babyai.open import Open
from minigrid.envs.babyai.other import ActionObjDoor
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.putnext import PutNextLocal
from minigrid.envs.babyai.core.verifier import GoToInstr, ObjDesc, OpenInstr, PickupInstr, BeforeInstr, AfterInstr, PutNextInstr
from minigrid.core.world_object import Ball, Box, Key
import re
import numpy as np

OBJECTS = ['ball', 'key']
# OBJECTS_ALL = ['ball', 'key','box']

COLORS = ['red', 'green', 'blue', 'purple']
# COLORS_ALL = ['red', 'green', 'blue', 'purple','yellow', 'grey']

PREP_LOCS = ['on', 'at', 'to']

# Location names
# LOC_NAMES_ALL = ['left', 'behind','right', 'front']
LOC_NAMES = ['left', 'behind']

# For GoToLocal
LOCAL_MISSIONS = [f"go to the {color} {obj}" for color in COLORS for obj in OBJECTS]
# LOCAL_MISSIONS_VOCAB = [f"go to the {color} {obj}" for color in COLORS_ALL for obj in OBJECTS_ALL]

# For Pickup
PICKUP_MISSIONS = [f"pick up the {color} {obj}" for color in COLORS for obj in OBJECTS]
# PICKUP_MISSIONS_VOCAB = [f"pick up the {color} {obj}" for color in COLORS_ALL for obj in OBJECTS_ALL]
 
# For GoToObjDoor, GoToOpen environments  
DOOR_MISSIONS = [f"go to the {color} door" for color in COLORS]
# DOOR_MISSIONS_VOCAB = [f"go to the {color} door" for color in COLORS_ALL]

# For OpenDoor Environment
OPEN_DOOR_MISSIONS = [f"open the {color} door" for color in COLORS]
# OPEN_DOOR_MISSIONS_VOCAB = [f"open the {color} door" for color in COLORS_ALL]

# For OpenDoorLoc Environment
DOOR_LOC_MISSIONS = [f"open the door {prep} the {loc}" for prep in PREP_LOCS for loc in LOC_NAMES]
# DOOR_LOC_MISSIONS_VOCAB = [f"open the door {prep} the {loc}" for prep in PREP_LOCS for loc in LOC_NAMES]

# For OpenDoorsOrder
OPEN_DOORS_ORDER_MISSIONS = (
    [f"open the {c1} door" for c1 in COLORS] +
    [f"open the {c1} door, then open the {c2} door" for c1 in COLORS for c2 in COLORS] +
    [f"open the {c1} door after you open the {c2} door" for c1 in COLORS for c2 in COLORS]
)
# OPEN_DOORS_ORDER_MISSIONS_VOCAB = (
#     [f"open the {c1} door" for c1 in COLORS_ALL] +
#     [f"open the {c1} door, then open the {c2} door" for c1 in COLORS_ALL for c2 in COLORS_ALL] +
#     [f"open the {c1} door after you open the {c2} door" for c1 in COLORS_ALL for c2 in COLORS_ALL]
# )

# For ActionObjDoor Environment
ACTION_OBJ_DOOR_MISSIONS = (
    [f"pick up the {c} ball" for c in COLORS] +
    [f"pick up the {c} key"  for c in COLORS] +
    [f"go to the {c} ball"   for c in COLORS] +
    [f"go to the {c} key"    for c in COLORS] +
    [f"go to the {c} door"   for c in COLORS] +
    [f"open a {c} door"      for c in COLORS]
)
# ACTION_OBJ_DOOR_MISSIONS_VOCAB = (
#     [f"pick up the {c} ball" for c in COLORS_ALL] +
#     [f"pick up the {c} key"  for c in COLORS_ALL] +
#     [f"pick up the {c} box"  for c in COLORS_ALL] +
#     [f"go to the {c} box"    for c in COLORS_ALL] +
#     [f"go to the {c} ball"   for c in COLORS_ALL] +
#     [f"go to the {c} key"    for c in COLORS_ALL] +
#     [f"go to the {c} door"   for c in COLORS_ALL] +
#     [f"open a {c} door"      for c in COLORS_ALL]
# )


# For PutNextLocal Environment
def _aug_phrase(c1, t1, c2, t2):
    return f"put the {c1} {t1} next to the {c2} {t2}"
PUTNEXT_MISSIONS = [
    _aug_phrase(c1, t1, c2, t2)
    for c1 in COLORS for t1 in OBJECTS
    for c2 in COLORS for t2 in OBJECTS
    if not (c1 == c2 and t1 == t2)
]
# PUTNEXT_MISSIONS_VOCAB = [
#     _aug_phrase(c1, t1, c2, t2)
#     for c1 in COLORS_ALL for t1 in OBJECTS_ALL
#     for c2 in COLORS_ALL for t2 in OBJECTS_ALL
#     if not (c1 == c2 and t1 == t2)
# ]


class GoToLocalMissionEnv(GoToLocal):
    def __init__(self, room_size=6, num_dists=2, **kwargs):
        super().__init__(room_size=room_size, num_dists=num_dists, **kwargs)
        self._forced_mission = None
        # self.render_mode = kwargs.get('render_mode', 'human')

    def set_forced_mission(self, mission):
        self._forced_mission = mission

    def gen_mission(self):
        
        if self._forced_mission is not None:
            m = re.match(r"go to the (\w+) (\w+)", self._forced_mission) 
            if m:
                color, obj_type = m.groups()
                self.agent_pos = np.array((1,1))
                
                # Map string to class
                OBJ_CLASS = {"ball": Ball, "box": Box, "key": Key}
                target_obj = OBJ_CLASS[obj_type](color)

                i = self._rand_int(0, self.num_cols)
                j = self._rand_int(0, self.num_rows)
                target_obj, _ = self.add_object(i, j, obj_type, color)

                # Add distractors 
                distractor_count = 0
                choices = [(t, c) for t in OBJECTS for c in COLORS if not (t == obj_type and c == color)]
                while distractor_count < self.num_dists:
                    dist_type, dist_color = self._rand_elem(choices)
                    try:
                        i = self._rand_int(0, self.num_cols)
                        j = self._rand_int(0, self.num_rows)
                        self.add_object(i, j, dist_type, dist_color)
                        distractor_count += 1
                    except Exception:
                        continue
                self.instrs = GoToInstr(ObjDesc(obj_type, color))
                self.check_objs_reachable()
            else:
                super().gen_mission()
        else:
            super().gen_mission()


class GoToObjMissionEnv(GoToObj):
    """
    Go to an object in a single room with NO distractors.
    This is the simplest navigation task - matches original BabyAI GoToObj.
    Compatible with the LA-MAML pipeline.
    """
    def __init__(self, room_size=8, **kwargs):
        super().__init__(room_size=room_size, **kwargs)
        self._forced_mission = None
        # self.render_mode = kwargs.get('render_mode', 'human')

    def set_forced_mission(self, mission):
        self._forced_mission = mission

    def gen_mission(self):
        if self._forced_mission is not None:
            m = re.match(r"go to the (\w+) (\w+)", self._forced_mission)
            if m:
                color, obj_type = m.groups()
                
                # Place agent
                self.place_agent()
                
                # Add ONLY the target object (no distractors)
                target_obj, _ = self.add_object(0, 0, obj_type, color)
                
                # Set instruction
                self.instrs = GoToInstr(ObjDesc(obj_type, color))
                return

            else:
                super().gen_mission()
        else:
            super().gen_mission()



class PickupDistMissionEnv(RoomGridLevel):
    def __init__(self, debug=False, room_size=7, num_dists=5,
                 num_rows=1, num_cols=1, max_steps=200, **kwargs):
        self.debug = debug
        super().__init__(num_rows=num_rows, num_cols=num_cols,
                         max_steps=max_steps, room_size=room_size, **kwargs)
        self.num_dists = num_dists
        self.fixed_max_steps = True
        self._forced_mission = None

    def set_forced_mission(self, mission: str):
        self._forced_mission = mission

    def gen_mission(self):
        if self._forced_mission is not None:
            m = re.match(r"pick up (a|the) (?:(\w+)\s)?(\w+)?",
                         self._forced_mission)
            if m:
                _, color, obj_type = m.groups()
                color    = color if color not in (None, "box", "ball", "key") else None
                obj_type = obj_type if obj_type in ("box", "ball", "key") else None

                self.place_agent(0, 0)

                placed = False
                if obj_type is not None and color is not None:
                    attempts = 0
                    while not placed and attempts < 50:
                        try:
                            x = self._rand_int(0, self.num_cols)
                            y = self._rand_int(0, self.num_rows)
                            self.add_object(x, y, obj_type, color)
                            placed = True
                        except Exception:
                            attempts += 1

                distractors_to_add = max(self.num_dists, 0)
                choices = [(t, c) for t in OBJECTS for c in COLORS]
                if obj_type is not None and color is not None:
                    choices = [(t, c) for (t, c) in choices
                               if not (t == obj_type and c == color)]
                distractor_count = 0
                while distractor_count < distractors_to_add:
                    dist_type, dist_color = self._rand_elem(choices)
                    try:
                        x = self._rand_int(0, self.num_cols)
                        y = self._rand_int(0, self.num_rows)
                        self.add_object(x, y, dist_type, dist_color)
                        distractor_count += 1
                    except Exception:
                        continue

                if not placed:
                    t_choice = obj_type if obj_type is not None else self._rand_elem(OBJECTS)
                    c_choice = color    if color    is not None else self._rand_elem(COLORS)
                    try:
                        x = self._rand_int(0, self.num_cols)
                        y = self._rand_int(0, self.num_rows)
                        self.add_object(x, y, t_choice, c_choice)
                    except Exception:
                        pass

                self.instrs = PickupInstr(ObjDesc(obj_type, color), strict=self.debug)
                self.mission = self.instrs.surface(self)
                return

        objs = self.add_distractors(num_distractors=self.num_dists)
        self.place_agent(0, 0)
        obj = self._rand_elem(objs)
        type_, color_ = obj.type, obj.color

        select_by = self._rand_elem(["type", "color", "both"])
        if select_by == "color":
            type_ = None
        elif select_by == "type":
            color_ = None

        self.instrs = PickupInstr(ObjDesc(type_, color_), strict=self.debug)
        self.mission = self.instrs.surface(self)
        return

    def _matches_desc(self, obj, desc):
        if obj is None:
            return False
        if (desc.type  is not None) and (obj.type  != desc.type):
            return False
        if (desc.color is not None) and (obj.color != desc.color):
            return False
        return True

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        carrying = getattr(self, "carrying", None)
        info["carrying"] = carrying

        if not terminated and isinstance(self.instrs, PickupInstr):
            if self._matches_desc(carrying, self.instrs.desc):
                terminated = True
                info["success"] = True
                if reward == 0:
                    try:
                        reward = self._reward()
                    except Exception:
                        pass

        return obs, reward, terminated, truncated, info


class GoToObjDoorMissionEnv(GoToObjDoor):
    def __init__(self,num_distractors=3, **kwargs):
        super().__init__( **kwargs)
        self._forced_mission = None
        self.num_distractors = num_distractors
        # self.render_mode = kwargs.get('render_mode', 'human')

    def set_forced_mission(self, mission):
        self._forced_mission = mission

    def gen_mission(self):

        if self._forced_mission is not None:
            m = re.match(r"go to (a|the) (\w+) (\w+)", self._forced_mission)
            if m:
                _, color, obj_type = m.groups()
                self.place_agent(1, 1)
                objs = []
                if obj_type == "door":
                    # Pick up to 4 door colors, one is the target, rest are distractors
                    distractor_colors = [c for c in COLORS if c != color]
                    all_door_colors = [color] + distractor_colors
                    all_door_colors = all_door_colors[:4]

                    # Shuffle target to change position
                    self.np_random.shuffle(all_door_colors)
                    for dcolor in all_door_colors:
                        door, _ = self.add_door(1, 1, color=dcolor)
                        objs.append(door)

                else:
                    obj, _ = self.add_object(1, 1, obj_type, color)
                    objs.append(obj)

                    # Add distractors
                    distractor_count = 0
                    while distractor_count < self.num_distractors:
                        distractor_color = self._rand_elem([c for c in COLORS if c != color])
                        distractor_type = self._rand_elem([t for t in OBJECTS if t != obj_type])
                        try:
                            self.add_object(1, 1, distractor_type, distractor_color)
                            distractor_count += 1
                        except Exception:
                            continue

                self.check_objs_reachable()
                self.instrs = GoToInstr(ObjDesc(obj_type, color))
            else:
                super().gen_mission()
        else:
            super().gen_mission()




class GoToOpenMissionEnv(GoTo):
    def __init__(self, room_size=6, num_rows=2, num_cols=2, num_dists=10, **kwargs):
        super().__init__(room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         num_dists=num_dists,
                         doors_open=True,
                         **kwargs)
        self._forced_mission = None
        # self.render_mode = kwargs.get('render_mode', 'human')

    def set_forced_mission(self, mission):
        self._forced_mission = mission

    def gen_mission(self):
        if self._forced_mission is not None:
            m = re.match(r"go to (a|the) (\w+) (\w+)", self._forced_mission)
            if m:
                _, color, obj_type = m.groups()

                self.place_agent()
                self.connect_all()

                if obj_type == "door":
                # verify at least one door color of forced mission exists
                    doors = []
                    for i in range(self.num_cols):
                        for j in range(self.num_rows):
                            room = self.get_room(i, j)
                            for door in room.doors:
                                if door and door.color == color:
                                    doors.append(door)
                    if not doors:
                        door, _ = self.add_door(0, 0, color=color, locked=False)
                    self.instrs = GoToInstr(ObjDesc("door", color))

                else:
                    # Add target object
                    obj, _ = self.add_object(self._rand_int(0, self.num_cols),
                                            self._rand_int(0, self.num_rows),
                                            obj_type, color)

                    # Add distractors from all except the exact target
                    choices = [(t, c) for t in OBJECTS for c in COLORS
                            if not (t == obj_type and c == color)]
                    placed = 0
                    while placed < self.num_dists:
                        dist_type, dist_color = self._rand_elem(choices)
                        try:
                            self.add_object(self._rand_int(0, self.num_cols),
                                            self._rand_int(0, self.num_rows),
                                            dist_type, dist_color)
                            placed += 1
                        except Exception:
                            continue


                    self.check_objs_reachable()
                    self.instrs = GoToInstr(ObjDesc(obj_type, color))
                    self.open_all_doors()
                    return
        super().gen_mission()


class OpenDoorMissionEnv(Open):
    def __init__(self, debug = False, **kwargs):
        self.debug = debug
        super().__init__(**kwargs)
        self._forced_mission = None
        # self.render_mode = kwargs.get('render_mode', 'human')

    def set_forced_mission(self, mission: str):
        self._forced_mission = mission

    def _valid_sides(self, i: int, j: int):
        # 0=E, 1=S, 2=W, 3=N; only sides that have a neighboring room
        sides = []
        if i + 1 < self.num_cols: sides.append(0)
        if j + 1 < self.num_rows: sides.append(1)
        if i - 1 >= 0:            sides.append(2)
        if j - 1 >= 0:            sides.append(3)
        return sides

    def gen_mission(self):
        i = j = 1
        self.place_agent(i, j)

        color = None
        if self._forced_mission:
            m = re.match(r"\s*open\s+(?:a|the)\s+(\w+)\s+door\s*$", self._forced_mission, re.IGNORECASE)
            if m:
                color = m.group(1).lower()
        if color is None:
            color = self._rand_elem(COLORS)

        # Choose forced color 
        target_color = color
        sides = self._valid_sides(i, j)
        self.np_random.shuffle(sides)

        # Place the target door first
        target_side = sides[0]
        target_door, _ = self.add_door(i, j, door_idx=target_side, color=target_color, locked=False)

        # Place distractor doors on remaining sides with different colors
        distractor_colors = [c for c in COLORS if c != target_color]
        distractor_colors = self._rand_subset(distractor_colors, len(sides) - 1)
        for side, dcol in zip(sides[1:], distractor_colors):
            self.add_door(i, j, door_idx=side, color=dcol, locked=False)

        self.instrs = OpenInstr(ObjDesc(target_door.type, target_color), strict=self.debug)
        self.mission = self.instrs.surface(self)


class OpenDoorLocMissionEnv(Open):
    def __init__(self, debug=False, select_by=None, **kwargs):
        self.debug = debug
        self.select_by = select_by  
        super().__init__(**kwargs)
        self._forced_mission = None
        # self.render_mode = kwargs.get('render_mode', 'human')

    def set_forced_mission(self, mission: str):
        self._forced_mission = mission

    def _valid_sides(self, i: int, j: int):
        # 0=E, 1=S, 2=W, 3=N (only sides that have a neighboring room)
        sides = []
        if i + 1 < self.num_cols: sides.append(0)
        if j + 1 < self.num_rows: sides.append(1)
        if i - 1 >= 0:            sides.append(2)
        if j - 1 >= 0:            sides.append(3)
        return sides

    def gen_mission(self):
        i = j = 1
        self.place_agent(i, j)

        # Selection mode (color vs loc)
        select_by = self.select_by
        forced_color = None
        forced_loc = None
        if self._forced_mission:
            m_color = re.match(r"\s*open\s+(?:a|the)\s+(\w+)\s+door\s*$", self._forced_mission, re.IGNORECASE)
            m_loc   = re.match(r"\s*open\s+(?:a|the)\s+door\s+(?:on|at|to)\s+the\s+(\w+)\s*$", self._forced_mission, re.IGNORECASE)
            if m_color:
                forced_color = m_color.group(1).lower()
                select_by = "color"
            elif m_loc:
                forced_loc = m_loc.group(1).lower()

                select_by = "loc"

        if select_by is None:
            select_by = self._rand_elem(["color", "loc"])

        # Add Doors
        sides = self._valid_sides(i, j)
        self.np_random.shuffle(sides)
        first_door = None 

        if select_by == "color":
            # target forced color
            target_color = forced_color if forced_color is not None else self._rand_elem(COLORS)

            # place target door
            target_side = sides[0]
            target_door, _ = self.add_door(i, j, door_idx=target_side, color=target_color, locked=False)
            first_door = target_door if first_door is None else first_door

            # distractor doors 
            distractor_colors = [c for c in COLORS if c != target_color]
            distractor_colors = self._rand_subset(distractor_colors, max(0, len(sides) - 1))
            for side, dcol in zip(sides[1:], distractor_colors):
                d, _ = self.add_door(i, j, door_idx=side, color=dcol, locked=False)
                if first_door is None: first_door = d

            self.instrs = OpenInstr(ObjDesc(target_door.type, color=target_color), strict=self.debug)

        else:  
            # place doors 
            door_colors = self._rand_subset(COLORS, len(sides))
            placed = []
            for side, col in zip(sides, door_colors):
                d, _ = self.add_door(i, j, door_idx=side, color=col, locked=False)
                placed.append(d)
            if placed:
                first_door = placed[0]

            # pick location: forced or random from LOC_NAMES
            loc = forced_loc if forced_loc in LOC_NAMES else None
            if loc is None:
                loc = forced_loc if forced_loc in LOC_NAMES else self._rand_elem(LOC_NAMES)

            self.instrs = OpenInstr(ObjDesc(first_door.type, loc=loc), strict=self.debug)

        self.mission = self.instrs.surface(self)


class OpenDoorsOrderMissionEnv(Open):
    def __init__(self, num_doors=2, debug=False, room_size=None, max_steps=None, **kwargs):
        assert num_doors >= 1
        self.num_doors = num_doors
        self.debug = debug
        self._forced_mission = None
        if room_size is None:
            room_size = 6
        if max_steps is None:
            max_steps = 20 * (room_size ** 2)
        super().__init__(room_size=room_size, max_steps=max_steps, **kwargs)
        # self.render_mode = kwargs.get('render_mode', 'human')

    def set_forced_mission(self, mission):
        self._forced_mission = mission

    def _parse_forced(self):
        if not self._forced_mission:
            return None
        m1 = re.match(r"^\s*open\s+(?:a|the)\s+(\w+)\s+door\s*$", self._forced_mission, re.IGNORECASE)
        if m1:
            return ("single", m1.group(1).lower(), None)
        m2 = re.match(
            r"^\s*open\s+(?:a|the)\s+(\w+)\s+door\s*,?\s*(?:then|and\s+then)\s+open\s+(?:a|the)\s+(\w+)\s+door\s*$",
            self._forced_mission, re.IGNORECASE
        )
        if m2:
            return ("before", m2.group(1).lower(), m2.group(2).lower())
        m3 = re.match(
            r"^\s*open\s+(?:a|the)\s+(\w+)\s+door\s+after\s+you\s+open\s+(?:a|the)\s+(\w+)\s+door\s*$",
            self._forced_mission, re.IGNORECASE
        )
        if m3:
            return ("after", m3.group(1).lower(), m3.group(2).lower())
        return None

    def gen_mission(self):
        spec = self._parse_forced()
        if spec is None:
            colors = list(self._rand_subset(COLORS, min(self.num_doors, len(COLORS))))
        else:
            mode, c1, c2 = spec
            need = [c1] if mode == "single" else [c1, c2]
            pool = list(COLORS)
            self.np_random.shuffle(pool)
            extras = [c for c in pool if c not in need]
            take = max(0, self.num_doors - len(need))
            colors = need + extras[:take]

        i = j = 1
        self.place_agent(i, j)

        doors = []
        for k in range(self.num_doors):
            d, _ = self.add_door(i, j, color=colors[k], locked=False)
            doors.append(d)

        if spec is None:
            if self.num_doors == 1:
                desc1 = ObjDesc(doors[0].type, doors[0].color)
                self.instrs = OpenInstr(desc1, strict=self.debug)
            else:
                d1, d2 = self._rand_subset(doors, 2)
                desc1 = ObjDesc(d1.type, d1.color)
                desc2 = ObjDesc(d2.type, d2.color)
                if self._rand_int(0, 2) == 0:
                    self.instrs = BeforeInstr(OpenInstr(desc1, strict=self.debug), OpenInstr(desc2, strict=self.debug))
                else:
                    self.instrs = AfterInstr(OpenInstr(desc1, strict=self.debug), OpenInstr(desc2, strict=self.debug))
        else:
            mode, c1, c2 = spec
            if mode == "single":
                d1 = next((d for d in doors if d.color == c1), doors[0])
                self.instrs = OpenInstr(ObjDesc(d1.type, d1.color), strict=self.debug)
            elif mode == "before":
                d1 = next((d for d in doors if d.color == c1), doors[0])
                d2 = next((d for d in doors if d.color == c2), doors[-1 if len(doors) > 1 else 0])
                self.instrs = BeforeInstr(
                    OpenInstr(ObjDesc(d1.type, d1.color), strict=self.debug),
                    OpenInstr(ObjDesc(d2.type, d2.color), strict=self.debug),
                )
            else:
                d1 = next((d for d in doors if d.color == c1), doors[0])
                d2 = next((d for d in doors if d.color == c2), doors[-1 if len(doors) > 1 else 0])
                self.instrs = AfterInstr(
                    OpenInstr(ObjDesc(d1.type, d1.color), strict=self.debug),
                    OpenInstr(ObjDesc(d2.type, d2.color), strict=self.debug),
                )

        self.mission = self.instrs.surface(self)
        self.check_objs_reachable()




class ActionObjDoorMissionEnv(ActionObjDoor):
    def __init__(self, objects=None, obj_colors=None, door_colors=None, num_dists = 3, **kwargs):
        self._forced_mission = None
        super().__init__(**kwargs)
        # self.render_mode = kwargs.get('render_mode', 'human')

        self.objects     = objects     if objects     is not None else OBJECTS
        self.obj_colors  = obj_colors  if obj_colors  is not None else COLORS
        self.door_colors = door_colors if door_colors is not None else COLORS
        self.num_dists = num_dists

    def set_forced_mission(self, mission):
        self._forced_mission = mission

    def _parse_forced(self):
        if not self._forced_mission:
            return None

        # Allow any object type from self.objects + door
        obj_alt = "|".join(map(re.escape, sorted(set(self.objects) | {"door"})))
        m = re.match(
            rf"^\s*(pick\s+up|go\s+to|open)\s+(?:a|the)\s+(\w+)\s+({obj_alt})\s*$",
            self._forced_mission, re.IGNORECASE
        )
        if not m:
            return None

        action = m.group(1).lower().replace(" ", "")
        color  = m.group(2).lower()
        otype  = m.group(3).lower()

        # verify color 
        valid_colors = self.door_colors if otype == "door" else self.obj_colors
        if color not in valid_colors:
            return None

        mode = "pickup" if action == "pickup" else ("goto" if action == "goto" else "open")
        return (mode, color, otype)

    def gen_mission(self):
        self.place_agent(1, 1)
        spec = self._parse_forced()
        objs, doors = [], []

        if spec is None:
            target_kind  = self._rand_elem(self.objects + ['door'])
            target_color = (self._rand_elem(self.door_colors) if target_kind == 'door'
                            else self._rand_elem(self.obj_colors))

            if target_kind == "door":
                target, _ = self.add_door(1, 1, color=target_color, locked=False)
            else:
                target, _ = self.add_object(1, 1, kind=target_kind, color=target_color)
            objs.append(target)

            # Add doors
            candidates = [x for x in self.door_colors if x != target_color]
            num = min(3, len(candidates))
            for c in list(self._rand_subset(candidates, num)):
                try:
                    d, _ = self.add_door(1, 1, color=c, locked=False)
                    doors.append(d); objs.append(d)
                except Exception:
                    pass

            # Add distractors
            for _ in range(self.num_dists):
                k = self._rand_elem(self.objects)
                c = self._rand_elem([x for x in self.obj_colors if not (k == target_kind and x == target_color)])
                try:
                    o, _ = self.add_object(1, 1, kind=k, color=c)
                    objs.append(o)
                except Exception:
                    pass

            desc = ObjDesc(target.type, target.color)
            self.instrs = (OpenInstr if target.type == "door"
                           else (PickupInstr if self._rand_bool() else GoToInstr))(desc)

        else:
            mode, color, otype = spec
            if otype == "door":
                target, _ = self.add_door(1, 1, color=color, locked=False)
                candidates = [x for x in self.door_colors if x != color]
                num = min(3, len(candidates))
                for c in list(self._rand_subset(candidates, num)):
                    try:
                        d, _ = self.add_door(1, 1, color=c, locked=False)
                        doors.append(d)
                    except Exception:
                        pass
                for _ in range(3):
                    k = self._rand_elem(self.objects)
                    c = self._rand_elem(self.obj_colors)
                    try:
                        self.add_object(1, 1, kind=k, color=c)
                    except Exception:
                        pass
            else:
                target, _ = self.add_object(1, 1, kind=otype, color=color)
                for c in list(self._rand_subset(self.door_colors, 3)):
                    try:
                        d, _ = self.add_door(1, 1, color=c, locked=False)
                        doors.append(d)
                    except Exception:
                        pass
                for _ in range(self.num_dists):
                    k = self._rand_elem(self.objects)
                    c = self._rand_elem([x for x in self.obj_colors if not (k == otype and x == color)])
                    try:
                        self.add_object(1, 1, kind=k, color=c)
                    except Exception:
                        pass

            desc = ObjDesc(target.type, target.color)
            self.instrs = (PickupInstr(desc) if mode == "pickup"
                           else (GoToInstr(desc) if mode == "goto" else OpenInstr(desc)))

        self.mission = self.instrs.surface(self)
        self.check_objs_reachable()

        

class PutNextLocalMissionEnv(PutNextLocal):

    def __init__(self, room_size=8,num_dists = 4, max_steps=None, **kwargs):
        super().__init__(room_size=room_size, max_steps=max_steps, **kwargs)
        if max_steps is not None:
            self.max_steps = max_steps 
        self._forced_mission = None
        # self.render_mode = kwargs.get('render_mode', 'human')

    def set_forced_mission(self, mission):
        self._forced_mission = mission

    def _parse_forced_putnext(self, text: str):
        pat = r"""^\s*put\s+the\s+(?P<c1>[a-z]+)\s+(?P<t1>[a-z]+)\s+
              next\s+to\s+the\s+(?P<c2>[a-z]+)\s+(?P<t2>[a-z]+)\s*$"""
        m = re.match(pat, text, flags=re.IGNORECASE | re.X)
        if not m:
            raise ValueError("Incorrect PutNext mission")
                
        c1, t1, c2, t2 = m.group("c1", "t1", "c2", "t2")
        return c1.lower(), t1.lower(), c2.lower(), t2.lower()
    
    def gen_mission(self):
        # If a mission was forced by the wrapper, use it verbatim.
        if getattr(self, "_forced_mission", None):
            c1, t1, c2, t2 = self._parse_forced_putnext(self._forced_mission)
            self.place_agent()
            self.add_object(0, 0, kind=t1, color=c1)
            self.add_object(0, 0, kind=t2, color=c2)
            self.check_objs_reachable()
            self.instrs = PutNextInstr(ObjDesc(t1, c1), ObjDesc(t2, c2))
            self.mission = f"put the {c1} {t1} next to the {c2} {t2}"
            return
        
        return super().gen_mission()




