"""
Activity taxonomy and sensor/modality filtering for HAR datasets.

classify_label()          — map a raw label string to a canonical activity group
get_dataset_groups()      — {group: [raw_labels]} for one dataset
filter_by_activity()      — datasets that contain a given activity group
filter_by_modality()      — datasets that provide all required modalities
filter_by_sensor()        — datasets that have a given (normalized) sensor location
get_activity_overlap()    — canonical groups shared between two datasets


note:
- currently hardcode activity taxonomy and sensor location mapping, but could be made configurable in the future
"""

import re
from typing import Dict, List, Optional, Set, Tuple
try:
    from .dataset_info import DATASETS
except ImportError:
    from dataset_info import DATASETS

# ---------------------------------------------------------------------------
# Activity taxonomy
# Keys are canonical group names.
# Values are lists of regex patterns matched case-insensitively against the
# raw label string.  Order matters: more specific groups must come before
# groups whose patterns would also match (e.g. stairs_up before walking).
# ---------------------------------------------------------------------------
ACTIVITY_TAXONOMY: Dict[str, List[str]] = {
    # --- Kitchen gestures (OPPORTUNITY) — before work_assembly to avoid \bopen\b clash ---
    "open_door": [r"open[\s_]*door"],
    "close_door": [r"close[\s_]*door"],
    "open_fridge": [r"open[\s_]*fridge"],
    "close_fridge": [r"close[\s_]*fridge"],
    "open_dishwasher": [r"open[\s_]*dishwasher"],
    "close_dishwasher": [r"close[\s_]*dishwasher"],
    "open_drawer": [r"open[\s_]*drawer"],
    "close_drawer": [r"close[\s_]*drawer"],
    "clean_table": [r"clean[\s_]*table"],
    "toggle_switch": [r"toggle[\s_]*switch"],

    # --- Posture transitions — before sitting/standing/lying so "Stand -> Sit" etc. resolve here ---
    "posture_transition": [
        r"->",
        r"\bgetup[\s_]*bed\b", r"\bliedown[\s_]*bed\b",
        r"sit[\s_]*down[\s_]*chair", r"stand[\s_]*up[\s_]*chair",
        r"sit[\s_]*to[\s_]*(stand|lie)", r"stand[\s_]*to[\s_]*(sit|lie|walk)",
        r"lie[\s_]*to[\s_]*(sit|stand)",
    ],

    # --- Stairs (before walking so "Walking Upstairs" doesn't fall into walking) ---
    "stairs_up": [
        r"stairsup", r"stairs[\s_]*up", r"climb[\s_]*stair", r"climbing[\s_]*stair",
        r"climbingstairs", r"ascending[\s_]*stair",
        r"upstairs", r"go[\s_]*upstairs",
        r"climbing[\s_-]*up",
        r"\bup(wards?)?\b",  # LARA "Upwards"
        r"stairsupdown",
    ],
    "stairs_down": [
        r"stairsdown", r"stairs[\s_]*down", r"descend[\s_]*stair",
        r"descending[\s_]*stair", r"climbing[\s_-]*down",
        r"downstairs", r"go[\s_]*downstairs",
        r"\bdown(wards?)?\b",
    ],
    "stairs_generic": [
        r"^\s*stairs\s*$",  # WISDM "Stairs" (ambiguous, no direction)
    ],

    # --- Eating / Drinking (before generic motion patterns) ---
    "eating": [
        r"\beat\b", r"eating", r"meal", r"soup", r"chip", r"pasta", r"sandwich",
    ],
    "drinking": [
        r"drink", r"coffee",
    ],

    # --- Brushing (teeth before hair) ---
    "brushing_teeth": [
        r"brush[\s_]*teeth", r"brushingteeth", r"brush[\s_]*tooth",
    ],
    "brushing_hair": [
        r"brush[\s_]*hair", r"brushing[\s_]*hair", r"comb[\s_]*hair",
    ],

    # --- Locomotion ---
    "running": [
        r"running", r"jogging", r"\bjog\b", r"\brun\b",
    ],
    "walking": [
        r"walk", r"gaitcycle", r"nordic[\s_]*walk", r"\bstep\b",
    ],
    "shuffling": [
        r"shuffl",
    ],
    "cycling": [
        r"cycling", r"elliptical[\s_]*bike", r"\bbike\b",
    ],

    # --- Static postures ---
    "sitting": [
        r"sitting", r"sitdown", r"\bsit\b",
    ],
    "standing": [
        r"standing", r"standup", r"\bstand\b",
    ],
    "lying": [
        r"lying", r"liedown", r"\blie\b", r"\blaying\b",
        r"lyingdown", r"sleeping", r"sleep",
    ],
    "stationary": [
        r"stationary", r"\bstill\b", r"\bidle\b", r"\bcentred?\b",
    ],

    # --- Jumping ---
    "jumping": [
        r"jump", r"rope[\s_]*jump", r"jumpfrontback",
    ],
    "skipping": [
        r"skipping",
    ],

    # --- Transport ---
    "transport_car": [r"\bcar\b"],
    "transport_train": [r"\btrain\b"],
    "transport_bus": [r"\bbus\b"],
    "elevator": [
        r"elevator", r"moving[\s_]*\(elevator\)",
    ],

    # --- Sports ---
    "sports_basketball": [r"basketball"],
    "sports_badminton": [r"badminton"],
    "sports_football": [r"football"],
    "sports_tabletennis": [r"table[\s_]*tennis"],
    "sports_kicking": [r"kicking"],
    "sports_catching": [r"catching"],
    "sports_dribbling": [r"dribbling"],
    "sports_clapping": [r"clap"],
    "sports_rowing": [r"rowing"],

    # --- No freeze (DOG dataset — normal gait, not a pathological state) ---
    "no_freeze": [r"nofreeze", r"no[\s_]*freeze"],

    # --- ADL ---
    "typing": [r"typing", r"type[\s_]*on[\s_]*keyboard", r"\btype\b"],
    "writing": [r"writing", r"\bwrite\b"],
    "ironing": [r"ironing"],
    "vacuum_cleaning": [r"vacuum"],
    "phone_call": [r"phone[\s_]*call", r"phone[\s_]*conversation", r"use[\s_]*telephone", r"\btalk\b"],
    "reading": [r"reading[\s_]*book", r"\bread\b"],
    "watching_tv": [r"watching[\s_]*tv"],
    "washing": [r"washing"],
    "cooking": [r"cooking"],
    "dressing": [r"put[\s_]*on[\s_]*(jacket|shoe|glasses)"],
    "undressing": [r"take[\s_]*off[\s_]*(jacket|shoe|glasses)"],
    "folding_clothes": [r"folding[\s_]*clothes"],
    "dusting": [r"dusting"],
    "blow_nose": [r"blow[\s_]*nose"],
    "sneeze_cough": [r"sneeze", r"cough"],
    "pour_water": [r"pour[\s_]*water"],
    "open_container": [r"open[\s_]*bottle", r"open[\s_]*box"],
    "smoking": [r"smok"],
    "computer_use": [r"using[\s_]*computer"],

    # --- Exercise / rehab ---
    "waist_bend": [
        r"waist[\s_]*bend", r"waistbends", r"lateral[\s_]*bend",
        r"reach[\s_]*heels", r"reach[\s_]*foot",
    ],
    "arm_elevation": [
        r"elevation[\s_]*of[\s_]*arms", r"elevationarms", r"arms[\s_]*inner[\s_]*rot",
        r"shoulders.*rotation", r"arms.*rotation",
    ],
    "trunk_rotation": [
        r"trunk[\s_]*twist", r"waist[\s_]*rotation", r"torsorotation",
        r"upper[\s_]*trunk.*twist",
    ],
    "knee_exercise": [
        r"knees.*breast", r"heels.*backside", r"knees[\s_]*bending",
        r"knees.*bending[\s_]*forward", r"rotation[\s_]*on[\s_]*the[\s_]*knee",
        r"kneesben", r"\bkneesbending\b",
    ],
    "stretching": [
        r"forward[\s_]*stretching", r"repetitive.*stretch",
    ],
    "arm_crossing": [r"crossing[\s_]*of[\s_]*arms"],
    "rehab_exercise": [
        r"knee[\s_-]*roll", r"bridging", r"pelvic[\s_]*tilt", r"\bclam\b",
        r"extension[\s_]*in[\s_]*lying", r"prone[\s_]*punch", r"superman",
    ],
    "exercise_stepper": [r"stepper"],
    "exercise_cross_trainer": [r"cross[\s_]*trainer"],
    "exercise_generic": [r"\bexercise\b"],

    # --- Work / industrial ---
    # Standalone "Open"/"Close"/"Pick" (OPENPACK) — after kitchen gestures so
    # "Open Door 1" etc. are already claimed and won't reach here.
    "work_assembly": [
        r"\bassemble\b", r"\binsert\b", r"\bscan\b", r"\bpress\b", r"\bput\b",
        r"^\s*open\s*$", r"^\s*close\s*$", r"^\s*pick\s*$",
    ],
    "painting_work": [r"roll[\s_]*paint", r"spraying[\s_]*paint", r"levelling[\s_]*paint"],
    "overhead_work": [r"hands[\s_]*up", r"layingback", r"laying[\s_]*back"],
    "floor_work": [r"crouch[\s_]*floor", r"kneel[\s_]*floor"],
    "pushing_cart": [r"push.*cart"],
    "picking_objects": [r"picking[\s_]*objects"],
    "lifting": [r"\blifting\b"],
    "lowering": [r"\blowering\b"],
    "turning": [r"turn[\s_]*left", r"turn[\s_]*right"],
    "wheelchair": [r"wheelchair"],

    # --- Medical specialty ---
    "freezing_of_gait": [r"\bfreeze\b", r"freezing"],

    # --- CAPTURE24 coarse intensity labels ---
    "sedentary": [r"sedentary"],
    "light_activity": [r"light[\s_]*activit"],
    "moderate_vigorous": [r"moderate", r"vigorous"],

    # --- LARA abstract motion types ---
    "other_motion": [r"othermotion", r"other[\s_]*motion"],

    # --- KDDI opaque cooking labels ---
    "unknown_cooking": [r"^activity_\d+$"],

    # --- Misc ---
    "salute": [r"salute"],
}

# ---------------------------------------------------------------------------
# Sensor location normalization
# Maps raw sensor names from dataset_info to canonical body-location strings.
# ---------------------------------------------------------------------------
SENSOR_LOCATION_MAP: Dict[str, str] = {
    # Wrist
    "Wrist": "wrist", "RightWrist": "wrist", "LeftWrist": "wrist",
    "R_WRIST": "wrist",
    # Upper arm / arm
    "RightUpperArm": "upper_arm", "LeftUpperArm": "upper_arm",
    "RightArm": "upper_arm", "LeftArm": "upper_arm",
    "UpperArm": "upper_arm",
    "RUA": "upper_arm", "LUA": "upper_arm",
    # Lower arm / forearm
    "Forearm": "forearm",
    "RLA": "forearm", "LLA": "forearm",
    "RightLowerArm": "forearm", "LeftLowerArm": "forearm",
    # Thigh
    "Thigh": "thigh", "RightThigh": "thigh", "LeftThigh": "thigh",
    "R_KNEE": "thigh",
    # Ankle / calf / shin
    "Ankle": "ankle", "RightAnkle": "ankle", "LeftAnkle": "ankle",
    "LeftCalf": "ankle", "RightCalf": "ankle",
    "Shin": "ankle",
    # Shoe
    "L_SHOE": "shoe", "R_SHOE": "shoe",
    # Lower back
    "LowerBack": "lower_back", "BACK": "lower_back", "Back": "lower_back",
    # Chest
    "Chest": "chest",
    # Trunk / torso
    "Trunk": "trunk", "Torso": "trunk",
    # Neck / head
    "Neck": "neck", "Head": "neck",
    # Waist / hip
    "Waist": "waist", "Hip": "waist",
    # Pocket / phone / device
    "Pocket": "phone", "Phone": "phone", "Smartphone": "phone",
    # Glasses
    "SmartGlasses": "glasses",
    # HHAR devices (smartphone/watch on various positions — keep as-is)
    "nexus4_1": "phone", "nexus4_2": "phone",
    "s3_1": "phone", "s3_2": "phone",
    "s3mini_1": "phone", "s3mini_2": "phone",
    "gear_1": "watch",
}


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def classify_label(label: str) -> str:
    """
    Map a raw activity label to its canonical taxonomy group.

    Returns "undefined" for null/undefined/transition labels and
    "uncategorized" for labels that don't match any group.
    """
    if label in ("-1", "Undefined", "Null", "other", "PostureTransition",
                 "NonExperiment", "null"):
        return "undefined"

    label_lower = label.lower().strip()

    for group, patterns in ACTIVITY_TAXONOMY.items():
        for pattern in patterns:
            if re.search(pattern, label_lower):
                return group

    return "uncategorized"


# ---------------------------------------------------------------------------
# Per-dataset helpers
# ---------------------------------------------------------------------------

def get_dataset_groups(dataset_name: str) -> Dict[str, List[str]]:
    """
    Return a mapping of canonical group → [raw label strings] for a dataset.
    Skips undefined/null labels (class index -1).
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    groups: Dict[str, List[str]] = {}
    for idx, label in DATASETS[dataset_name]["labels"].items():
        if idx == -1:
            continue
        group = classify_label(label)
        groups.setdefault(group, []).append(label)

    return groups


def get_all_groups_for_dataset(dataset_name: str) -> Set[str]:
    """Return the set of canonical activity groups present in a dataset."""
    return set(get_dataset_groups(dataset_name).keys())


# ---------------------------------------------------------------------------
# Filtering functions
# ---------------------------------------------------------------------------

def filter_by_activity(activity: str) -> List[str]:
    """
    Return dataset names that contain at least one label mapping to `activity`.
    """
    return [
        name for name in DATASETS
        if activity in get_all_groups_for_dataset(name)
    ]


def filter_by_modality(required: List[str]) -> List[str]:
    """
    Return dataset names that provide ALL of the required modalities for at
    least one sensor location.

    `required` example: ["ACC", "GYRO"]
    """
    required_set = set(required)
    result = []
    for name, meta in DATASETS.items():
        mods = meta["modalities"]
        if isinstance(mods, list):
            available = set(mods)
        else:  # per-sensor dict
            available = set(m for ms in mods.values() for m in ms)
        if required_set <= available:
            result.append(name)
    return result


def filter_by_sensor(location: str) -> List[str]:
    """
    Return dataset names that include a sensor at the given canonical location.

    `location` is a normalized body-location string (see SENSOR_LOCATION_MAP),
    e.g. "wrist", "thigh", "lower_back", "phone".
    """
    result = []
    for name, meta in DATASETS.items():
        for sensor in meta["sensor_list"]:
            if SENSOR_LOCATION_MAP.get(sensor) == location:
                result.append(name)
                break
    return result


def filter_datasets(
    activities: Optional[List[str]] = None,
    modalities: Optional[List[str]] = None,
    sensor_location: Optional[str] = None,
) -> List[str]:
    """
    Compound filter: return datasets matching ALL supplied criteria.

    Parameters
    ----------
    activities      : list of canonical group names — dataset must contain ALL
    modalities      : list of modality strings — dataset must provide ALL
    sensor_location : canonical sensor location — dataset must have this sensor
    """
    candidates = set(DATASETS.keys())

    if activities:
        for act in activities:
            candidates &= set(filter_by_activity(act))

    if modalities:
        candidates &= set(filter_by_modality(modalities))

    if sensor_location:
        candidates &= set(filter_by_sensor(sensor_location))

    return sorted(candidates)


# ---------------------------------------------------------------------------
# Cross-dataset comparison
# ---------------------------------------------------------------------------

def get_activity_overlap(dataset_a: str, dataset_b: str) -> Set[str]:
    """Return canonical activity groups shared by both datasets."""
    return get_all_groups_for_dataset(dataset_a) & get_all_groups_for_dataset(dataset_b)


def group_datasets_by_activity() -> Dict[str, List[str]]:
    """
    Invert the mapping: for every canonical group, list datasets that contain it.
    """
    result: Dict[str, List[str]] = {}
    for name in DATASETS:
        for group in get_all_groups_for_dataset(name):
            result.setdefault(group, []).append(name)
    return result
