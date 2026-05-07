from enum import Enum

class PassHeight(str, Enum):
    GROUND_PASS = "Ground Pass"
    LOW_PASS = "Low Pass"
    HIGH_PASS = "High Pass"

class Bodypart(str, Enum):
    RIGHT_FOOT = "Right Foot"
    LEFT_FOOT = "Left Foot"
    UNKNOWN = "Unknown"
    DROP_KICK = "Drop Kick"
    HEAD =  "Head"
    OTHER = "Other"
    KEEPER_ARM = "Keeper Arm"