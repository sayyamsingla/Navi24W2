"""
This code implements a simple decision-making system for an autonomous agent, 
such as a car in a simulation. It defines five possible actions—LEFT, RIGHT, 
ACCELERATE, BRAKE, and REVERSE—using an Action enum. The ActionState class tracks
the current action, its duration, position history, and whether the agent is stuck. 
The random_move function decides what the agent should do next. It monitors recent
positions, and if the agent stops making progress (as if it has hit a wall), a 
stuck_counter increases. Once the counter gets high enough, the system detects that the 
agent is stuck. When stuck, it prioritizes REVERSE (most of the time) or BRAKE actions to 
try and get free, holding that action for a random duration. If not stuck, it randomly 
selects a new action weighted toward turning and accelerating. The function outputs an 
action array representing how much to steer, accelerate, brake, or reverse, with values
 kept in valid ranges. This allows the agent to move, detect when it’s stuck 
 (like hitting a wall), and take corrective actions automatically.
"""

import numpy as np
import numpy.typing as npt
import random
from enum import Enum

class Action(Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    ACCELERATE = "ACCELERATE"
    BRAKE = "BRAKE"
    REVERSE = "REVERSE"

class ActionState:
    def __init__(self):
        self.current_action = None
        self.action_duration = 0
        self.stuck_counter = 0
        self.prev_position = None
        self.position_history = []
        self.last_reward = 0

action_state = ActionState()

def random_move(s, s_prime=None, reward=None) -> npt.NDArray[np.float32]:
    global action_state
    
    action = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    if s is not None and action_state.prev_position is None:
        action_state.prev_position = s
        action_state.position_history = [s]
   
    if s is not None:
        if action_state.prev_position is not None:
            action_state.prev_position = s
            action_state.position_history.append(s)

            if len(action_state.position_history) > 10:
                action_state.position_history.pop(0)
    

    if reward is not None:
        action_state.last_reward = reward
    

    is_stuck = False
    if len(action_state.position_history) >= 5:
        print("hit a wall")
        is_stuck = action_state.stuck_counter > 15
        action_state.stuck_counter += 1
    else:
        action_state.stuck_counter = 0

    if action_state.current_action is None or action_state.action_duration <= 0 or is_stuck:
        if is_stuck:
            action_state.current_action = Action.REVERSE if random.random() < 0.7 else Action.BRAKE
            action_state.action_duration = random.randint(10, 20)
            action_state.stuck_counter = 0
        else:
            actions = list(Action)
            weights = [0.25, 0.25, 0.35, 0.0, 0.0] 
            action_state.current_action = random.choices(actions, weights=weights, k=1)[0]
            action_state.action_duration = random.randint(10, 30)
    
   
    action_state.action_duration -= 1
    
    if action_state.current_action == Action.LEFT:
        action[0] = -random.uniform(0.5, 1.0) 
    elif action_state.current_action == Action.RIGHT:
        action[0] = random.uniform(0.5, 1.0)
    elif action_state.current_action == Action.ACCELERATE:
        action[1] = random.uniform(0.7, 1.0)
        # action[0] = random.uniform(-0.3, 0.3)
    elif action_state.current_action == Action.BRAKE:
        action[2] = random.uniform(0.5, 1.0)
    elif action_state.current_action == Action.REVERSE:
        action[3] = random.uniform(0.5, 1.0)
        action[0] = random.uniform(-1.0, 1.0)
    
 
    action[0] = np.clip(action[0], -1.0, 1.0)
    action[1] = np.clip(action[1], 0.0, 1.0)   
    action[2] = np.clip(action[2], 0.0, 1.0)   
    action[3] = np.clip(action[3], 0.0, 1.0)   
    
    return action


