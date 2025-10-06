import json
import time
import os
import threading

def read_pipe(pipe_in, last_line,lock):
    for line in iter(pipe_in.readline, ''):
        lock.acquire()
        line = line.strip()
        if line:
            if line=="reset":
                last_line["reset"] = True
            else:
                last_line["line"] = line  # Update the shared variable
        lock.release()



class RealRobotInterface:
    def __init__(self,state_path, action_path):
        self.statefile = open(state_path, 'r')
        self.actionfile = open(action_path, 'w')
        self.last_line={"line":"", "reset":False}
        self.statelock = threading.Lock()
        # Start the thread to read from the pipe
        reader = threading.Thread(target=read_pipe, args=(self.statefile, self.last_line, self.statelock), daemon=True)
        reader.start()
        time.sleep(1)
        self.num_envs = 1
        names_to_id ={
                "left_hip_yaw_joint": 0,
                "left_hip_roll_joint": 1,
                "left_hip_pitch_joint": 2,
                "left_knee_joint": 3,
                "left_ankle_joint": 4,
                "right_hip_yaw_joint": 5,
                "right_hip_roll_joint": 6,
                "right_hip_pitch_joint": 7,
                "right_knee_joint": 8,
                "right_ankle_joint": 9,
                "torso_joint": 10,
                "left_shoulder_pitch_joint": 11,
                "left_shoulder_roll_joint": 12,
                "left_shoulder_yaw_joint": 13,
                "left_elbow_joint": 14,
                "right_shoulder_pitch_joint": 15,
                "right_shoulder_roll_joint": 16,
                "right_shoulder_yaw_joint": 17,
                "right_elbow_joint": 18,
                }
        self.last_ts = 0

    def get_last_state_dict(self):
        self.statelock.acquire()
        lastline = self.last_line["line"]
        state_dict = json.loads(lastline)
        if self.last_line["reset"]:
            state_dict["reset"] = True
            self.last_line["reset"] = False
        else:
            state_dict["reset"] = False
        self.last_ts = state_dict["timems"]
        self.statelock.release()
        return state_dict

    def act(self, action):
        action_str="\n"
        for i in range(19):
            action_str+= f'{action[0,i].item():.4g} '
        action_str += "0.0 "# unused joint
        action_str += str(self.last_ts) # timestep
        self.actionfile.write(action_str)
        self.actionfile.flush()

