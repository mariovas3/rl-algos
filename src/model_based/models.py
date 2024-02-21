import random


class HashTableModel:
    def __init__(self):
        self.experience = {}
        self.search_control = []

    def update(self, obs, action, R, obstp1):
        key = obs + (action,)
        if key not in self.experience:
            self.search_control.append(key)
        self.experience[key] = obstp1 + (R,)

    def sample(self):
        if not self.search_control:
            raise ValueError("no experience seen!")
        temp = random.choice(self.search_control)
        temp2 = self.experience[temp]
        # return obs, action, R, obstp1
        return temp[:-1], temp[-1], temp2[-1], temp2[:-1]
