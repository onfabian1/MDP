import sys
from enum import Enum
from typing import Tuple

import numpy as np

from .base.base_space.default_value_space import DefaultValueSpace
from .base.base_space.discrete_space import DiscreteSpace
from .base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .base.base_space.space import Space
from .base.blockchain_model import BlockchainModel
from .base.state_transitions import StateTransitions
from enum import Enum


class BitcoinEclipseModel(BlockchainModel):
    def __init__(self, alpha: float, WW: float, gamma: float, max_fork: int):
        self.alpha = alpha
        self.gamma = gamma
        self.WW = WW
        self.max_fork = max_fork

        self.Fork = self.create_int_enum('Fork', ['Irrelevant', 'Relevant', 'Active'])
        self.Action = self.create_int_enum('Action', ['Illegal', 'Adopt', 'Override', 'Match', 'Wait', 'Switch', 'Reveal'])
        self.Group = self.create_int_enum('Group', ['Green, Blue'])
        self.Miner = self.create_int_enum('Miner', ['Attacker, Honest, Nope'])
        self.BlockStatus = self.create_int_enum('BlockStatus', ['On, Off'])
        super().__init__()

    # Set the block status and owner
    def set_block_status(self, vector, blockNum, owner, taken):
        vector[taken:taken + blockNum] = [(self.BlockStatus.On, owner)] * blockNum

    def initializeVector(self, vector):
        vector = [(BlockStatus.Off, Miner.Nope) for _ in range(self.max_fork)]

    @staticmethod
    def copy_vector(vector_a, vector_b, max_idx):
        for i in range(max_idx + 1):
            vector_a[i] = vector_b[i]

    # Calculate the reward for a given vector
    def calc_reward(self, vector, max_idx):
        for i in range(max_idx):
            if self.Miner.Attacker == vector[i][1]:
                count += 1
        return count


    def vacant_blocks(self, vector):
        for i in range(self.max_fork):
            if self.vector[i][0] == self.BlockStatus.Off:
                count += 1
        return count


    # TODO: does h in calc_v is relevant??
    @staticmethod
    def calc_v(v, h):
        sum = 0
        for i in range(h):
            sum += v[i]
        return sum

    @staticmethod
    def cut_v(v, h):
        v_new = v[h + 1:] + (ATTACKER,) * (h + 1)
        return v_new

    @staticmethod
    def update_v(v, tmp, a):
        v_new = list(v)
        v_new[a] = tmp
        return tuple(v_new)

    # @staticmethod
    # def new_update_v(original_tuple, count, block_owner):
    #     # Create a new tuple by concatenating the original tuple with the variables
    #     new_tuple = original_tuple + tuple(block_owner) * count
    #     return new_tuple

    # def clear_v(self): # vector is empty -> all rows are 0
    #     return (0,) * self.max_fork

    def final_v(self): #TODO: ask roi
        return (-1,) * self.max_fork

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.alpha}, {self.gamma}, {self.WW}, {self.max_fork})'

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.gamma, self.WW, self.max_fork)

    def get_state_space(self) -> Space:
        # state_types = [(0, self.max_fork), (0, self.max_fork), (0, self.max_fork), self.Fork]
        # state_types = [(0, self.max_fork), (0, self.max_fork), (0, self.max_fork)] + [(0, 1)] * (self.max_fork) + [
        #     self.Fork]
        state_types = [(0, self.max_fork), (0, self.max_fork), (0, self.max_fork), (0, self.max_fork)] + [
            (ATTACKER, HONEST)] * (self.max_fork) + [(ATTACKER, HONEST)] * (self.max_fork) + [self.Fork] + [self.Fork]

        underlying_space = MultiDimensionalDiscreteSpace(*state_types)
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        return MultiDimensionalDiscreteSpace(self.Action, self.Group)

    def get_initial_state(self) -> BlockchainModel.State:
        return self.combine_state(0, 0, 0, 0, self.self.initializeVector(), self.self.initializeVector(), self.Fork.Irrelevant, self.Fork.Irrelevant)

    def get_final_state(self) -> BlockchainModel.State:
        return self.combine_state(-1, -1, -1, -1, self.final_v(), self.final_v(), self.Fork.Irrelevant, self.Fork.Irrelevant)

    # def dissect_state(self, state: BlockchainModel.State) -> Tuple[int, int, int, int, tuple, tuple, Enum, Enum]:
    #     att_up = state[0]
    #     green = state[1]
    #     att_down = state[2]
    #     blue = state[3]
    #     v_ag = state[4:4 + self.max_fork]
    #     v_ab = state[4 + self.max_fork: 4 + self.max_fork + self.max_fork]
    #     fork_green = state[-2]
    #     fork_blue = state[-1]
    #     return self.combine_state(att_up, green, att_down, blue, v_ag, v_ab, fork_green, fork_blue)

    def dissect_state_NOtuple(self, state: BlockchainModel.State):
        att_up = state[0]
        green = state[1]
        att_down = state[2]
        blue = state[3]
        v_ag = state[4:4 + self.max_fork]
        v_ab = state[4 + self.max_fork: 4 + self.max_fork + self.max_fork]
        fork_green = state[-2]
        fork_blue = state[-1]
        return att_up, green, att_down, blue, v_ag, v_ab, fork_green, fork_blue


    @staticmethod
    def combine_state(att_up, green, att_down, blue, v_ag, v_ab, fork_green, fork_blue) -> BlockchainModel.State:
        return (att_up, green, att_down, blue) + v_ag + v_ab + (fork_green, fork_blue)

    # noinspection DuplicatedCode
    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action, check_valid: bool = True)-> StateTransitions:
        transitions = StateTransitions()
        if state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        # att_up, green, att_down, blue, v_ag, v_ab, fork_green, fork_blue = self.dissect_state_NOtuple(state)

        # if a < be or sum(v[:a]) != be or sum(v[a:]) > 0:
        #     # Bad states
        #     transitions.add(self.final_state, probability=1)
        #     return transitions

        att_up, green, att_down, blue, v_ag, v_ab, fork_green, fork_blue = self.dissect_state_NOtuple(state)
        action_type, group = action
        if (sum(v_ag[att_up+green:]) > 0) or (sum(v_ab[att_down+blue:]) > 0):
            # Bad states
            transitions.add(self.final_state, probability=1)
            return transitions

        if action_type is self.Action.Adopt and group is self.Group.Green:
            vacant = vacant_blocks(v_ag)
            if green > 0 and green > att_up and vacant > green:
                next_state = self.combine_state(0, 0, att_down, blue, self.set_block_status(v_ag, green,
                                                self.Miner.Honest, self.max_fork - vacant)
                                               , v_ab, self.Fork.Irrelevant, self.Fork.Irrelevant)
                transitions.add(next_state, probability=1)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        elif action_type is self.Action.Adopt and group is self.Group.Blue:
            vacant = vacant_blocks(v_ab)
            if blue > 0 and blue > att_down and vacant > blue:
                next_state = self.combine_state(att_up, green, 0, 0, v_ag, self.set_block_status(v_ab, blue, self.Miner.Honest, self.max_fork - vacant), self.Fork.Irrelevant, self.Fork.Irrelevant)
                transitions.add(next_state, probability=1)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        if action_type is self.Action.Override and group is self.Group.Green:
            vacant = vacant_blocks(v_ag)
            if att_up > green and vacant > (green + 1):
                next_state = self.combine_state(att_up - green - 1, 0, att_down, blue, self.set_block_status(v_ag, green+1, self.Miner.Attacker, self.max_fork - vacant), v_ab, self.Fork.Irrelevant, fork_blue)
                transitions.add(next_state, probability=1)

            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        elif action_type is self.Action.Override and group is self.Group.Blue:
            vacant = vacant_blocks(v_ab)
            if att_down > blue and vacant > (blue + 1):
                next_state = self.combine_state(att_up, green, att_down - blue - 1, 0, v_ag, self.set_block_status(v_ab, blue+1, self.Miner.Attacker, self.max_fork - vacant), fork_green, self.Fork.Irrelevant)
                transitions.add(next_state, probability=1)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        if action_type is self.Action.Reveal and group is self.Group.Green:
            vacant_b = vacant_blocks(v_ab)
            vacant_g = vacant_blocks(v_ag)
            if (self.max_fork - vacant_g) > (self.max_fork - vacant_b):
                next_state = self.combine_state(att_up, green, 0, 0, self.copy_vector(v_ag, v_ab, self.max_fork - vacant_g), self.initializeVector(v_ab), fork_green, self.Fork.Irrelevant)
                reward_attacker = self.calc_reward(v_ag, self.max_fork - vacant_b)
                transitions.add(next_state, probability=1, reward=reward_attacker, difficulty_contribution=self.max_fork-vacant_b-reward_attacker)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        elif action_type is self.Action.Reveal and group is self.Group.Blue:
            vacant_b = vacant_blocks(v_ab)
            vacant_g = vacant_blocks(v_ag)
            if (self.max_fork - vacant_g) < (self.max_fork - vacant_b):
                next_state = self.combine_state(0, 0, att_down, blue, self.initializeVector(v_ag), self.copy_vector(v_ab, v_ag, self.max_fork - vacant_b), self.Fork.Irrelevant, fork_blue)
                reward_attacker = self.calc_reward(v_ab, self.max_fork - vacant_g)
                transitions.add(next_state, probability=1, reward=reward_attacker,
                                difficulty_contribution=self.max_fork - vacant_g - reward_attacker)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        if action_type is self.Action.Match and group is self.Group.Green:
            if self.max_fork > att_up >= green > 0 and fork_green is self.Fork.Relevant:
                next_state = self.combine_state(att_up, green, att_down, blue, v_ag, v_ab, self.Fork.Active, fork_blue)
                transitions.add(next_state, probability=1)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        elif action_type is self.Action.Match and group is self.Group.Blue:
            if self.max_fork > att_down >= blue > 0 and fork_blue is self.Fork.Relevant:
                next_state = self.combine_state(att_up, green, att_down, blue, v_ag, v_ab, fork_green, self.Fork.Active)
                transitions.add(next_state, probability=1)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        if action_type is self.Action.Wait and group is self.Group.Green:
            if fork_green is not self.Fork.Active and fork_blue is not self.Fork.Active and att_up < self.max_fork and green < self.max_fork and blue < self.max_fork:

                # TODO: ask Roi about attacker_block
                # attacker_block = self.combine_state(a + 1, h, be, self.update_v(v, 0, a), self.Fork.Irrelevant)
                # transitions.add(attacker_block, probability=self.alpha)

                next_state = self.combine_state(att_up+1, green, att_down, blue, v_ag, v_ab, self.Fork.Irrelevant, fork_blue)
                transitions.add(next_state, probability=self.alpha)

                next_state = self.combine_state(att_up, green+1, att_down, blue, v_ag, v_ab, self.Fork.Relevant, fork_blue)
                transitions.add(next_state, probability=self.WW)

                next_state = self.combine_state(att_up, green, att_down, blue+1, v_ag, v_ab, fork_green, self.Fork.Relevant)
                transitions.add(next_state, probability= (1 - self.WW - self.alpha))


            elif fork_blue is not self.Fork.Active and fork_green is self.Fork.Active and att_up < self.max_fork and green < self.max_fork and blue < self.max_fork:
                vacant = vacant_blocks(v_ag)
                if att_up > green and vacant > (green + 1):
                    next_state = self.combine_state(att_up + 1, green, att_down, blue, v_ag, v_ab, self.Fork.Active,
                                                    fork_blue)
                    transitions.add(next_state, probability=self.alpha)

                    next_state = self.combine_state(att_up, green, att_down, blue + 1, v_ag, v_ab, self.Fork.Active,
                                                    self.Fork.Relevant)
                    transitions.add(next_state, probability=(1 - self.WW - self.alpha))

                    next_state = self.combine_state(att_up - green, 1, att_down, blue, self.set_block_status(v_ag, green, self.Miner.Attacker, self.max_fork - vacant), v_ab, self.Fork.Relevant,
                                                    fork_blue)
                    transitions.add(next_state, probability=(self.gamma * self.WW))

                    next_state = self.combine_state(att_up, green+1, att_down, blue, v_ag, v_ab, self.Fork.Relevant,
                                                    fork_blue)
                    transitions.add(next_state, probability=((1-self.gamma) * self.WW))

            elif fork_green is not self.Fork.Active and fork_blue is self.Fork.Active and att_up < self.max_fork and green < self.max_fork and blue < self.max_fork:
                vacant = vacant_blocks(v_ab)
                if att_down > blue and vacant > (blue + 1):
                    next_state = self.combine_state(att_up + 1, green, att_down, blue, v_ag, v_ab, self.Fork.Irrelevant,
                                                    self.Fork.Active)
                    transitions.add(next_state, probability=self.alpha)

                    next_state = self.combine_state(att_up, green+1, att_down, blue, v_ag, v_ab, self.Fork.Relevant,
                                                    self.Fork.Active)
                    transitions.add(next_state, probability=self.WW)

                    next_state = self.combine_state(att_up, green, att_down, blue+1, v_ag, v_ab, fork_green, self.Fork.Relevant)
                    transitions.add(next_state, probability=(self.gamma * (1 - self.WW - self.alpha)))

                    next_state = self.combine_state(att_up, green, att_down - blue, 1, v_ag, self.set_block_status(v_ab, blue, self.Miner.Attacker, self.max_fork - vacant),
                                                    fork_green, self.Fork.Relevant)
                    transitions.add(next_state, probability=((1 - self.gamma) * (1 - self.WW - self.alpha)))

            elif fork_green is self.Fork.Active and fork_blue is self.Fork.Active and att_up < self.max_fork and green < self.max_fork and blue < self.max_fork:
                vacant_b = vacant_blocks(v_ab)
                vacant_g = vacant_blocks(v_ag)
                #TODO: ask Roi how to separate the conditions about att_up > green and att_down > blue
                if att_down > blue and vacant_b > (blue + 1) and att_up > green and vacant_g > (green + 1):
                    next_state = self.combine_state(att_up + 1, green, att_down, blue, v_ag, v_ab,
                                                    self.Fork.Irrelevant,
                                                    self.Fork.Active)
                    transitions.add(next_state, probability=self.alpha)

                    next_state = self.combine_state(att_up - green, 1, att_down, blue, self.set_block_status(v_ag, green, self.Miner.Attacker, self.max_fork - vacant_g), v_ab,
                                                    self.Fork.Relevant, fork_blue)
                    transitions.add(next_state, probability=(self.gamma * self.WW))

                    next_state = self.combine_state(att_up, green + 1, att_down, blue, v_ag, v_ab, self.Fork.Relevant, fork_blue)
                    transitions.add(next_state, probability=((1 - self.gamma) * self.WW))

                    next_state = self.combine_state(att_up, green, att_down, blue + 1, v_ag, v_ab, fork_green, self.Fork.Relevant)
                    transitions.add(next_state, probability=(self.gamma * (1 - self.WW - self.alpha)))

                    next_state = self.combine_state(att_up, green, att_down - blue, 1, v_ag, self.set_block_status(v_ab, blue, self.Miner.Attacker, self.max_fork - vacant_b), fork_green, self.Fork.Relevant)
                    transitions.add(next_state, probability=((1 - self.gamma) * (1 - self.WW - self.alpha)))

                else:
                    transitions.add(self.final_state, probability=1, reward=self.error_penalty)
        if action_type is self.Action.Wait and group is self.Group.Blue:
            if fork_green is not self.Fork.Active and fork_blue is not self.Fork.Active and att_down < self.max_fork and green < self.max_fork and blue < self.max_fork:

                next_state = self.combine_state(att_up, green, att_down + 1, blue, v_ag, v_ab, fork_green, self.Fork.Irrelevant)
                transitions.add(next_state, probability=self.alpha)

                next_state = self.combine_state(att_up, green, att_down, blue + 1, v_ag, v_ab, fork_green, self.Fork.Relevant)
                transitions.add(next_state, probability=(1- self.WW - self.alpha))

                next_state = self.combine_state(att_up, green + 1, att_down, blue, v_ag, v_ab, fork_green, self.Fork.Relevant)
                transitions.add(next_state, probability=self.WW)

            elif fork_green is not self.Fork.Active and fork_blue is self.Fork.Active and att_down < self.max_fork and green < self.max_fork and blue < self.max_fork:
                vacant = vacant_blocks(v_ab)
                if att_down > blue and vacant > (blue + 1):
                    next_state = self.combine_state(att_up, green, att_down + 1, blue, v_ag, v_ab, fork_green,
                                                    self.Fork.Active)
                    transitions.add(next_state, probability=self.alpha)

                    next_state = self.combine_state(att_up, green + 1, att_down, blue, v_ag, v_ab, self.Fork.Relevant,
                                                    self.Fork.Active)
                    transitions.add(next_state, probability=self.WW)

                    next_state = self.combine_state(att_up, green, att_down - blue, 1, v_ag, self.set_block_status(v_ab, blue, self.Miner.Attacker, self.max_fork - vacant), fork_green,
                                                    self.Fork.Relevant)
                    transitions.add(next_state, probability=(self.gamma * (1 - self.alpha - self.WW)))

                    next_state = self.combine_state(att_up, green, att_down, blue + 1, v_ag, v_ab, fork_green, self.Fork.Relevant)
                    transitions.add(next_state, probability=((1-self.gamma) * (1 - self.WW - self.alpha)))

            elif fork_green is self.Fork.Active and fork_blue is not self.Fork.Active and att_down < self.max_fork and green < self.max_fork and blue < self.max_fork:
                vacant = vacant_blocks(v_ag)
                if att_up > green and vacant > (green + 1):
                    next_state = self.combine_state(att_up, green, att_down + 1, blue, v_ag, v_ab, self.Fork.Active,
                                                    self.Fork.Irrelevant)
                    transitions.add(next_state, probability=self.alpha)

                    next_state = self.combine_state(att_up, green, att_down, blue + 1, v_ag, v_ab, self.Fork.Active,
                                                    self.Fork.Relevant)
                    transitions.add(next_state, probability=(1 - self.alpha - self.WW))

                    next_state = self.combine_state(att_up, green + 1, att_down, blue, v_ag, v_ab, self.Fork.Relevant, fork_blue)
                    transitions.add(next_state, probability=(self.gamma * self.WW))

                    next_state = self.combine_state(att_up - green, 1, att_down, blue, self.set_block_status(v_ag, green, self.Miner.Attacker, self.max_fork - vacant), v_ab,
                                                    self.Fork.Relevant, fork_blue)
                    transitions.add(next_state, probability=((1 - self.gamma) * self.WW))

            elif fork_green is self.Fork.Active and fork_blue is self.Fork.Active and att_down < self.max_fork and green < self.max_fork and blue < self.max_fork:
                vacant_b = vacant_blocks(v_ab)
                vacant_g = vacant_blocks(v_ag)
                #TODO: ask Roi how to separate the conditions about att_up > green and att_down > blue
                if att_down > blue and vacant_b > (blue + 1) and att_up > green and vacant_g > (green + 1):
                    next_state = self.combine_state(att_up, green, att_down + 1, blue, v_ag, v_ab,
                                                    self.Fork.Active,
                                                    self.Fork.Irrelevant)
                    transitions.add(next_state, probability=self.alpha)

                    next_state = self.combine_state(att_up - green, 1, att_down, blue, self.set_block_status(v_ag, green, self.Miner.Attacker, self.max_fork - vacant_g), v_ab,
                                                    self.Fork.Relevant, fork_blue)
                    transitions.add(next_state, probability=(self.gamma * self.WW))

                    next_state = self.combine_state(att_up, green + 1, att_down, blue, v_ag, v_ab, self.Fork.Relevant, fork_blue)
                    transitions.add(next_state, probability=((1 - self.gamma) * self.WW))

                    next_state = self.combine_state(att_up, green, att_down, blue + 1, v_ag, v_ab, fork_green, self.Fork.Relevant)
                    transitions.add(next_state, probability=(self.gamma * (1 - self.WW - self.alpha)))

                    next_state = self.combine_state(att_up, green, att_down - blue, 1, v_ag, self.set_block_status(v_ab, blue, self.Miner.Attacker, self.max_fork - vacant_b), fork_green, self.Fork.Relevant)
                    transitions.add(next_state, probability=((1 - self.gamma) * (1 - self.WW - self.alpha)))

                else:
                    transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        return transitions

    #TODO: ask Roi if this functions are relevant?
    def get_honest_revenue(self) -> float:
        return (self.alpha/(1-self.WW))

    # def is_policy_honest(self, policy: BlockchainModel.Policy) -> bool:
    #    return policy[self.state_space.element_to_index((0, 0, self.Fork.Irrelevant))] == self.Action.Wait \
    #           and policy[self.state_space.element_to_index((1, 0, self.Fork.Irrelevant))] == self.Action.Override \
    #           and policy[self.state_space.element_to_index((0, 1, self.Fork.Irrelevant))] == self.Action.Adopt \
    #           and policy[self.state_space.element_to_index((0, 1, self.Fork.Relevant))] == self.Action.Adopt

    def build_honest_policy(self) -> BlockchainModel.Policy:
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(self.state_space.size):
            a, h, be, v, fork = self.state_space.index_to_element(i)

            if h > a:
                action = self.Action.Adopt
            elif a > h:
                action = self.Action.Override
            else:
                action = self.Action.Wait

            policy[i] = action

        return tuple(policy)

    def build_sm1_policy(self) -> BlockchainModel.Policy:
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(self.state_space.size):
            a, h, be, v, fork = self.state_space.index_to_element(i)

            if h > a:
                action = self.Action.Adopt
            elif (h == a - 1 and a >= 2) or a == self.max_fork:
                action = self.Action.Override
            elif (h == 1 and a == 1) and fork is self.Fork.Relevant:
                action = self.Action.Match
            else:
                action = self.Action.Wait

            policy[i] = action

        return tuple(policy)


if __name__ == '__main__':
    print('bitcoin_mdp module test')
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    mdp = BitcoinEclipseModel(0.35, 0.3, 0.5, 100)
    print(mdp.state_space.size)
    p = mdp.build_sm1_policy()
    print(p[:10])
