import sys
from enum import Enum
from typing import Tuple

import numpy as np

from base.base_space.default_value_space import DefaultValueSpace
from base.base_space.discrete_space import DiscreteSpace
from base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from base.base_space.space import Space
from base.blockchain_model import BlockchainModel
from base.state_transitions import StateTransitions
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
        self.Miner = self.create_int_enum('Miner', ['Attacker', 'Honest', 'Nope'])
        self.BlockStatus = self.create_int_enum('BlockStatus', ['Off', 'On'])
        super().__init__()

    # Set the block status and owner
    def set_block_status(self, vector, blockNum, owner, taken):
        return vector[:2*taken] + (self.BlockStatus.On, owner) * blockNum + vector[2*taken + 2*blockNum:]

    def initializeVector(self):
        return (self.BlockStatus.Off, self.Miner.Nope) * self.max_fork

    @staticmethod
    def copy_vector(vector_a, max_idx_a, max_idx_b):
      return vector_a[max_idx_b:max_idx_a]
      #  for i in range(max_idx + 1):
      #      vector_a[i] = vector_b[i]

    # Calculate the reward for a given vector
    def calc_reward(self, vector, max_idx):
        count_1 = 0
        for i in range(max_idx):
            if self.Miner.Attacker == vector[2*i+1]:
                count_1 += 1
        return count_1

    def taken_blocks(self, vector):
        count_2 = 0
        for i in range(self.max_fork):
            if vector[2*i] == self.BlockStatus.On:
                count_2 += 1
        return count_2

    def vacant_blocks(self, vector):
        count_3 = 0
        for i in range(self.max_fork):
            if vector[2*i] == self.BlockStatus.Off:
                count_3 += 1
        return count_3

    def sum_blocks(self, vector, att, green, blue):
        flag = False
        for i in range(self.max_fork):
            if vector[att + green + 2*i] == self.BlockStatus.On or vector[att + blue + 2*i] == self.BlockStatus.On:
                flag = True
        return flag

    def final_v(self): #TODO: ask roi
        return (-1,) * self.max_fork

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.alpha}, {self.gamma}, {self.WW}, {self.max_fork})'

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.gamma, self.WW, self.max_fork)

    def get_state_space(self) -> Space:
        state_types = [(0, self.max_fork), (0, self.max_fork), (0, self.max_fork), (0, self.max_fork)] + [
            self.BlockStatus, self.Miner] * self.max_fork + [self.BlockStatus, self.Miner] * self.max_fork + [self.Fork] + [self.Fork]
        underlying_space = MultiDimensionalDiscreteSpace(*state_types)
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        return MultiDimensionalDiscreteSpace(self.Action, self.Group)

    def get_initial_state(self) -> BlockchainModel.State:
        return self.combine_state(0, 0, 0, 0, self.initializeVector(), self.initializeVector(), self.Fork.Irrelevant, self.Fork.Irrelevant)

    def get_final_state(self) -> BlockchainModel.State:
        return self.combine_state(-1, -1, -1, -1, self.final_v(), self.final_v(), self.Fork.Irrelevant, self.Fork.Irrelevant)

    def dissect_state_NOtuple(self, state: BlockchainModel.State):
        att_up = state[0]
        green = state[1]
        att_down = state[2]
        blue = state[3]
        v_ag = state[4:4 + 2 * self.max_fork]
        v_ab = state[4 + 2 * self.max_fork: 4 + 2 * self.max_fork + 2 * self.max_fork]
        fork_green = state[-2]
        fork_blue = state[-1]
        return att_up, green, att_down, blue, v_ag, v_ab, fork_green, fork_blue


    @staticmethod
    def combine_state(att_up, green, att_down, blue, v_ag, v_ab, fork_green, fork_blue) -> BlockchainModel.State:
        return (att_up, green, att_down, blue) + v_ag + v_ab + (fork_green, fork_blue)

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action, check_valid: bool = True)-> StateTransitions:
        transitions = StateTransitions()
        if state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        att_up, green, att_down, blue, v_ag, v_ab, fork_green, fork_blue = self.dissect_state_NOtuple(state)
        action_type, group = action
        # Bad states
        if vacant_blocks(v_ag) < green or vacant_blocks(v_ab) < blue:
            raise Exception("Bad State")
        if sum_blocks(v_ag, att_up, green, blue):
            raise Exception("Bad State")

        if action_type is self.Action.Adopt and group is self.Group.Green:
            taken = taken_blocks(v_ag)
            if green > 0:
                next_state = self.combine_state(0, 0, att_down, blue, self.set_block_status(v_ag, green,
                                                self.Miner.Honest, taken)
                                               , v_ab, self.Fork.Irrelevant, fork_blue)
                transitions.add(next_state, probability=1)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        elif action_type is self.Action.Adopt and group is self.Group.Blue:
            taken = taken_blocks(v_ab)
            if blue > 0:
                next_state = self.combine_state(att_up, green, 0, 0, v_ag, self.set_block_status(v_ab, blue, self.Miner.Honest, taken), fork_green, self.Fork.Irrelevant)
                transitions.add(next_state, probability=1)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        elif action_type is self.Action.Override and group is self.Group.Green:
            taken = taken_blocks(v_ag)
            if att_up > green and (self.max_fork - taken) >= (green + 1):
                next_state = self.combine_state(att_up - green - 1, 0, att_down, blue, self.set_block_status(v_ag, green+1, self.Miner.Attacker, taken), v_ab, self.Fork.Irrelevant, fork_blue)
                transitions.add(next_state, probability=1)

            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        elif action_type is self.Action.Override and group is self.Group.Blue:
            taken = taken_blocks(v_ab)
            if att_down > blue and (self.max_fork - taken) >= (blue + 1):
                next_state = self.combine_state(att_up, green, att_down - blue - 1, 0, v_ag, self.set_block_status(v_ab, blue+1, self.Miner.Attacker, taken), fork_green, self.Fork.Irrelevant)
                transitions.add(next_state, probability=1)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        elif action_type is self.Action.Reveal and group is self.Group.Green:
            taken_b = taken_blocks(v_ab)
            taken_g = taken_blocks(v_ag)
            if taken_g > taken_b:
                next_state = self.combine_state(att_up, green, 0, 0, self.copy_vector(v_ag, taken_g, taken_b + 1), self.initializeVector(), fork_green, self.Fork.Irrelevant)
                reward_attacker = self.calc_reward(v_ag, taken_b)
                transitions.add(next_state, probability=1, reward=reward_attacker, difficulty_contribution= taken_b)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        elif action_type is self.Action.Reveal and group is self.Group.Blue:
            taken_b = taken_blocks(v_ab)
            taken_g = taken_blocks(v_ag)
            if (taken_g) < (taken_b):
                next_state = self.combine_state(0, 0, att_down, blue, self.initializeVector(), self.copy_vector(v_ab, v_ag, taken_b), self.Fork.Irrelevant, fork_blue)
                reward_attacker = self.calc_reward(v_ab, taken_g)
                transitions.add(next_state, probability=1, reward=reward_attacker,
                                difficulty_contribution=taken_g)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        elif action_type is self.Action.Match and group is self.Group.Green:
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

        elif action_type is self.Action.Wait and group is self.Group.Green:
            if fork_green is not self.Fork.Active and fork_blue is not self.Fork.Active:
                if att_up < self.max_fork and green < self.max_fork and blue < self.max_fork:
                    next_state = self.combine_state(att_up+1, green, att_down, blue, v_ag, v_ab, self.Fork.Irrelevant, fork_blue)
                    transitions.add(next_state, probability=self.alpha)

                    next_state = self.combine_state(att_up, green+1, att_down, blue, v_ag, v_ab, self.Fork.Relevant, fork_blue)
                    transitions.add(next_state, probability=self.WW)

                    next_state = self.combine_state(att_up, green, att_down, blue+1, v_ag, v_ab, fork_green, self.Fork.Relevant)
                    transitions.add(next_state, probability= (1 - self.WW - self.alpha))
                else:
                    transitions.add(self.final_state, probability=1, reward=self.error_penalty)


            elif fork_blue is not self.Fork.Active and fork_green is self.Fork.Active and att_up < self.max_fork and green < self.max_fork and blue < self.max_fork:
                vacant = vacant_blocks(v_ag)
                taken = self.max_fork - vacant
                if att_up > green and vacant > (green + 1):
                    next_state = self.combine_state(att_up + 1, green, att_down, blue, v_ag, v_ab, self.Fork.Active,
                                                    fork_blue)
                    transitions.add(next_state, probability=self.alpha)

                    next_state = self.combine_state(att_up, green, att_down, blue + 1, v_ag, v_ab, self.Fork.Active,
                                                    self.Fork.Relevant)
                    transitions.add(next_state, probability=(1 - self.WW - self.alpha))

                    next_state = self.combine_state(att_up - green, 1, att_down, blue, self.set_block_status(v_ag, green, self.Miner.Attacker, taken), v_ab, self.Fork.Relevant,
                                                    fork_blue)
                    transitions.add(next_state, probability=(self.gamma * self.WW))

                    next_state = self.combine_state(att_up, green+1, att_down, blue, v_ag, v_ab, self.Fork.Relevant,
                                                    fork_blue)
                    transitions.add(next_state, probability=((1-self.gamma) * self.WW))
                else:
                    transitions.add(self.final_state, probability=1, reward=self.error_penalty)

            elif fork_green is not self.Fork.Active and fork_blue is self.Fork.Active and att_up < self.max_fork and green < self.max_fork and blue < self.max_fork:
                vacant = vacant_blocks(v_ab)
                taken = self.max_fork - vacant
                if att_down > blue and vacant > (blue + 1):
                    next_state = self.combine_state(att_up + 1, green, att_down, blue, v_ag, v_ab, self.Fork.Irrelevant,
                                                    self.Fork.Active)
                    transitions.add(next_state, probability=self.alpha)

                    next_state = self.combine_state(att_up, green+1, att_down, blue, v_ag, v_ab, self.Fork.Relevant,
                                                    self.Fork.Active)
                    transitions.add(next_state, probability=self.WW)

                    next_state = self.combine_state(att_up, green, att_down, blue+1, v_ag, v_ab, fork_green, self.Fork.Relevant)
                    transitions.add(next_state, probability=(self.gamma * (1 - self.WW - self.alpha)))

                    next_state = self.combine_state(att_up, green, att_down - blue, 1, v_ag, self.set_block_status(v_ab, blue, self.Miner.Attacker, taken),
                                                    fork_green, self.Fork.Relevant)
                    transitions.add(next_state, probability=((1 - self.gamma) * (1 - self.WW - self.alpha)))
                else:
                    transitions.add(self.final_state, probability=1, reward=self.error_penalty)

            elif fork_green is self.Fork.Active and fork_blue is self.Fork.Active and att_up < self.max_fork and green < self.max_fork and blue < self.max_fork:
                vacant_b = vacant_blocks(v_ab)
                taken_b = self.max_fork - vacant_b
                vacant_g = vacant_blocks(v_ag)
                taken_g = self.max_fork - vacant_g
                #TODO: ask Roi how to separate the conditions about att_up > green and att_down > blue
                if att_down > blue and vacant_b > (blue + 1) and att_up > green and vacant_g > (green + 1):
                    next_state = self.combine_state(att_up + 1, green, att_down, blue, v_ag, v_ab,
                                                    self.Fork.Irrelevant,
                                                    self.Fork.Active)
                    transitions.add(next_state, probability=self.alpha)

                    next_state = self.combine_state(att_up - green, 1, att_down, blue, self.set_block_status(v_ag, green, self.Miner.Attacker, taken_g), v_ab,
                                                    self.Fork.Relevant, fork_blue)
                    transitions.add(next_state, probability=(self.gamma * self.WW))

                    next_state = self.combine_state(att_up, green + 1, att_down, blue, v_ag, v_ab, self.Fork.Relevant, fork_blue)
                    transitions.add(next_state, probability=((1 - self.gamma) * self.WW))

                    next_state = self.combine_state(att_up, green, att_down, blue + 1, v_ag, v_ab, fork_green, self.Fork.Relevant)
                    transitions.add(next_state, probability=(self.gamma * (1 - self.WW - self.alpha)))

                    next_state = self.combine_state(att_up, green, att_down - blue, 1, v_ag, self.set_block_status(v_ab, blue, self.Miner.Attacker, taken_b), fork_green, self.Fork.Relevant)
                    transitions.add(next_state, probability=((1 - self.gamma) * (1 - self.WW - self.alpha)))

                else:
                    transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        elif action_type is self.Action.Wait and group is self.Group.Blue:
            if fork_green is not self.Fork.Active and fork_blue is not self.Fork.Active:
                if att_down < self.max_fork and green < self.max_fork and blue < self.max_fork:
                    next_state = self.combine_state(att_up, green, att_down + 1, blue, v_ag, v_ab, fork_green, self.Fork.Irrelevant)
                    transitions.add(next_state, probability=self.alpha)

                    next_state = self.combine_state(att_up, green, att_down, blue + 1, v_ag, v_ab, fork_green, self.Fork.Relevant)
                    transitions.add(next_state, probability=(1- self.WW - self.alpha))

                    next_state = self.combine_state(att_up, green + 1, att_down, blue, v_ag, v_ab, fork_green, self.Fork.Relevant)
                    transitions.add(next_state, probability=self.WW)
                else:
                    transitions.add(self.final_state, probability=1, reward=self.error_penalty)

            elif fork_green is not self.Fork.Active and fork_blue is self.Fork.Active and att_down < self.max_fork and green < self.max_fork and blue < self.max_fork:
                vacant = vacant_blocks(v_ab)
                taken = self.max_fork - vacant
                if att_down > blue and vacant > (blue + 1):
                    next_state = self.combine_state(att_up, green, att_down + 1, blue, v_ag, v_ab, fork_green,
                                                    self.Fork.Active)
                    transitions.add(next_state, probability=self.alpha)

                    next_state = self.combine_state(att_up, green + 1, att_down, blue, v_ag, v_ab, self.Fork.Relevant,
                                                    self.Fork.Active)
                    transitions.add(next_state, probability=self.WW)

                    next_state = self.combine_state(att_up, green, att_down - blue, 1, v_ag, self.set_block_status(v_ab, blue, self.Miner.Attacker, taken), fork_green,
                                                    self.Fork.Relevant)
                    transitions.add(next_state, probability=(self.gamma * (1 - self.alpha - self.WW)))

                    next_state = self.combine_state(att_up, green, att_down, blue + 1, v_ag, v_ab, fork_green, self.Fork.Relevant)
                    transitions.add(next_state, probability=((1-self.gamma) * (1 - self.WW - self.alpha)))
                else:
                    transitions.add(self.final_state, probability=1, reward=self.error_penalty)

            elif fork_green is self.Fork.Active and fork_blue is not self.Fork.Active and att_down < self.max_fork and green < self.max_fork and blue < self.max_fork:
                vacant = vacant_blocks(v_ag)
                taken = self.max_fork
                if att_up > green and vacant > (green + 1):
                    next_state = self.combine_state(att_up, green, att_down + 1, blue, v_ag, v_ab, self.Fork.Active,
                                                    self.Fork.Irrelevant)
                    transitions.add(next_state, probability=self.alpha)

                    next_state = self.combine_state(att_up, green, att_down, blue + 1, v_ag, v_ab, self.Fork.Active,
                                                    self.Fork.Relevant)
                    transitions.add(next_state, probability=(1 - self.alpha - self.WW))

                    next_state = self.combine_state(att_up, green + 1, att_down, blue, v_ag, v_ab, self.Fork.Relevant, fork_blue)
                    transitions.add(next_state, probability=(self.gamma * self.WW))

                    next_state = self.combine_state(att_up - green, 1, att_down, blue, self.set_block_status(v_ag, green, self.Miner.Attacker, taken), v_ab,
                                                    self.Fork.Relevant, fork_blue)
                    transitions.add(next_state, probability=((1 - self.gamma) * self.WW))
                else:
                    transitions.add(self.final_state, probability=1, reward=self.error_penalty)

            elif fork_green is self.Fork.Active and fork_blue is self.Fork.Active and att_down < self.max_fork and green < self.max_fork and blue < self.max_fork:
                vacant_b = vacant_blocks(v_ab)
                taken_b = self.max_fork - vacant_b
                vacant_g = vacant_blocks(v_ag)
                taken_g = self.max_fork - vacant_g
                #TODO: ask Roi how to separate the conditions about att_up > green and att_down > blue
                if att_down > blue and vacant_b > (blue + 1) and att_up > green and vacant_g > (green + 1):
                    next_state = self.combine_state(att_up, green, att_down + 1, blue, v_ag, v_ab,
                                                    self.Fork.Active,
                                                    self.Fork.Irrelevant)
                    transitions.add(next_state, probability=self.alpha)

                    next_state = self.combine_state(att_up - green, 1, att_down, blue, self.set_block_status(v_ag, green, self.Miner.Attacker, taken_g), v_ab,
                                                    self.Fork.Relevant, fork_blue)
                    transitions.add(next_state, probability=(self.gamma * self.WW))

                    next_state = self.combine_state(att_up, green + 1, att_down, blue, v_ag, v_ab, self.Fork.Relevant, fork_blue)
                    transitions.add(next_state, probability=((1 - self.gamma) * self.WW))

                    next_state = self.combine_state(att_up, green, att_down, blue + 1, v_ag, v_ab, fork_green, self.Fork.Relevant)
                    transitions.add(next_state, probability=(self.gamma * (1 - self.WW - self.alpha)))

                    next_state = self.combine_state(att_up, green, att_down - blue, 1, v_ag, self.set_block_status(v_ab, blue, self.Miner.Attacker, taken_b), fork_green, self.Fork.Relevant)
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

    # def build_honest_policy(self) -> BlockchainModel.Policy:
    #     policy = np.zeros(self.state_space.size, dtype=int)
    #
    #     for i in range(self.state_space.size):
    #         a, h, be, v, fork = self.state_space.index_to_element(i)
    #
    #         if h > a:
    #             action = self.Action.Adopt
    #         elif a > h:
    #             action = self.Action.Override
    #         else:
    #             action = self.Action.Wait
    #
    #         policy[i] = action
    #
    #     return tuple(policy)

    # def build_sm1_policy(self) -> BlockchainModel.Policy:
    #     policy = np.zeros(self.state_space.size, dtype=int)
    #
    #     for i in range(self.state_space.size):
    #         a, h, be, v, fork = self.state_space.index_to_element(i)
    #
    #         if h > a:
    #             action = self.Action.Adopt
    #         elif (h == a - 1 and a >= 2) or a == self.max_fork:
    #             action = self.Action.Override
    #         elif (h == 1 and a == 1) and fork is self.Fork.Relevant:
    #             action = self.Action.Match
    #         else:
    #             action = self.Action.Wait
    #
    #         policy[i] = action
    #
    #     return tuple(policy)


if __name__ == '__main__':
    print('bitcoin_mdp module test')
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    mdp = BitcoinEclipseModel(0.35, 0.3, 0.5, 100)
    print(mdp.state_space.size)
    # p = mdp.build_sm1_policy()
    # print(p[:10])
