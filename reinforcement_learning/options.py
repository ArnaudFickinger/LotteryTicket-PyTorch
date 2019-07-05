###
'''
June 2019
Code by: Arnaud Fickinger
'''
###

import argparse
import os


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--batch_size', type=int, default=64)
        self.parser.add_argument('--gamma', type=float, default=0.95)
        self.parser.add_argument('--eps_start', type=float, default=0.9)
        self.parser.add_argument('--lr', type=float, default=0.001)
        self.parser.add_argument('--eps_end', type=int, default=0.01)
        self.parser.add_argument('--eps_decay', type=int, default=200)
        self.parser.add_argument('--target_update', type=int, default=10)
        self.parser.add_argument('--episodes', type=int, default=500)
        self.parser.add_argument('--render_at', type=int, default=800)
        self.parser.add_argument('--h1_dim', type=int, default=300)
        self.parser.add_argument('--h2_dim', type=int, default=100)

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt