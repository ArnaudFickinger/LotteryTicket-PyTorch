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
        self.parser.add_argument('--batch_size', type=int, default=60)
        self.parser.add_argument('--epochs', type=int, default=3)
        self.parser.add_argument('--lr', type=float, default=1.2e-3)
        self.parser.add_argument('--rate', type=int, default=20)
        self.parser.add_argument('--output_rate', type=int, default=10)
        self.parser.add_argument('--h1_dim', type=int, default=300)
        self.parser.add_argument('--h2_dim', type=int, default=100)
        self.parser.add_argument('--record_every', type=int, default=50)
        self.parser.add_argument('--plot', dest='plot', action='store_true', default=False)

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt