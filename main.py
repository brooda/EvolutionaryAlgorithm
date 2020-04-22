import argparse
from EvolutionaryAlgorithm import EvolutionaryAlgorithm

'''
parser = argparse.ArgumentParser(description='Evolutionary algorithm')
parser.add_argument('goalfunction', type=str, help='Goal function name')

args = parser.parse_args()
'''

alg = EvolutionaryAlgorithm()
alg.Solve()