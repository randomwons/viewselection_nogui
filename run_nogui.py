import argparse

from python.auto import run

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default='objs/armadillo.obj')
    parser.add_argument("-d", "--data", default='dataset/armadillo')
    parser.add_argument("-p", "--policy", default=0, type=int)
    parser.add_argument("-s", "--show", action='store_true')
    args = parser.parse_args()
    run(args)