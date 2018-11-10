import numpy as np 
import argparse

__author__ = "Yang Long"
__version__ = "0.0.1"

__default_state_shape__ = 10, 10, 1
__filename__ = "model.h5"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
    """
    Play/Train \"Five Or More\" game with AI.
    Author: {author}
    Version: {version}
    """.format(author=__author__, version=__version__))

    parser.add_argument("--retrain", action='store_true', default=False, help="Re-Train AI")
    parser.add_argument("--train",  action='store_true', default=False, help="Train AI")
    parser.add_argument("--verbose", action='store_true', default=False, help="Verbose")
    parser.add_argument("--playai", action='store_true', default=False, help="Play by AI")
    parser.add_argument("--play", action='store_true', default=False, help="Play by human")

    args = parser.parse_args()
    verbose = args.verbose

    if args.train:
        if verbose:
            print("Continue to train AI model for the game.")

    if args.retrain:
        if verbose:
            print("Re-train the AI model for the game.")

        from train import TrainAI

        trainai = TrainAI(state_shape=__default_state_shape__,verbose=verbose)
        trainai.start(filename=__filename__)

        if verbose:
            print("The AI model is saved as the file [{0}].".format(__filename__))

    if args.playai:
        print("Play \"Five or More\" game by AI. Please close game in terminal after closing window (i.e, Press Ctrl+C).")
        
        # TODO Play Game with AI model

    if args.play:
        print("Play \"Five or More\" game. Please close game in terminal after closing window (i.e, Press Ctrl+C).")
        from game import GameEngine
    
        gameengine = GameEngine(state_shape=__default_state_shape__,verbose=verbose)
        gameengine.start()