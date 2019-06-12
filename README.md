# AlphaFiveormore
Play "Five or More" game with Artificial intelligence 

![](./pics/fiveormore.png)

## Features
+ Home-made ["Five or More"](https://wiki.gnome.org/Apps/Five%20or%20more) game
+ Curiosity-driven Exploration by Self-supervised Prediction [1,2]

## How to use
+ Play "Five or more" game by yourself

        cd alphafiveormore
        python alphafiveormore.py --playai

+ Re-train a new AI model

        cd alphafiveormore
        python alphafiveormore.py --retrain --verbose

+ Continue to train the AI model

        cd alphafiveormore
        python alphafiveormore.py --train --verbose

+ View the AI model to play game

        cd alphafiveormore
        python alphafiveormore.py --playai --verbose

Press key `Space` to evaluate the step

## Status
GameEngine is currently inadequate. Working on it

## References
1. Pathak, Deepak, et al. "Curiosity-driven exploration by self-supervised prediction." International Conference on Machine Learning (ICML). Vol. 2017. 2017.
2. Achiam, Joshua, and Shankar Sastry. "Surprise-based intrinsic motivation for deep reinforcement learning." arXiv preprint arXiv:1703.01732 (2017).

## E-mail
longyang_123@yeah.net  
You're most welcome to contact with me to discuss any details about this project