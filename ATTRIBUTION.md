# Attribution and sources

Original code, simplified robot MJCF, experiment data, report, and rendered media: Copyright (c) 2026 Ehsan Cheraghi, MIT License. The robot is a generic educational model, not a reproduction of a commercial humanoid.

- [MuJoCo](https://github.com/google-deepmind/mujoco), Google DeepMind, Apache-2.0. Version 3.3.7 is the tested simulation dependency. [Python documentation](https://mujoco.readthedocs.io/en/stable/python.html) informed use of the model/data, stepping, and rendering APIs.
- [The Cross Entropy Method for Fast Policy Search](https://cdn.aaai.org/ICML/2003/ICML03-068.pdf), Shie Mannor, Reuven Rubinstein, Yohai Gat, ICML 2003. Algorithmic reference; no paper text or code is bundled.
- [Gymnasium's MuJoCo environments](https://gymnasium.farama.org/environments/mujoco/) were consulted for context. This project uses its own model and direct MuJoCo API, not a Gymnasium environment or a standard benchmark score.
- NumPy provides numerical arrays and random sampling. Pillow and ImageIO provide optional media encoding; ImageIO-FFmpeg supplies the optional encoder. Dependencies retain their own licenses and are installed separately.

There are no pretrained third-party policies, robot mesh downloads, commercial simulation services, or real-robot recordings in this project.
