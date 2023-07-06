# mujoco_kitchen

Cloned and Edited from this repository: https://github.com/google-research/relay-policy-learning

## Getting Started (User)

Be sure to install mujoco_py properly. For further instructions see: https://github.com/openai/mujoco-py

After installing mujoco_py, export the repo's PYTHONPATH:

```
export PYTHONPATH={your-path-to-mujoco_kitchen}/adept_envs
```

Finally, be sure to use virtual environments in your code (Python==3.8), and pip install (After including in the PYTHONPATH)

```
pip install -e .
```

Now you can run the test script

```
python test.py
```

