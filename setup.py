from setuptools import setup, find_packages

setup(
    name='mujoco_kitchen',
    version='1.0',
    packages=find_packages(include=["mujoco_kitchen", "mujoco_kitchen.*"])
)