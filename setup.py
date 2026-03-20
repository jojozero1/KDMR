from setuptools import setup, find_packages

setup(
    name="kdmr",
    version="0.1.0",
    description="Kinodynamic Motion Retargeting for Humanoid Locomotion",
    author="KDMR Team",
    author_email="kdmr@example.com",
    url="https://github.com/kdmr/kdmr",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "mujoco>=3.0.0",
        "mink>=0.0.5",
        "cvxpy>=1.3.0",
        "matplotlib>=3.5.0",
        "rich>=13.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "isort>=5.0.0"],
        "all": ["casadi>=3.6.0", "torch>=2.0.0", "imageio>=2.0.0", "smplx>=0.1.28"],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
)
