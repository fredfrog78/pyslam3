from setuptools import setup, find_packages

setup(
    name="wildcat_slam",
    version="0.1.0",
    description="Wildcat SLAM: Online Continuous-Time 3D Lidar-Inertial SLAM",
    author="AI Agent", # Or your name/organization
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    entry_points={
        'console_scripts': [
            'wildcat_slam = wildcat_slam.cli:main',
        ],
    },
    install_requires=[
        "numpy",
        "scipy", # Will be needed later
    ],
    extras_require={
        "dev": [
            "pytest",
            "flake8",
            "black",
        ]
    }
)
