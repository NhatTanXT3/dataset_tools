from setuptools import setup, find_packages

setup(
    name="euroc_dataset_tools",
    version="1.0.0",
    description="Python implementation of ASL dataset tools for EuRoC MAV datasets",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0", 
        "PyYAML>=5.4.0",
        "transformations>=2021.6.6",
    ],
    extras_require={
        "viz": ["rerun-sdk>=0.8.0", "opencv-python>=4.5.0"],
    },
    python_requires=">=3.8",
) 