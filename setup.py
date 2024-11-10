from setuptools import setup, find_packages

setup(
    name="ai-powered-zoom",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "opencv-python",
        "numpy",
        "ultralytics",
        "pytubefix",
        "moviepy",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "run-zoom=src.main:main",
        ],
    },
)