from setuptools import setup, find_packages


setup(
    name="gutils",
    version="0.0.1",
    author="weiwang",
    author_email="weiwang201806@gmail.com",
    description="my python general utils",

    url="https://github.com/fuzzypants123/general_utils", 
    python_requires='>=3.6',
    packages=find_packages(exclude=["test"]),
    install_requires=[
        'numpy>=1.19.0',
        'opencv-python>=4.2.0.34',
        'Pillow>=8.0.0',
        'torch>=1.6.0',
        'tqdm>=4.11.0',
        'matplotlib>=3.6.0',
    ],
    
)