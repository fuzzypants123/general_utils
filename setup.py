from setuptools import setup, find_packages


setup(
    name="general_utils",
    version="1.0",
    author="weiwang",
    author_email="weiwang201806@gmail.com",
    description="my python general utils",

    url="https://github.com/fuzzypants123/general_utils", 
    python_requires='>=3.5',
    
    install_requires=[
        'numpy>=1.19.0',
        'opencv_python>=4.2.0.34',
        'Pillow>=8.0.0',
        'torch>=1.6.0',
        'tqdm>=4.11.0'
    ],
    packages=find_packages()
)