
import setuptools

with open("README.md", "r") as f:
    long_description = f.read()
setuptools.setup(
    name="robonnx", 
    version="0.0.1",
    author="Rhys Williams",
    author_email="rhysdgwilliams@gmail.com",
    description="An example onnx implementation of RoBERTa for demonstration purposes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/https://github.com/rhysdg/robonnx",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['transformers==4.21.1',
                      'onnxruntime-gpu==1.12.1',
                      'gdown',
                      'scipy',
                      'pytest'
    ],
    tests_require=['pytest'],
    python_requires='>=3.7',
)
