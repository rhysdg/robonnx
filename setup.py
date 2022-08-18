
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="inference", 
    version="0.0.1",
    author="Rhys Williams",
    author_email="rhysdgwilliams@gmail.com",
    description="An onnx implementation of RoBERTa",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusrname/projectnam",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['transformers==4.21.1',
                      'onnxruntime-gpu==1.12.1',
                      'gdown',
                      'scipy'
    ],
    python_requires='>=3.7',
)