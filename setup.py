from setuptools import find_packages, setup

requirements = [
    "networkx",
    "numpy",
    "pyyaml",
    "scipy",
    "torch",
    "webdataset",
    "matplotlib",
    "tqdm",
    "pandas",
    "joblib",
    "wandb",
    "pyreadline3",
    "codecarbon",
    "tensorflow_gnn",
    "tensorflow",
]
setup(
    name="spreadnet",
    version="0.0.1",
    description="""
    Spreadnet
    """,
    # long_description=open('../README.rst').read(),
    # long_description_content_type="text/markdown",
    author="Spreadnet Team",
    # author_email='shenghui.li@it.uu.se',
    # url='https://bladesteam.github.io/',
    # py_modules=['blades'],
    python_requires=">=3.7",
    license="Apache License 2.0",
    zip_safe=False,
    # entry_points={
    #     'console_scripts': [""]
    # },
    install_requires=requirements,
    keywords="Graph Neural Networks",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
