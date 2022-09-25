from setuptools import find_packages, setup

requirements = [
    "networkx>=2.8.4",
    "numpy>=1.22.3",
    "pyyaml>=6.0",
    "scipy>=1.8.0",
    "torch>=1.11.0",
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
    python_requires=">=3.8",
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
