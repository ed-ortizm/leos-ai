"""Install AI module for LEOs sat"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as file:
    long_description = file.read()

setup(
    name="leosAi",
    version="0.1.0",
    author="Edgar Ortiz",
    author_email="ed.ortizm@gmail.com",
    packages=find_packages(where="src", include=["[a-z]*"], exclude=[]),
    package_dir={"": "src"},
    description=(
        "Python Code For explainable artificial intelligence in astronomy"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ed-ortizm/xai-astronomy",
    license="MIT",
    keywords="astrophysics, Low Earth Orbit, Machine Learning",
)
