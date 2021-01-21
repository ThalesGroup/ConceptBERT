from setuptools import setup, find_packages

from os.path import join

with open('README.md') as readme_file:
    readme = readme_file.read()


with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read().splitlines()


with open('CHANGELOG.md') as history_file:
    history = history_file.read()


setup(
    author="Thales",
    author_email='developer@ca.thalesgroup.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
    name='Kilbert',
    license="MIT license",
    long_description=readme + '\n\n' + history,
    keywords="kilbert",
    packages=find_packages(),
    install_requires=requirements,
    url='https://sc01-trt.thales-systems.ca/gitlab/human-ai-dialog/kilbert/',
    description=readme
)
