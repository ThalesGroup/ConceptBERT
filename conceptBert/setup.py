from setuptools import setup, find_packages

from os.path import join

with open('README.md') as readme_file:
    readme = readme_file.read()


with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read().splitlines()


with open('CHANGELOG.md') as history_file:
    history = history_file.read()

def _version():
    """ Get the local package version."""
    path = join("conceptBert", "__version__.py")
    namespace = {}
    with open(path) as stream:
        exec(stream.read(), namespace)
    return namespace["__version__"]


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
    name='ConceptBert',
    long_description=readme + '\n\n' + history,
    package_data={'': ['*.md', '*.txt']},
    keywords="conceptBert",
    packages=find_packages(),
    install_requires=requirements,
    url='https://github.com/ThalesGroup/ConceptBERT',
    version=_version(),
    zip_safe=False,
    description=readme
)
