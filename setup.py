from setuptools import setup, find_packages

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
    use_scm_version={'version_scheme': 'guess-next-dev',
                     'local_scheme': 'node-and-timestamp'},
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    url='https://github.com/ThalesGroup/ConceptBERT',
    license='',
    keywords="conceptBert",
)
