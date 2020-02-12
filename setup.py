from setuptools import setup, find_packages

setup(
    name='ca-thalesgroup-human-ai-dialog-vilbert',
    use_scm_version={'version_scheme': 'guess-next-dev',
                     'local_scheme': 'node-and-timestamp'},
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    url='https://sc01-trt.thales-systems.ca/gitlab/human-ai-dialog/vilbert/',
    license='',
    author='Thales',
    author_email='',
    description=''
)

