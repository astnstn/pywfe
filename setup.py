from setuptools import setup, find_packages

setup(
    name='pywfe',
    version='0.1.0',    # Update this for new versions
    description='Python implementation of Wave Finite Element method',
    url='https://github.com/yourusername/pywfe',  # Update this
    author='Your Name',  # Update this
    author_email='your.email@example.com',  # Update this
    license='MIT',  # Or another license
    packages=find_packages(),
    install_requires=[
        # List your package dependencies here
        # For example: 'numpy', 'scipy', etc.
    ],
    classifiers=[
        # Classifiers help users find your project by categorizing it.

        # For a list of valid classifiers, see https://pypi.org/classifiers/

        # For example:
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
