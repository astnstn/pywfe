from setuptools import setup, find_packages

setup(
    name='pywfe',
    version='0.1.0',    # Update this for new versions
    description='Python implementation of Wave Finite Element method',
    url='https://github.com/yourusername/pywfe',
    author='Austen Stone',  # Update this
    author_email='austenkentellstone@gmail.com',
    license='MIT',  # Or another license
    packages=find_packages(),
    install_requires=[
        matplotlib,
	numpy,
	pyevtk,
	scipy,
	tqdm
    ],
    classifiers=[
        # Classifiers help users find your project by categorizing it.
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
