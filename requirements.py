import sys
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict

assert sys.version_info >= (3, 6), 'Incorrect Python verion. Requires Python >= 3.5'

print('Python version requirement satisfied!')

dependencies = [
	'torch==1.1.0',
	'torchvision==0.2.2',
 	'numpy>=1.16.1',
 	'scipy>=1.1.0',
 	'tqdm>=4.19.9',
 	'nltk>=3.4.1'
]

pkg_resources.require(dependencies)
print('All required packages found!')