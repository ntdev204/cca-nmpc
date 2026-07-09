from setuptools import setup
import os
from glob import glob

package_name = 'cca_nmpc_context'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='CCA-NMPC Team',
    maintainer_email='dev@example.com',
    description='Continuous context estimation for CCA-NMPC (Eqs. 7-9, 12.6-12.7)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'context_node = cca_nmpc_context.context_node:main',
        ],
    },
)
