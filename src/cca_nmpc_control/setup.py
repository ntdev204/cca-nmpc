from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'cca_nmpc_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test', 'test.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'numpy', 'casadi'],
    zip_safe=True,
    maintainer='CCA-NMPC Team',
    maintainer_email='dev@example.com',
    description='CCA-NMPC control node + NMPC solver library (Sections 11-12)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nmpc_controller_node = cca_nmpc_control.nmpc_controller_node:main',
        ],
    },
)
