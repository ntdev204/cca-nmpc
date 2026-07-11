from glob import glob
import os

from setuptools import setup

package_name = "cca_nmpc_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="CCA-NMPC Team",
    maintainer_email="dev@example.com",
    description="CCA-NMPC bringup package",
    license="MIT",
)
