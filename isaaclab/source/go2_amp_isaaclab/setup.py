from setuptools import find_packages, setup

setup(
    name="go2_amp_isaaclab",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={"go2_amp_isaaclab": ["config/*.toml"]},
    zip_safe=False,
)
