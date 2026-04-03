from setuptools import find_packages, setup

package_name = 'mocap_relay'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dequan Ou',
    maintainer_email='d.ou@tudelft.nl',
    description='Relays motion capture data to PX4 with timestamp synchronization.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mocap_relay_node = mocap_relay.mocap_relay_node:main'
        ],
    },
)