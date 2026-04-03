from setuptools import setup

package_name = 'learning_flight'

setup(
 name=package_name,
 version='0.0.0',
 packages=[package_name],
 data_files=[
     ('share/ament_index/resource_index/packages',
             ['resource/' + package_name]),
     ('share/' + package_name, ['package.xml']),
     ('share/' + package_name + '/launch', ['launch/learning_launch.py']),
     ('share/' + package_name + '/config', ['config/params.yaml']),
   ],
 install_requires=['setuptools'],
 zip_safe=True,
 maintainer='Dequan Ou',
 maintainer_email='d.ou@tudelft.nl',
 description='ROS 2 package for automated learning flights with PX4.',
 license='MIT',
 tests_require=['pytest'],
 entry_points={
     'console_scripts': [
             'learning = learning_flight.learning_control:main',
     ],
   },
)