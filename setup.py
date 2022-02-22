from setuptools import setup, find_packages
import os

# Fill datafiles you want to deploy
data_files = []


# List your package requirements.
requires = []

setup(
	name='stochoptipy',
	version='1.0',
	# declare your packages
	packages=find_packages(where='src', exclude=("test", )),
	package_dir={"": "src"},
	# include data files
	data_files=data_files,
	install_requires=requires,
	# If you want to create a cli list your entry points in the following format:
	# [console_scripts]
	# 'command_name' = 'path_to_python_file:function_to_call'
	entry_points="",
	# optional: test_command = "your test command"
	# optional: doc_command = "your doc command"

	# Enable build-time format checking
	check_format=True,
	# Enable build-time format checking
	test_flake8=True,
	url='',
	license='',
	author='David Corredor',
	author_email='d.corredor@uniandes.edu.co',
	description=''
)
