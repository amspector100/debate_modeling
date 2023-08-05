import os

def create_directory(output_dir):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	proc_dir = output_dir.replace("raw", "processed")
	if not os.path.exists(proc_dir):
		os.makedirs(proc_dir)
	return None
