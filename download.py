"""
download.py

下载 AISHELL-1 数据集后仅提取 test 子集（约 50MB），并解压到 data/AISHELL_test 目录下。
"""
import os
import urllib.request
import urllib.error
import tarfile

def download_aishell_test(output_dir="data/AISHELL_test"):
	"""
	下载完整的 AISHELL-1 数据集，提取其中 test/ 子目录到 output_dir。
	output_dir: 解压后存放 test 子集的根目录。
	"""
	# 完整数据包 URL
	full_url = "https://www.openslr.org/resources/33/data_aishell.tgz"
	# 准备目录和文件名
	os.makedirs(output_dir, exist_ok=True)
	archive_name = "data_aishell.tgz"
	archive_path = os.path.join(output_dir, archive_name)
	# 下载完整数据包
	if not os.path.exists(archive_path):
		print(f"Downloading full AISHELL-1 dataset to {archive_path} ...")
		urllib.request.urlretrieve(full_url, archive_path)
	else:
		print(f"Archive already exists at {archive_path}, skipping download.")
	# 解压，只提取 test/ 子目录
	print(f"Extracting test subset from {archive_path} to {output_dir} ...")
	with tarfile.open(archive_path, "r:gz") as tar:
		for member in tar.getmembers():
			# 仅提取 test/ 下的内容
			if member.name.startswith("test/"):
				tar.extract(member, path=output_dir)
	print("Extraction complete.")

if __name__ == '__main__':
	# 默认下载并解压到 data/AISHELL_test/test/
	download_aishell_test()
