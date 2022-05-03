from typing import Dict, Union, Optional, Any
from pathlib import Path
import logging
import tempfile
import json
from pandas import DataFrame

from common.pandas_utils import DataFrameSummary
from common.types import FileFormat

logger = logging.getLogger(__name__)


class WriteToFile:
	def __init__(local_dir: str, file_version: Union[int, str]) -> None:
		self.local_dir = local_dir
		self.file_version = file_version

	def save(self, data: Any, filename: str, **kwargs):
		raise NotImplementedError

	def save_metadata(self, metadata: Dict[Any, Any], filename: str) -> Path:
		metadata_path = self._build_full_path(self.local_dir, filename, self.file_version,
											  FileFormat.JSON, is_meta=True)
		# TODO: make sure metadata is key value, where value is not DF, if not convert before write
		with open(file_name, 'a', encoding="utf-8") as f: 
            for name, value in metadata.items():
                if isinstance(value, pd.DataFrame):
                    content = df.to_json(orient="records")
                else:
                    content = value
                json.dump({name: content}, f, indent=2)
        return metadata_path

	@staticmethod
	def _build_full_path(local_dir: str, filename: str, file_version: Union[int, str], 
						 file_format: FileFormat, is_meta: bool=False) -> Path:
		meta_suffix = "_meta" if is_meta else ""
		return Path(local_dir, f"{filename}_{file_version}{meta_suffix}.{file_format.value}")



class CommonFIleWriter(WriteToFile):

	def save_as_csv(self, data: Union[DataFrame, List[Any]], filename: str, **kwargs) -> Path:
		"""
		csv_writer = CommonFIleWriter(self.local_dir, self.file_version)
		csv_writer.save_as_csv(data, filename, **kwargs)
		if write_metadata:
			metadata = DataFrameSummary(data).get_metadata({}, groupby_cols, value2metrics)
			self.save_metadata(metadata, filename)
		"""
		file_path = self._build_full_path(self.local_dir, filename, self.file_version, FileFormat.CSV)
		if isinstance(data, list):
			try:
				data = DataFrame(data)
			except ValueError as e:
				logger.error(f"Error {e} occur when convert list to DataFrame!")
				data = DataFrame([], dtype=object)
		if not data.empty:
			data.to_csv(file_path, index=False, encoding="utf-8", **kwargs)
			logger.debug(f'Store data into {file_path} as {FileFormat.CSV}')
		return file_path

	def save_as_json(self, data: Dict, filename: str, **kwargs) -> Path:
		file_path = self._build_full_path(self.local_dir, filename, self.file_version, FileFormat.JSON)
		with open(file_path, "w", encoding="utf-8") as f:
			json.dump(data, f)
		logger.debug(f'Store data into {file_path} as {FileFormat.JSON}')
		return file_path


	def save_as_pickle(self, data: Any, filename: str, **kwargs) -> Path:
		file_path = self._build_full_path(self.local_dir, filename, self.file_version, FileFormat.PICKLE)
		with open(file_path, "wb") as f:
			pickle.dump(data, f)
		logger.debug(f'Store data into {file_path} as {FileFormat.PICKLE}')
		return file_path

	def save_as_joblib(self, data: Any, filename: str, **kwargs) -> Path:
		file_path = self._build_full_path(self.local_dir, filename, self.file_version, FileFormat.JOBLIB)
		joblib.dump(data, filename)
		logger.debug(f'Store data into {file_path} as {FileFormat.JOBLIB}')
		return file_path
