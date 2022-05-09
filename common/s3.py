from typing import List, IO, Union, Optional
import logging
from os import path, makedirs
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def _build_full_path(folder_paths: Optional[str, List[str]])
	_paths = []
	if isinstance(folder_paths, str):
		_paths.append(folder_paths.rstrip("/"))
	else:
		for _path in folder_paths:
			_paths.append(_path.rstrip("/"))
	return "/".join(_paths)


# TODO: better capture common s3 exceptions
def StoreDataToS3:

	def __init__(self, bucket: str, key: str):
        self.client = boto3.client("s3")
        self.resource = boto3.resource("s3")
        self.bucket = bucket
        self.key = key

    def check_asset_existence(self, s3_key: str) -> bool:
    	try:
    		self.client.head_object(Bucket=self.bucket, key=s3_key)
    		return True
    	except ClientError:
    		return False

	def upload_asset(self, filename: str, extend_key: Optional[str, List[str]] = None) -> None:
        key = _build_full_path(self.key, extend_key)
        self.client.upload_file(Filename=filename, Bucket=self.bucket, Key=key)
        logger.info(f"Uploaded file {filename} to {self.bucket}:{key}")

    def upload_asset_object(self, asset_obj: IO, extend_key: Optional[str, List[str]]=None) -> None:
    	key = _build_full_path(self.key, extend_key)
    	self.client.upload_fileobj(Fileobj=asset_obj, Bucket=self.bucket, Key=key)
    	logger.info(f"Uploaded file object to {self.bucket}:{key}")

    def delete_s3_asset(self, extend_key: Optional[str, List[str]]=None) -> None:
    	key = _build_full_path(self.key, extend_key)
    	response = self.resource.Object(self.bucket, key).delete()
    	assert response['ResponseMetadata']['HTTPStatusCode'] == 204

	@overrides
	def store(self, filename: str, extend_key: Optional[str, List[str]]=None, refresh: bool=False) -> bool:
		key = _build_full_path(self.key, extend_key)
        if self.check_asset_existence(key) and not refresh:
            logger.debug(f"Asset {key} already exists on S3 and no refresh requirement. Skip uploading.")
            return False
        else:
            self.upload_asset(filename=filename, extend_key=extend_key)
            return True


  def DownloadDataFromS3:

  	def __init__(self, bucket: str, key: str):
        self.client = boto3.client("s3")
        self.resource = boto3.resource("s3")
        self.bucket = bucket
        self.key = key

 	def download_asset(self, filename: str, extend_key: Optional[str, List[str]]=None):
 		file_dir = path.dirname(filename)
 		makedirs(file_dir, exist_ok=True)
 		key = _build_full_path(self.key, extend_key)
 		self.client.download_file(Bucket=self.bucket, Key=key, Filename=filename)
 		logger.info(f"Downloaded asset to {filename} from S3 {self.bucket}:{key}")

 	def download_asset_object(self, asset_obj: IO, extend_key: Optional[str, List[str]]=None):
 		key = _build_full_path(self.key, extend_key)
 		self.client.download_fileobj(Bucket=self.bucket, Key=key, Fileojb=asset_obj)
 		logger.info(f"Downloaded asset object from {self.bucket}:{key}")

 	def get_bucket_keys(self, prefix: str):
 		response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
 		keys = response.get("Content", [])
 		logger.info(f"Got S3 Keys: {keys} from bucket: {self.bucket}")



