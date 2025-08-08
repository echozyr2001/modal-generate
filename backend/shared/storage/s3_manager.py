import boto3
import logging
from typing import Optional
from botocore.exceptions import ClientError
from shared.config import settings

logger = logging.getLogger(__name__)


class S3Manager:
    """Dedicated S3 storage manager"""
    
    def __init__(self, bucket_name: str = None, region: str = None):
        self.bucket_name = bucket_name or settings.s3_bucket_name
        self.region = region or settings.s3_region
        
        if not self.bucket_name:
            raise ValueError("S3 bucket name is required")
        
        self.s3_client = boto3.client("s3", region_name=self.region)
    
    def upload_file(self, file_path: str, s3_key: str) -> str:
        """Upload file to S3 and return the S3 key"""
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
            logger.info(f"Successfully uploaded {file_path} to s3://{self.bucket_name}/{s3_key}")
            return s3_key
        except ClientError as e:
            logger.error(f"Failed to upload {file_path} to S3: {e}")
            raise
    
    def download_file(self, s3_key: str, local_path: str) -> str:
        """Download file from S3 to local path"""
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Successfully downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return local_path
        except ClientError as e:
            logger.error(f"Failed to download {s3_key} from S3: {e}")
            raise
    
    def delete_file(self, s3_key: str) -> bool:
        """Delete file from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Successfully deleted s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete {s3_key} from S3: {e}")
            return False
    
    def file_exists(self, s3_key: str) -> bool:
        """Check if file exists in S3"""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            logger.error(f"Error checking file existence {s3_key}: {e}")
            return False
    
    def get_presigned_url(self, s3_key: str, expiration: int = 3600) -> str:
        """Generate presigned URL for S3 object"""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL for {s3_key}: {e}")
            raise
    
    def list_files(self, prefix: str = "") -> list:
        """List files in S3 bucket with optional prefix"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        except ClientError as e:
            logger.error(f"Failed to list files with prefix {prefix}: {e}")
            return []