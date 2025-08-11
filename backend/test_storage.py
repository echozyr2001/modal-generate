#!/usr/bin/env python3
"""
测试存储配置脚本
验证当前的存储设置是否正确
"""

import os

def test_storage_config():
    """测试存储配置"""
    print("=== 存储配置测试 ===")
    
    # 检查环境变量
    storage_mode = os.environ.get("STORAGE_MODE", "direct_download")
    use_s3 = os.environ.get("USE_S3_STORAGE", "false").lower() == "true"
    local_dir = os.environ.get("LOCAL_STORAGE_DIR", "/tmp/music_outputs")
    s3_bucket = os.environ.get("S3_BUCKET_NAME", "")
    
    print(f"STORAGE_MODE: {storage_mode}")
    print(f"USE_S3_STORAGE: {use_s3}")
    print(f"LOCAL_STORAGE_DIR: {local_dir}")
    print(f"S3_BUCKET_NAME: {s3_bucket}")
    
    if storage_mode == "direct_download":
        print(f"✅ 使用直接下载模式")
        print(f"   - 文件将直接返回给客户端（base64 编码）")
        print(f"   - 零存储费用，无需持久化存储")
        print(f"   - 临时文件目录: {local_dir}")
        
        # 创建临时目录
        if not os.path.exists(local_dir):
            print(f"创建临时文件目录: {local_dir}")
            os.makedirs(local_dir, exist_ok=True)
        print(f"✓ 临时文件目录已准备就绪")
        
    elif storage_mode == "s3" or use_s3:
        print(f"⚠️  当前配置使用 S3 存储")
        if not s3_bucket:
            print(f"❌ S3_BUCKET_NAME 未设置")
        else:
            print(f"✓ S3 存储桶: {s3_bucket}")
            
    elif storage_mode == "local":
        print(f"✓ 使用本地持久化存储模式")
        print(f"本地存储目录: {local_dir}")
        
        # 创建本地存储目录
        if not os.path.exists(local_dir):
            print(f"创建本地存储目录: {local_dir}")
            os.makedirs(local_dir, exist_ok=True)
        print(f"✓ 本地存储目录已准备就绪")
    
    else:
        print(f"❌ 未知的存储模式: {storage_mode}")
    
    # 检查 .env 文件
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"\n✅ .env 文件存在")
    else:
        print(f"\n⚠️  .env 文件不存在")
    
    print(f"\n=== 配置说明 ===")
    print("直接下载模式: STORAGE_MODE=direct_download (推荐)")
    print("S3 存储模式: STORAGE_MODE=s3 + S3 配置")

if __name__ == "__main__":
    test_storage_config()