# 服务部署和测试指南

## 快速开始

### 1. 部署所有服务

使用服务管理器按正确顺序部署所有服务：

```bash
cd backend
python service_manager.py deploy-all
```

或者手动逐个部署：

```bash
# 1. 部署基础服务（无依赖）
python service_manager.py deploy lyrics
python service_manager.py deploy music  
python service_manager.py deploy image

# 2. 部署编排服务（依赖其他服务）
python service_manager.py deploy integrated
```

### 2. 配置服务 URL

部署完成后，你需要配置 integrated service 的依赖服务 URL。

#### 方法 1: 环境变量配置
```bash
export LYRICS_SERVICE_URL="https://your-lyrics-service.modal.run"
export MUSIC_SERVICE_URL="https://your-music-service.modal.run"
export IMAGE_SERVICE_URL="https://your-image-service.modal.run"
```

#### 方法 2: 修改配置文件
在 `shared/config.py` 中更新服务 URL：

```python
# Service URLs for integrated service
lyrics_service_url: str = "https://your-lyrics-service.modal.run"
music_service_url: str = "https://your-music-service.modal.run"
image_service_url: str = "https://your-image-service.modal.run"
```

### 3. 测试服务

#### 测试单个服务
```bash
# 测试歌词服务
modal run -m services.lyrics_service

# 测试音乐服务
modal run -m services.music_service

# 测试图片服务
modal run -m services.cover_image_service

# 测试集成服务
modal run -m services.integrated_service
```

#### 使用服务管理器测试
```bash
# 运行所有服务测试
python service_manager.py run-all
```

## 详细部署步骤

### 步骤 1: 准备环境

#### 选择存储模式

**选项 A: 直接下载模式（推荐，零存储费用）**
```bash
# 文件直接返回给客户端，无存储费用
export STORAGE_MODE=direct_download
export USE_S3_STORAGE=false
```

**选项 B: S3 存储（用于生产环境）**
```bash
# S3 存储配置
export STORAGE_MODE=s3
export USE_S3_STORAGE=true
export S3_BUCKET_NAME="your-music-generation-bucket"
export S3_REGION="us-east-1"
```

#### Modal 认证
```bash
modal token set --token-id your-token-id --token-secret your-token-secret
```

#### 验证配置
```bash
cd backend
python test_storage.py  # 验证存储模式配置正确
```

### 步骤 2: 部署基础服务

#### 部署 Lyrics Service
```bash
modal deploy services.lyrics_service
```

预期输出：
```
✓ Created function LyricsGenServer.*
✓ Created web endpoint for LyricsGenServer.generate_lyrics
✓ Created web endpoint for LyricsGenServer.generate_prompt  
✓ Created web endpoint for LyricsGenServer.generate_categories
✓ Created web endpoint for LyricsGenServer.health_check
```

#### 部署 Music Service
```bash
modal deploy services.music_service
```

预期输出：
```
✓ Created function MusicGenServer.*
✓ Created web endpoint for MusicGenServer.generate_music
✓ Created web endpoint for MusicGenServer.generate_demo_music
✓ Created web endpoint for MusicGenServer.health_check
```

#### 部署 Cover Image Service
```bash
modal deploy services.cover_image_service
```

预期输出：
```
✓ Created function CoverImageGenServer.*
✓ Created web endpoint for CoverImageGenServer.generate_cover_image
✓ Created web endpoint for CoverImageGenServer.get_available_styles
✓ Created web endpoint for CoverImageGenServer.download_image
```

### 步骤 3: 获取服务 URL

部署完成后，记录每个服务的 URL：

```bash
# 查看已部署的应用
modal app list

# 查看特定应用的端点
modal app show your-app-name
```

### 步骤 4: 配置 Integrated Service

更新 `integrated_service.py` 中的服务 URL 或设置环境变量。

### 步骤 5: 部署 Integrated Service

```bash
modal deploy services.integrated_service
```

## 测试验证

### 健康检查
```bash
# 检查服务状态
curl https://your-service.modal.run/health_check
```

### 功能测试
```bash
# 测试服务
modal run -m services.integrated_service
```

## 故障排除

### 常见问题
- **模型加载失败**: 检查 GPU 配置和内存
- **服务超时**: 增加超时时间或检查资源配置
- **存储配置错误**: 运行 `python test_storage.py` 验证

### 调试工具
```bash
modal logs your-app-name        # 查看日志
modal app list                  # 查看应用状态
modal run -m services.service_name  # 本地测试
```

---

## 总结

按照这个指南，你应该能够：

1. ✅ 成功部署所有微服务
2. ✅ 配置服务间的通信
3. ✅ 验证功能正确性
4. ✅ 监控服务健康状态
5. ✅ 处理常见问题

如果遇到问题，请检查：
- Modal 认证配置
- 环境变量设置
- 服务 URL 配置
- GPU 资源可用性
- S3 权限配置

需要帮助时，可以查看详细的错误日志或联系技术支持。