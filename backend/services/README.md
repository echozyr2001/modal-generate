# Music Generation Services Architecture

## 概述

本项目采用微服务架构，将原本在 `main.py` 中的单体应用拆分为多个独立的专门化服务。每个服务专注于特定的功能领域，提供更好的可扩展性、维护性和资源利用率。

## 服务架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Integrated Service (编排层)                    │
│                integrated_service.py                        │
│  • 提供与 main.py 相同的 API 接口                           │
│  • 协调调用其他服务完成完整工作流                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Lyrics    │ │    Music    │ │Cover Image  │
│   Service   │ │   Service   │ │   Service   │
│             │ │             │ │             │
│ • 歌词生成   │ │ • 音乐生成   │ │ • 封面生成   │
│ • 提示词生成 │ │ • ACE-Step  │ │ • SDXL-Turbo│
│ • 分类生成   │ │   Pipeline  │ │   Model     │
│ • LLM 推理   │ │             │ │             │
└─────────────┘ └─────────────┘ └─────────────┘
```

## 服务详情

### 1. Lyrics Service (`lyrics_service.py`)
- 歌词生成、音乐提示词生成、分类生成
- 使用 Qwen LLM 模型
- API: `/generate_lyrics`, `/generate_prompt`, `/generate_categories`

### 2. Music Service (`music_service.py`)
- 音乐生成使用 ACE-Step Pipeline
- 支持自定义歌词和器乐模式
- API: `/generate_music`, `/generate_demo_music`

### 3. Cover Image Service (`cover_image_service.py`)
- 专辑封面生成使用 SDXL-Turbo
- 支持多种艺术风格
- API: `/generate_cover_image`, `/get_available_styles`

### 4. Integrated Service (`integrated_service.py`)
- 服务编排器，提供完整工作流
- 与原 `main.py` API 兼容
- API: `/generate_from_description`, `/generate_with_lyrics`

## 服务管理

### Service Manager (`service_manager.py`)
统一的服务管理工具，支持批量操作和依赖管理。

#### 基本用法
```bash
# 列出所有服务
python service_manager.py list

# 运行单个服务
python service_manager.py run lyrics
python service_manager.py run music
python service_manager.py run image
python service_manager.py run integrated

# 部署单个服务
python service_manager.py deploy lyrics

# 运行所有服务 (按依赖顺序)
python service_manager.py run-all

# 部署所有服务
python service_manager.py deploy-all
```

#### 服务依赖关系
```
lyrics    ← 无依赖
music     ← 无依赖  
image     ← 无依赖
integrated ← 依赖: lyrics, music, image
```

## 共享基础设施

### Base Service Mixin (`shared/base_service.py`)
所有服务都继承自 `ServiceMixin`，提供统一的基础功能：

#### 核心功能
- **操作监控**: 超时检测、性能监控
- **文件管理**: 统一的文件存储和管理
- **健康检查**: 标准化的健康检查接口
- **错误处理**: 统一的异常处理和清理
- **配置管理**: 服务配置的标准化处理

#### 通用方法
```python
# 操作管理
start_operation(operation_id, operation_type)
end_operation(operation_id, success=True)
cleanup_operation(operation_id)

# 健康检查
health_check() -> Dict[str, Any]

# 文件管理
save_file(file_path, file_type) -> str
get_storage_mode() -> str

# 元数据创建
create_metadata(operation_id, model_name, start_time) -> GenerationMetadata
```

## 存储配置

### 直接下载模式（推荐）
```bash
# 文件直接返回给客户端，零存储费用
STORAGE_MODE=direct_download
USE_S3_STORAGE=false
```

### S3 存储模式
```bash
# 用于生产环境
STORAGE_MODE=s3
USE_S3_STORAGE=true
S3_BUCKET_NAME=your-bucket-name
```

## 客户端使用

### 直接下载模式示例
```python
import requests
import base64

# 生成封面图片
response = requests.post(
    "https://your-service.modal.run/generate_cover_image",
    json={"prompt": "electronic music cover", "style": "electronic"}
)

result = response.json()

# 直接获取文件数据
file_data = base64.b64decode(result['file_data'])

# 保存到本地
with open("cover.png", 'wb') as f:
    f.write(file_data)
```

## 部署和测试

### 快速部署
```bash
# 部署所有服务
cd backend
python service_manager.py deploy-all

# 测试服务
modal run -m services.integrated_service
```

### 验证配置
```bash
python test_storage.py  # 验证存储配置
```

---

## 快速开始

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **配置环境变量**
   ```bash
   export S3_BUCKET_NAME=your-bucket
   ```

3. **部署服务**
   ```bash
   python service_manager.py deploy-all
   ```

4. **测试服务**
   ```bash
   python service_manager.py run integrated
   ```

5. **开始使用**
   ```python
   import requests
   
   response = requests.post(
       "https://your-integrated-service.modal.run/generate_from_description",
       json={"full_described_song": "happy electronic music"}
   )
   ```

更多详细信息请参考各个服务的源代码和注释。