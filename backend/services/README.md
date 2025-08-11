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
**专门负责文本生成和 LLM 推理**

#### 功能
- 歌词生成 (Lyrics Generation)
- 音乐提示词生成 (Prompt Generation)  
- 音乐分类生成 (Category Generation)
- LLM 推理服务

#### 技术栈
- **模型**: Qwen LLM (可配置)
- **GPU**: 支持 CUDA 加速
- **框架**: Transformers, PyTorch

#### API 端点
```python
POST /generate_lyrics          # 生成歌词
POST /generate_prompt          # 生成音乐提示词
POST /generate_categories      # 生成音乐分类
GET  /health_check            # 健康检查
GET  /service_info            # 服务信息
```

#### 配置示例
```python
lyrics_config = ServiceConfig(
    service_name="lyrics-generator",
    gpu_type=GPUType.A10G,
    scaledown_window=300,
    max_runtime_seconds=1800
)
```

### 2. Music Service (`music_service.py`)
**专门负责音乐生成**

#### 功能
- 音乐生成 (使用 ACE-Step Pipeline)
- 支持自定义歌词和器乐模式
- 音频参数调节 (时长、推理步数、引导比例等)

#### 技术栈
- **模型**: ACE-Step Pipeline
- **GPU**: 高性能 GPU (A100/H100)
- **格式**: WAV 音频输出

#### API 端点
```python
POST /generate_music           # 生成音乐 (返回 base64)
POST /generate_demo_music      # 生成演示音乐
GET  /health_check            # 健康检查
GET  /service_info            # 服务信息
```

#### 请求参数
```python
class MusicGenerationRequest:
    prompt: str                # 音乐生成提示词
    lyrics: str               # 歌词内容
    audio_duration: float     # 音频时长 (秒)
    inference_steps: int      # 推理步数
    guidance_scale: float     # 引导比例
    seed: int                 # 随机种子
    instrumental: bool        # 是否为器乐
```

### 3. Cover Image Service (`cover_image_service.py`)
**专门负责封面图片生成**

#### 功能
- 专辑封面生成
- 多种艺术风格支持
- 图片尺寸自定义
- S3 存储集成

#### 技术栈
- **模型**: SDXL-Turbo
- **GPU**: 支持 CUDA 加速
- **格式**: PNG 图片输出

#### API 端点
```python
POST /generate_cover_image     # 生成封面图片
GET  /get_available_styles     # 获取可用风格
GET  /download_image          # 下载图片 (base64)
GET  /health_check            # 健康检查
GET  /service_info            # 服务信息
```

#### 支持的风格
- `album cover art` - 专业专辑封面
- `vintage` - 复古风格
- `modern` - 现代简约
- `abstract` - 抽象艺术
- `photorealistic` - 写实摄影
- `illustration` - 数字插画
- `grunge` - 朋克风格
- `electronic` - 电子音乐风格

### 4. Integrated Service (`integrated_service.py`)
**服务编排器 - 提供完整的音乐生成工作流**

#### 功能
- 协调调用其他服务
- 提供与原 `main.py` 相同的 API 接口
- 完整的音乐生成工作流编排
- 错误处理和重试机制

#### API 端点 (与 main.py 兼容)
```python
POST /generate_from_description        # 从描述生成完整音乐包
POST /generate_with_lyrics            # 使用自定义歌词生成
POST /generate_with_described_lyrics   # 使用描述的歌词生成
GET  /health_check                    # 健康检查
GET  /service_info                    # 服务信息
```

#### 工作流程
1. **接收用户请求**
2. **调用 Lyrics Service** 生成歌词/提示词
3. **调用 Music Service** 生成音频
4. **调用 Cover Image Service** 生成封面
5. **调用 Lyrics Service** 生成分类标签
6. **整合结果并返回**

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

## 部署和配置

### Modal 部署
每个服务都是独立的 Modal 应用，可以单独部署和扩展：

```python
# 服务配置示例
ServiceConfig(
    service_name="lyrics-generator",
    gpu_type=GPUType.A10G,
    scaledown_window=300,
    max_runtime_seconds=1800
)
```

### 环境变量
```bash
# S3 存储配置
S3_BUCKET_NAME=your-bucket-name

# 服务 URL 配置 (用于 integrated service)
LYRICS_SERVICE_URL=https://your-lyrics-service.modal.run
MUSIC_SERVICE_URL=https://your-music-service.modal.run
IMAGE_SERVICE_URL=https://your-image-service.modal.run
```

## 监控和日志

### 健康检查
每个服务都提供标准化的健康检查端点：

```json
{
  "status": "healthy",
  "model_loaded": true,
  "service_name": "lyrics-generator",
  "gpu_type": "A10G",
  "uptime": 3600,
  "active_operations": 2,
  "total_operations": 150
}
```

### 日志格式
统一的日志格式，便于监控和调试：

```
[operation_id] Action: Description
[abc123] Generating lyrics: happy song about sunshine...
[abc123] Lyrics generated successfully
```

## 性能优化

### 资源配置
- **Lyrics Service**: CPU 密集型，适中 GPU
- **Music Service**: 高性能 GPU，多核 CPU
- **Image Service**: 适中 GPU，快速存储
- **Integrated Service**: 轻量级，主要用于编排

### 缓存策略
- 模型缓存在 HuggingFace volume
- 生成结果可选择本地或 S3 存储
- 支持操作超时和自动清理

## 错误处理

### 统一错误响应
```json
{
  "detail": "Error description",
  "status_code": 500,
  "operation_id": "abc123",
  "service": "lyrics-generator"
}
```

### 常见错误码
- `400`: 请求参数错误
- `408`: 操作超时
- `500`: 内部服务错误
- `507`: GPU 内存不足

## 测试

每个服务都包含完整的测试套件：

```bash
# 测试单个服务
python -m services.lyrics_service
python -m services.music_service
python -m services.cover_image_service
python -m services.integrated_service
```

测试包括：
- 模型加载验证
- API 端点测试
- 错误处理验证
- 性能基准测试

## 迁移指南

### 从 main.py 迁移
1. **部署专门化服务**: 先部署 lyrics, music, image 服务
2. **配置服务 URL**: 在 integrated service 中配置其他服务的 URL
3. **部署编排服务**: 部署 integrated service
4. **更新客户端**: 将客户端请求指向 integrated service
5. **验证功能**: 确保所有功能正常工作
6. **停用旧服务**: 停用原来的 main.py 服务

### API 兼容性
Integrated Service 提供与 `main.py` 完全相同的 API 接口，确保无缝迁移。

## 最佳实践

### 开发
- 使用 `ServiceMixin` 基类确保一致性
- 实现完整的错误处理和日志记录
- 添加详细的 API 文档和类型注解
- 编写全面的测试用例

### 部署
- 根据负载需求配置合适的 GPU 类型
- 设置合理的超时和扩缩容参数
- 监控服务健康状态和性能指标
- 实施适当的安全措施和访问控制

### 监控
- 定期检查服务健康状态
- 监控资源使用情况和成本
- 设置告警机制
- 定期备份重要数据和配置

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