# 快速开始指南

## 🎯 直接下载模式 - 零存储费用

文件直接以 base64 编码返回给客户端，无需任何存储费用。

## ⚡ 快速部署

### 1. 检查现有应用（如果遇到 endpoints 限制）

```bash
modal app list  # 查看当前应用
modal app stop <app-id>  # 停止不需要的应用
```

### 2. 验证配置

```bash
python test_storage.py
```

### 3. 部署服务

```bash
python service_manager.py deploy-all
```

### 4. 测试服务

```bash
modal run -m services.integrated_service
```

## 🚨 Endpoints 限制解决方案

如果遇到 "reached limit of 8 web endpoints" 错误：

1. **检查应用**: `modal app list`
2. **停止应用**: `modal app stop <app-id>`
3. **重新部署**: 现在每个服务只有 1-3 个 endpoints

## 💻 客户端使用

```python
import requests
import base64

# 生成封面图片
response = requests.post(
    "https://your-service.modal.run/generate_cover_image",
    json={"prompt": "electronic music cover", "style": "electronic"}
)

result = response.json()

# 保存到本地
file_data = base64.b64decode(result['file_data'])
with open("cover.png", 'wb') as f:
    f.write(file_data)
```

## 📁 文件结构

```
backend/
├── services/
│   ├── lyrics_service.py      # 歌词生成服务
│   ├── music_service.py       # 音乐生成服务
│   ├── cover_image_service.py # 封面生成服务
│   ├── integrated_service.py  # 集成服务
│   ├── service_manager.py     # 服务管理器
│   └── README.md             # 详细文档
├── .env                      # 配置文件
└── test_storage.py          # 配置验证
```

## 🔧 配置选项

```bash
# 直接下载模式（推荐）
STORAGE_MODE=direct_download

# S3 存储模式
STORAGE_MODE=s3
S3_BUCKET_NAME=your-bucket
```

## 📚 更多信息

- [详细文档](./README.md)
- [部署指南](./DEPLOYMENT_GUIDE.md)
