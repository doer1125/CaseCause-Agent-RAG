## 1. 安装LlamaFactory

### 1.1 克隆LlamaFactory仓库

```bash
本机LlamaFactory路径：D:\workspace\Agent\LLaMA-Factory
cd LlamaFactory
```

### 1.2 安装依赖

```bash
项目环境在：conda activate CaseCause-RAG_Agent
pip install -e .[torch,metrics]

```

## 2. 数据准备

### 2.1 转换数据格式

LlamaFactory要求数据格式为特定的JSON格式，需要将`health_commission_data_1000_unique.json`转换为LlamaFactory支持的格式。

### 2.2 创建转换脚本

创建一个Python脚本将现有数据转换为LlamaFactory格式：

```python
import json

# 读取原始数据
with open('../health_commission_data_1000_unique.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 转换为LlamaFactory格式
llama_factory_data = []
for item in data:
    llama_factory_data.append({
        "conversations": [
            {
                "from": "human",
                "value": item["instruction"]
            },
            {
                "from": "assistant",
                "value": item["output"]
            }
        ]
    })

# 保存转换后的数据
with open('data/health_commission_data.json', 'w', encoding='utf-8') as f:
    json.dump(llama_factory_data, f, ensure_ascii=False, indent=2)
```

## 3. 配置微调参数

### 3.1 创建配置文件

在`configs`目录下创建`qwen2.5-7b-health.yaml`配置文件：

```yaml
# 基本配置
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
data_path: data/health_commission_data.json
dataset: health_commission_data
dataset_dir: data
output_dir: ../qwen2.5-7b-health-llamafactory

# 训练配置
training_type: lora
lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05

# 训练参数
train_batch_size: 2
eval_batch_size: 2
num_train_epochs: 3
learning_rate: 2e-5
warmup_ratio: 0.1
gradient_accumulation_steps: 8
max_seq_length: 1024

# 硬件配置
device_map: auto
fp16: true
gradient_checkpointing: true

# 优化器配置
optim: paged_adamw_8bit
lr_scheduler_type: cosine
weight_decay: 0.01
```

## 4. 执行微调命令

### 4.1 运行微调脚本

```bash
python src/train_bash.py --config configs/qwen2.5-7b-health.yaml
```

## 5. 验证微调结果

### 5.1 测试微调后的模型

```bash
python src/cli_demo.py --model_name_or_path ../qwen2.5-7b-health-llamafactory --adapter_name_or_path ../qwen2.5-7b-health-llamafactory
```

## 6. 注意事项

1. 确保GPU显存足够，批次大小设置为2可以有效避免显存溢出
2. 使用QLoRA微调可以大幅减少显存占用
3. 微调过程中可以通过TensorBoard监控训练进度
4. 微调完成后可以使用生成的适配器模型进行推理

## 7. 预期输出

* 微调后的模型保存在`../qwen2.5-7b-health-llamafactory`目录

* 包含适配器权重文件和配置文件

* 可以直接用于推理或进一步部署

