import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os

# 确保CUDA可用
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 设置环境变量
os.environ["WANDB_DISABLED"] = "true"

# 1. 加载模型和分词器
def load_model_tokenizer():
    print("Loading model and tokenizer...")
    
    model_name = "Qwen/Qwen2.5-7B"
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # 使用BitsAndBytesConfig配置4-bit量化
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    # 加载模型，使用4-bit量化
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="eager",  # 使用默认注意力机制，避免FlashAttention2依赖
        offload_buffers=True,  # 启用缓冲区卸载，节省内存
        offload_folder="./offload",  # 设置卸载文件夹
        torch_dtype=torch.bfloat16
    )
    
    # 设置模型配置
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.config.gradient_checkpointing = True
    
    return model, tokenizer

# 2. 配置QLoRA
def configure_lora(model):
    print("Configuring LoRA...")
    
    lora_config = LoraConfig(
        r=16,  # LoRA秩，控制参数数量
        lora_alpha=32,  # LoRA缩放因子
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head"
        ],  # 目标模块
        lora_dropout=0.05,  # Dropout率
        bias="none",  # 不训练偏置
        task_type="CAUSAL_LM"  # 因果语言模型任务
    )
    
    # 应用LoRA配置
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    return model

# 3. 加载和处理数据集
def load_and_preprocess_data(tokenizer, data_path="health_commission_data.json"):
    print(f"Loading dataset from {data_path}...")
    
    # 加载数据集
    dataset = load_dataset("json", data_files=data_path)
    
    # 预处理函数
    def preprocess_function(examples):
        # 构建指令格式
        messages = []
        for instr, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            if input_text:
                user_content = f"{instr}\n{input_text}"
            else:
                user_content = instr
            
            message = [
                {"role": "system", "content": "你是一名专业的卫健委公文撰写专家，请严格按照公文规范格式和术语要求撰写文书。"},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output}
            ]
            messages.append(message)
        
        # 应用聊天模板
        texts = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # 编码文本
        encoding = tokenizer(
            texts,
            truncation=True,
            max_length=1024,  # 减小最大序列长度，降低显存占用
            padding="max_length",
            return_tensors="pt"
        )
        
        # 设置标签（与输入相同，用于自回归训练）
        encoding["labels"] = encoding["input_ids"].clone()
        
        return encoding
    
    # 预处理数据集
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_dataset

# 4. 设置训练参数
def setup_training_args():
    training_args = TrainingArguments(
        output_dir="./qwen2.5-7b-health",  # 输出目录
        per_device_train_batch_size=1,  # 每个设备的批次大小，降低以减少显存占用
        gradient_accumulation_steps=8,  # 梯度累积步数，增加以保持有效批次大小
        learning_rate=2e-5,  # 学习率
        num_train_epochs=3,  # 训练轮次
        weight_decay=0.01,  # 权重衰减
        warmup_ratio=0.1,  # 预热比例
        logging_steps=10,  # 日志记录步数
        save_strategy="epoch",  # 按轮次保存模型
        fp16=True,  # 使用16位浮点训练
        bf16=False,  # 不使用bf16
        gradient_checkpointing=True,  # 启用梯度检查点，节省内存但减慢速度
        optim="paged_adamw_8bit",  # 优化器
        lr_scheduler_type="cosine",  # 学习率调度器
        report_to="none"  # 不报告到外部服务
    )
    
    return training_args

# 5. 训练模型
def train_model():
    # 加载模型和分词器
    model, tokenizer = load_model_tokenizer()
    
    # 配置LoRA
    model = configure_lora(model)
    
    # 加载和预处理数据
    tokenized_dataset = load_and_preprocess_data(tokenizer)
    
    # 设置训练参数
    training_args = setup_training_args()
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
        data_collator=None  # 使用默认的数据收集器
    )
    
    # 开始训练
    print("Starting training...")
    trainer.train()
    
    # 保存模型
    print("Saving model...")
    model.save_pretrained("./qwen2.5-7b-health-final")
    tokenizer.save_pretrained("./qwen2.5-7b-health-final")
    
    print("Training completed!")

# 6. 测试模型
def test_model():
    from peft import AutoPeftModelForCausalLM
    
    print("Testing fine-tuned model...")
    
    # 加载微调后的模型
    model = AutoPeftModelForCausalLM.from_pretrained(
        "./qwen2.5-7b-health-final",
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "./qwen2.5-7b-health-final",
        trust_remote_code=True
    )
    
    # 测试指令
    test_instruction = "撰写一份关于开展医疗机构专项检查的通知"
    
    # 构建聊天模板
    messages = [
        {"role": "system", "content": "你是一名专业的卫健委公文撰写专家，请严格按照公文规范格式和术语要求撰写文书。"},
        {"role": "user", "content": test_instruction}
    ]
    
    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 编码文本
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to("cuda")
    
    # 生成文本
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1
    )
    
    # 解码并输出结果
    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    
    print("\nTest Result:")
    print("=" * 50)
    print(generated_text)
    print("=" * 50)

# 主函数
if __name__ == "__main__":
    try:
        train_model()
        test_model()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
