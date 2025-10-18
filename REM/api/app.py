from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
from model import RelationPredictor
from REM.config import Config
import uvicorn
import os
from typing import Optional

app = FastAPI(title="关系抽取API", version="1.0.0")

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 初始化模板
templates = Jinja2Templates(directory="templates")

# 全局变量存储预测器
predictor = None


class RelationRequest(BaseModel):
    text: str
    entity1: str
    entity2: str


class RelationResponse(BaseModel):
    relation: str
    confidence: float
    text: str
    entity1: str
    entity2: str


def load_model():
    """加载模型"""
    global predictor
    try:
        config = Config()
        model_path = f"{config.MODEL_DIR}/checkpoint/model.pth"

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"警告: 模型文件不存在 {model_path}，请先训练模型")
            return False

        predictor = RelationPredictor(model_path, config)
        print("模型加载成功!")
        return True
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    print("正在加载模型...")
    if not load_model():
        print("模型加载失败，API将无法正常工作")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """返回Web界面"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_model=RelationResponse)
async def predict_relation(request: RelationRequest):
    """预测两个实体之间的关系"""
    if predictor is None:
        raise HTTPException(status_code=500, detail="模型未加载")

    try:
        result = predictor.predict(
            request.text,
            request.entity1,
            request.entity2
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查"""
    status = "healthy" if predictor is not None else "model_not_loaded"
    return {"status": status}


@app.get("/relations")
async def get_relation_types():
    """获取支持的关系类型"""
    config = Config()
    return {"relations": config.RELATION_TYPES}


@app.post("/reload_model")
async def reload_model():
    """重新加载模型"""
    if load_model():
        return {"status": "success", "message": "模型重新加载成功"}
    else:
        raise HTTPException(status_code=500, detail="模型重新加载失败")


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)