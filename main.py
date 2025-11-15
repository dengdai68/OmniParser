# pip install fastapi uvicorn python-multipart pillow requests

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import base64
from typing import Optional
import numpy as np
import logging
import traceback
import sys

from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

# 配置详细的日志记录
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# 初始化模型
logger.info("正在初始化模型...")
yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
logger.info("YOLO模型加载完成")
caption_model_processor = get_caption_model_processor(
    model_name="florence2",
    model_name_or_path="weights/icon_caption_florence"
)
logger.info("Caption模型加载完成")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {DEVICE}")

@app.post("/process_image")
async def process_image(
    file: UploadFile = File(...),
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1,
    use_paddleocr: bool = True,
    imgsz: int = 640
):
    try:
        logger.info("="*80)
        logger.info(f"收到图片处理请求: filename={file.filename}")
        logger.info(f"参数: box_threshold={box_threshold}, iou_threshold={iou_threshold}, use_paddleocr={use_paddleocr}, imgsz={imgsz}")

        # 读取上传的图片
        logger.debug("正在读取上传的图片...")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        logger.info(f"图片读取成功: size={image.size}, mode={image.mode}")

        # 临时保存图片
        image_save_path = 'imgs/temp_image.png'
        image.save(image_save_path)
        logger.debug(f"图片已保存到: {image_save_path}")

        # 配置绘制边界框的参数
        box_overlay_ratio = image.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        logger.debug(f"边界框配置: {draw_bbox_config}")

        # OCR处理
        logger.info("开始OCR处理...")
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            image_save_path,
            display_img=False,
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold': 0.9},
            use_paddleocr=use_paddleocr
        )
        text, ocr_bbox = ocr_bbox_rslt
        logger.info(f"OCR处理完成: 检测到 {len(text) if text else 0} 个文本区域")
        logger.debug(f"OCR文本: {text}")
        logger.debug(f"OCR边界框类型: {type(ocr_bbox)}, 内容: {ocr_bbox}")

        # 获取标记后的图片和解析内容
        logger.info("开始生成标记图片...")
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image_save_path,
            yolo_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption_model_processor,
            ocr_text=text,
            iou_threshold=iou_threshold,
            imgsz=imgsz,
        )
        logger.info(f"标记图片生成完成: 检测到 {len(parsed_content_list)} 个元素")
        logger.debug(f"标签坐标类型: {type(label_coordinates)}")
        logger.debug(f"解析内容列表: {parsed_content_list}")

        # 格式化解析结果
        parsed_content = '\n'.join([f'icon {i}: {str(v)}' for i, v in enumerate(parsed_content_list)])
        logger.info("图片处理成功完成")

        return JSONResponse({
            "status": "success",
            "labeled_image": dino_labled_img,  # base64编码的图片
            "parsed_content": parsed_content,
            "label_coordinates": label_coordinates
        })

    except Exception as e:
        # 打印详细的错误信息和堆栈跟踪
        logger.error("="*80)
        logger.error("处理图片时发生错误!")
        logger.error(f"错误类型: {type(e).__name__}")
        logger.error(f"错误信息: {str(e)}")
        logger.error("完整堆栈跟踪:")
        logger.error(traceback.format_exc())
        logger.error("="*80)

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("="*80)
    logger.info("启动 OmniParser FastAPI 服务器")
    logger.info("监听地址: 0.0.0.0:7861")
    logger.info("="*80)
    uvicorn.run(app, host="0.0.0.0", port=7861)