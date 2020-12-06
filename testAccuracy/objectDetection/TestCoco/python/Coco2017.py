from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from progressbar import progressbar



if __name__ == "__main__":

    #coco Ground Truth
    cocoGt = COCO("instances_val2017.json")
    
    #Detection results
    cocoDt = cocoGt.loadRes("yoloVal.json")

    imgIds = sorted(cocoGt.getImgIds())
    
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    print(cocoEval.summarize())
