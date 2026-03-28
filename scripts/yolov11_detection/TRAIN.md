
    --hsv-h
        HSV hue jitter strength. Useful for small color variation, but keep it low
        when class identity depends on subtle color cues.
        HSV 色相抖動強度。適合少量顏色變化，但如果類別差異仰賴細微色彩，
        建議不要設太高。

    --hsv-s
        HSV saturation jitter strength. Helps cover different lighting/material
        conditions, but too much can make classes look unrealistic.
        HSV 飽和度抖動強度。可模擬不同光線或材質狀態，但過強會讓物件外觀失真。

    --hsv-v
        HSV value/brightness jitter strength. Commonly useful for indoor datasets
        with changing exposure or illumination.
        HSV 明度/亮度抖動強度。對室內場景或曝光變化較大的資料集通常很有用。

    --degrees
        Rotation augmentation in degrees. Good for viewpoint variation; keep it
        conservative if objects are usually upright.
        旋轉擴增角度。適合視角變化較多的情境；若物件大多維持正向，建議保守設定。

    --translate
        Translation ratio. Simulates object shift inside the frame.
        平移比例。用來模擬物件在畫面中的位置偏移。

    --scale
        Scale augmentation ratio. Useful when target objects appear at different
        distances or sizes.
        縮放比例。適合物件會以不同距離、不同大小出現的情況。

    --shear
        Shear augmentation in degrees. Can help with perspective-like distortion,
        but large values may create unrealistic shapes.
        剪切角度。可模擬部分透視變形，但數值過大會造成不自然的物件形狀。

    --perspective
        Perspective augmentation ratio. Usually keep this small unless your camera
        viewpoints vary a lot.
        透視變形比例。除非相機視角變化很大，否則通常建議維持較小數值。

    --flipud
        Vertical flip probability. Only use when upside-down objects are realistic.
        上下翻轉機率。只有在物件上下顛倒仍合理時才建議使用。

    --fliplr
        Horizontal flip probability. Often safe for many detection tasks.
        左右翻轉機率。對多數物件偵測任務通常是相對安全的擴增。

    --mosaic
        Mosaic augmentation probability. Can help small-object detection, but too
        much may make the training distribution less realistic.
        Mosaic 擴增機率。對小物件偵測常有幫助，但太高可能讓訓練分布偏離真實場景。

    --mixup
        MixUp augmentation probability. Sometimes useful for regularization, but
        often kept low or disabled for cleaner detection labels.
        MixUp 擴增機率。有時可增加正則化效果，但為了保持 detection label 清楚，
        通常會設低一點或直接關閉。

