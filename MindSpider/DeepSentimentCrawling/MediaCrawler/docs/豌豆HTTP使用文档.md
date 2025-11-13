## 豌豆HTTP代理使用文檔 （只支持企業用戶）

## 準備代理 IP 信息
點擊 <a href="https://h.wandouip.com?invite_code=rtnifi">豌豆HTTP代理</a> 官網註冊並實名認證（國內使用代理 IP 必須要實名，懂的都懂）

## 獲取 IP 代理的密鑰信息 appkey
從 <a href="https://h.wandouip.com?invite_code=rtnifi">豌豆HTTP代理</a> 官網獲取免費試用，如下圖所示
![img.png](static/images/wd_http_img.png)

選擇自己需要的套餐
![img_4.png](static/images/wd_http_img_4.png)


初始化一個豌豆HTTP代理的示例，如下代碼所示，需要1個參數： app_key

```python
# 文件地址： proxy/providers/wandou_http_proxy.py
# -*- coding: utf-8 -*-

def new_wandou_http_proxy() -> WanDouHttpProxy:
    """
    構造豌豆HTTP實例
    Returns:

    """
    return WanDouHttpProxy(
        app_key=os.getenv(
            "wandou_app_key", "你的豌豆HTTP app_key"
        ),  # 通過環境變量的方式獲取豌豆HTTP app_key
    )

```

在個人中心的`開放接口`找到 `app_key`，如下圖所示

![img_2.png](static/images/wd_http_img_2.png)


