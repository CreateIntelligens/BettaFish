# 訂閱MediaCrawlerPro版本源碼訪問權限

## 獲取Pro版本的訪問權限
> MediaCrawler開源超過一年了，相信該倉庫幫過不少朋友低門檻的學習和了解爬蟲。維護真的耗費了大量精力和人力 <br>
> 
> 所以Pro版本不會開源，可以訂閱Pro版本讓我更加有動力去更新。<br>
> 
> 如果感興趣可以加我微信，訂閱Pro版本訪問權限哦，有門檻💰。<br>
> 
> 僅針對想學習Pro版本源碼實現的用戶，如果是公司或者商業化盈利性質的就不要加我了，謝謝🙏
> 
> 代碼設計拓展性強，可以自己擴展更多的爬蟲平臺，更多的數據存儲方式，相信對你架構這種爬蟲代碼有所幫助。
> 
> 
> **MediaCrawlerPro項目主頁地址**
> [MediaCrawlerPro Github主頁地址](https://github.com/MediaCrawlerPro)



掃描下方我的個人微信，備註：pro版本（如果圖片展示不出來，可以直接添加我的微信號：relakkes）

![relakkes_weichat.JPG](static/images/relakkes_weichat.jpg)


##  Pro版本誕生的背景
[MediaCrawler](https://github.com/NanmiCoder/MediaCrawler)這個項目開源至今獲得了大量的關注，同時也暴露出來了一系列問題，比如：
- 能否支持多賬號？
- 能否在linux部署？
- 能否去掉playwright的依賴？
- 有沒有更簡單的部署方法？
- 有沒有針對新手上門檻更低的方法？

諸如上面的此類問題，想要在原有項目上去動刀，無疑是增加了複雜度，可能導致後續的維護更加困難。
出於可持續維護、簡便易用、部署簡單等目的，對MediaCrawler進行徹底重構。

## 項目介紹
### [MediaCrawler](https://github.com/NanmiCoder/MediaCrawler)的Pro版本python實現
**小紅書爬蟲**，**抖音爬蟲**， **快手爬蟲**， **B站爬蟲**， **微博爬蟲**，**百度貼吧**，**知乎爬蟲**...。

支持多種平臺的爬蟲，支持多種數據的爬取，支持多種數據的存儲，最重要的**完美支持多賬號+IP代理池，讓你的爬蟲更加穩定**。
相較於MediaCrawler，Pro版本最大的變化：
- 去掉了playwright的依賴，不再將Playwright集成到爬蟲主幹中，依賴過重。
- 增加了Docker，Docker-compose的方式部署，讓部署更加簡單。
- 多賬號+IP代理池的支持，讓爬蟲更加穩定。
- 新增簽名服務，解耦簽名邏輯，讓爬蟲更加靈活。
