import {defineConfig} from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
    title: "MediaCrawler自媒體爬蟲",
    description: "小紅書爬蟲，抖音爬蟲， 快手爬蟲， B站爬蟲， 微博爬蟲，百度貼吧爬蟲，知乎爬蟲...。  ",
    lastUpdated: true,
    base: '/MediaCrawler/',
    head: [
        [
            'script',
            {async: '', src: 'https://www.googletagmanager.com/gtag/js?id=G-5TK7GF3KK1'}
        ],
        [
            'script',
            {},
            `window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-5TK7GF3KK1');`
        ]
    ],
    themeConfig: {
        editLink: {
            pattern: 'https://github.com/NanmiCoder/MediaCrawler/tree/main/docs/:path'
        },
        search: {
            provider: 'local'
        },
        // https://vitepress.dev/reference/default-theme-config
        nav: [
            {text: '首頁', link: '/'},
            {text: '聯繫我', link: '/作者介紹'},
            {text: '支持我', link: '/知識付費介紹'},
        ],

        sidebar: [
            {
                text: '作者介紹',
                link: '/作者介紹',
            },
            {
                text: 'MediaCrawler使用文檔',
                items: [
                    {text: '基本使用', link: '/'},
                    {text: '常見問題彙總', link: '/常見問題'},
                    {text: 'IP代理使用', link: '/代理使用'},
                    {text: '詞雲圖使用', link: '/詞雲圖使用配置'},
                    {text: '項目目錄結構', link: '/項目代碼結構'},
                    {text: '手機號登錄說明', link: '/手機號登錄說明'},
                ]
            },
            {
                text: '知識付費',
                items: [
                    {text: '知識付費介紹', link: '/知識付費介紹'},
                    {text: 'MediaCrawlerPro訂閱', link: '/mediacrawlerpro訂閱'},
                    {
                        text: 'MediaCrawler源碼剖析課',
                        link: 'https://relakkes.feishu.cn/wiki/JUgBwdhIeiSbAwkFCLkciHdAnhh'
                    },
                    {text: '知識星球文章專欄', link: '/知識星球介紹'},
                    {text: '開發者諮詢服務', link: '/開發者諮詢'},
                ]
            },
            {
                text: 'MediaCrawler項目交流羣',
                link: '/微信交流羣',
            },
            {
                text: '爬蟲入門教程分享',
                items: [
                    {text: "我寫的爬蟲入門教程", link: 'https://github.com/NanmiCoder/CrawlerTutorial'}
                ]
            },
            {
                text: 'MediaCrawler捐贈名單',
                items: [
                    {text: "捐贈名單", link: '/捐贈名單'}
                ]
            },

        ],

        socialLinks: [
            {icon: 'github', link: 'https://github.com/NanmiCoder/MediaCrawler'}
        ]
    }
})
