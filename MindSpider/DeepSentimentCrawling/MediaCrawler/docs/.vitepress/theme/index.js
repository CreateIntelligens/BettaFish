// .vitepress/theme/index.js
import DefaultTheme from 'vitepress/theme'
import MyLayout from './MyLayout.vue'

export default {
  extends: DefaultTheme,
  // 使用注入插槽的包裝組件覆蓋 Layout
  Layout: MyLayout
}