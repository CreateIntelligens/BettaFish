# 貢獻指南

感謝你願意爲本項目做出貢獻！
爲了保持代碼質量和版本管理的清晰，請按照以下步驟提交你的修改。

# 🪄 提交 Pull Request（PR）步驟

## 1️⃣ Fork 倉庫

將本倉庫 Fork 到你的 GitHub 賬戶。

## 2️⃣ 克隆到本地

```bash
git clone https://github.com/<你的用戶名>/<倉庫名>.git
cd <倉庫名>
```

## 3️⃣ 創建功能分支

```bash
git checkout -b feature/你的功能名
```

> 建議分支命名規範：`feature/xxx` 或 `fix/xxx`，便於識別功能或修復類型。

## 4️⃣ 開發與測試

* 進行代碼修改，保持項目代碼風格一致。
* 確保新增功能或修復通過測試。

## 5️⃣ 提交修改

```bash
git add .
git commit -m "類型: 簡短描述"
```

> 推薦遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/)，保持提交記錄清晰。

## 6️⃣ 推送到遠程倉庫

```bash
git push origin feature/你的功能名
```

## 7️⃣ 發起 Pull Request

1. 在 GitHub 上點擊 **New Pull Request**。
2. **目標分支必須是本倉庫的 `main` 分支**。
3. 填寫 PR 描述：

   * 說明主要改動內容
   * 如有相關 issue，請在 PR 中關聯
